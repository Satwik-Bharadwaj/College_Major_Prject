"""A/B evaluation: does TrainOS help the heavy job on a busy machine?

This runs the real experiment behind the project's goal -- "run a demanding
job (ML training) well on a machine that is not built for it" -- and measures
it with actual processes and real scheduling controls, not a simulation.

Experiment (repeated once per phase):

  1. Spawn a crowd of ordinary background jobs (the generator) to saturate the
     machine, mimicking what a non-ML box is normally doing.
  2. Spawn ONE tracked ML training job (trainos.workloads.ml_train) that
     reports its own rounds completed and wall time.
  3. PHASE "baseline": let it run with the default OS scheduler (TrainOS off).
     PHASE "trainos": flag the ML job as heavy_compute and run the TrainOS
     control loop alongside it so it is boosted and the background jobs are
     de-prioritised.
  4. Record the ML job's completion time and throughput (rounds/sec).

Then compare: lower completion time / higher rounds-per-sec under TrainOS means
the heavy job got more effective CPU because TrainOS cleared the background
churn out of its way.

Honesty notes:
  * This measures a REAL effect only on Linux, where nice/affinity actually
    change scheduling. Off Linux the enforcer only simulates, so the two phases
    will look the same -- run the evaluation on the target Ubuntu box.
  * Boosting to negative nice / applying affinity needs root; run under sudo
    for the trainos phase to see the full effect. Without root TrainOS can
    still de-prioritise background jobs (positive nice), which is the larger
    lever here, so partial effect is visible without root too.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional

from . import override
from .workloads import generator


def _python() -> str:
    return sys.executable or "python3"


def _spawn_ml_job(rounds: int, progress: str, summary: str) -> subprocess.Popen:
    return subprocess.Popen(
        [_python(), "-m", "trainos.workloads.ml_train",
         "--rounds", str(rounds), "--progress", progress, "--summary", summary],
    )


def _read_summary(path: str) -> Optional[Dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _run_trainos_loop(ml_proc: subprocess.Popen, interval: float, dry_run: bool):
    """Run the TrainOS control loop until the tracked ML process exits.

    We poll the Popen object (not os.kill) so a finished-but-unreaped zombie
    correctly counts as "done" and the loop terminates.
    """
    from .model import Classifier, DEFAULT_MODEL_PATH
    from .scheduler import TrainOSScheduler

    clf = Classifier(DEFAULT_MODEL_PATH)
    sched = TrainOSScheduler(
        classifier=clf,
        interval=interval,
        dry_run=dry_run,
        min_runtime=0.0,
        verbose=False,
    )
    sched.collector.sample()
    time.sleep(min(1.0, interval))
    while ml_proc.poll() is None:  # None => still running
        sched.tick()
        if ml_proc.poll() is not None:
            break
        time.sleep(interval)
    return sched.stats


def _phase(name: str, *, count: int, bg_duration: float, rounds: int,
           interval: float, use_trainos: bool, dry_run: bool) -> Dict:
    print(f"\n=== PHASE: {name} ===")
    workdir = tempfile.mkdtemp(prefix=f"trainos_eval_{name}_")
    progress = os.path.join(workdir, "ml.progress")
    summary = os.path.join(workdir, "ml.summary.json")

    print(f"  spawning {count} background jobs ...")
    generator.spawn_background(count=count, duration=bg_duration, seed=1)
    time.sleep(2.0)  # let background load ramp up

    print("  starting tracked ML training job ...")
    ml = _spawn_ml_job(rounds=rounds, progress=progress, summary=summary)

    if use_trainos:
        override.set_pid(ml.pid, "heavy_compute")
        print(f"  flagged ML pid {ml.pid} as heavy_compute; running TrainOS loop ...")
        _run_trainos_loop(ml, interval=interval, dry_run=dry_run)
    else:
        print("  running under default OS scheduler (TrainOS off) ...")

    ml.wait()
    result = _read_summary(summary) or {}

    # cleanup
    stopped = generator.stop_all()
    if use_trainos:
        override.clear_pid(ml.pid)
    try:
        os.remove(progress); os.remove(summary); os.rmdir(workdir)
    except OSError:
        pass
    print(f"  {name}: {result}  (stopped {stopped} bg jobs)")
    return result


def run_evaluation(count: int, bg_duration: float, rounds: int,
                   interval: float, dry_run: bool) -> Dict:
    baseline = _phase("baseline", count=count, bg_duration=bg_duration,
                      rounds=rounds, interval=interval,
                      use_trainos=False, dry_run=dry_run)
    trainos = _phase("trainos", count=count, bg_duration=bg_duration,
                     rounds=rounds, interval=interval,
                     use_trainos=True, dry_run=dry_run)

    def pct(a, b):  # improvement of b over a
        return round((a - b) / a * 100.0, 2) if a else 0.0

    b_wall = baseline.get("wall_seconds", 0.0)
    t_wall = trainos.get("wall_seconds", 0.0)
    b_rate = baseline.get("rounds_per_sec", 0.0)
    t_rate = trainos.get("rounds_per_sec", 0.0)

    # throughput gain: how much higher trainos rounds/sec is vs baseline
    throughput_gain = round((t_rate - b_rate) / b_rate * 100.0, 2) if b_rate else 0.0

    comparison = {
        "baseline": baseline,
        "trainos": trainos,
        "completion_time_reduction_percent": pct(b_wall, t_wall),
        "throughput_gain_percent": throughput_gain,
    }
    return comparison


def _print_report(cmp: Dict) -> None:
    b = cmp["baseline"]; t = cmp["trainos"]
    print("\n" + "=" * 60)
    print("EVALUATION: ML training job under load (baseline vs TrainOS)")
    print("=" * 60)
    print(f"{'metric':32s} {'baseline':>12s} {'trainos':>12s}")
    print("-" * 60)
    print(f"{'rounds completed':32s} {b.get('rounds_completed',0):>12} "
          f"{t.get('rounds_completed',0):>12}")
    print(f"{'wall seconds (lower better)':32s} {b.get('wall_seconds',0):>12} "
          f"{t.get('wall_seconds',0):>12}")
    print(f"{'rounds/sec (higher better)':32s} {b.get('rounds_per_sec',0):>12} "
          f"{t.get('rounds_per_sec',0):>12}")
    print("-" * 60)
    print(f"completion time reduction: "
          f"{cmp['completion_time_reduction_percent']:+.2f}%  "
          f"(positive = TrainOS finished the ML job faster)")
    print("\nNote: real scheduling effects require Linux + (ideally) root. "
          "Off Linux the phases will look similar by design.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trainos evaluate",
                                description="A/B: ML job under load, TrainOS off vs on")
    p.add_argument("--count", type=int, default=40,
                   help="background jobs per phase (raise to 100s on a real box)")
    p.add_argument("--bg-duration", type=float, default=120.0,
                   help="approx seconds background jobs live (should outlast the ML job)")
    p.add_argument("--rounds", type=int, default=15,
                   help="ML training rounds (the measured heavy job)")
    p.add_argument("--interval", type=float, default=2.0,
                   help="TrainOS control-loop interval (trainos phase)")
    p.add_argument("--dry-run", action="store_true",
                   help="run TrainOS loop without applying changes (sanity only)")
    args = p.parse_args(argv)

    cmp = run_evaluation(
        count=args.count, bg_duration=args.bg_duration, rounds=args.rounds,
        interval=args.interval, dry_run=args.dry_run,
    )
    _print_report(cmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
