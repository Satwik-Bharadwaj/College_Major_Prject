"""Workload generator: spawn many mixed background jobs at once.

Writing one script per job doesn't scale. This generator spawns N background
`normal` jobs -- drawn from the existing cpu/io/interactive workload profiles
with randomised durations and a small startup stagger -- to create realistic
contention on the machine (the "stuff a non-ML box is normally doing").

It is the *background load* for the evaluation. The demanding job it competes
with (the ML training run) is `trainos.workloads.ml_train`, spawned separately
by the evaluation harness so its progress can be measured.

    # spawn 100 mixed background jobs, each ~90s, from a Python API or CLI
    python -m trainos.workloads.generator --count 100 --duration 90
    python -m trainos.workloads.generator --stop

State (the spawned PIDs) is recorded so `--stop` can clean them all up.
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import subprocess
import sys
import tempfile
from typing import List, Tuple

# Background profiles. All of these classify as `normal` -- they are the
# ordinary, non-heavy workloads that should yield to the heavy job.
_PROFILES = ("cpu_bound", "io_bound", "interactive")

_STATE_FILE = os.path.join(tempfile.gettempdir(), "trainos_generator.pids")


def _python() -> str:
    return sys.executable or "python3"


def spawn_background(
    count: int,
    duration: float,
    mix: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    stagger: float = 0.02,
    seed: int = 0,
) -> List[Tuple[str, int]]:
    """Spawn `count` background jobs; return list of (profile, pid).

    `mix` is the (cpu, io, interactive) proportion. Durations are randomised
    around `duration` so jobs don't all finish simultaneously.
    """
    rng = random.Random(seed)
    cum = []
    acc = 0.0
    for w in mix:
        acc += w
        cum.append(acc)
    total = cum[-1] or 1.0

    spawned: List[Tuple[str, int]] = []
    for _ in range(count):
        r = rng.random() * total
        if r < cum[0]:
            profile = "cpu_bound"
        elif r < cum[1]:
            profile = "io_bound"
        else:
            profile = "interactive"

        # randomise duration +/-30% so completions spread out
        d = max(1.0, duration * rng.uniform(0.7, 1.3))
        proc = subprocess.Popen(
            [_python(), "-m", f"trainos.workloads.{profile}", "--duration", f"{d:.1f}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        spawned.append((profile, proc.pid))
        if stagger:
            import time
            time.sleep(stagger)

    _record(spawned)
    return spawned


def _record(spawned: List[Tuple[str, int]]) -> None:
    with open(_STATE_FILE, "w") as fh:
        for profile, pid in spawned:
            fh.write(f"{pid} {profile}\n")


def _read_state() -> List[Tuple[int, str]]:
    if not os.path.exists(_STATE_FILE):
        return []
    entries = []
    with open(_STATE_FILE) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                entries.append((int(parts[0]), parts[1]))
    return entries


def stop_all() -> int:
    entries = _read_state()
    stopped = 0
    for pid, _profile in entries:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped += 1
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.remove(_STATE_FILE)
    except OSError:
        pass
    return stopped


def count_alive() -> int:
    alive = 0
    for pid, _ in _read_state():
        try:
            os.kill(pid, 0)
            alive += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            alive += 1
    return alive


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trainos.workloads.generator")
    p.add_argument("--count", type=int, default=60, help="number of background jobs")
    p.add_argument("--duration", type=float, default=90.0,
                   help="approx seconds each job runs")
    p.add_argument("--cpu", type=float, default=0.5, help="cpu_bound proportion")
    p.add_argument("--io", type=float, default=0.25, help="io_bound proportion")
    p.add_argument("--interactive", type=float, default=0.25,
                   help="interactive proportion")
    p.add_argument("--stagger", type=float, default=0.02,
                   help="seconds between launches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stop", action="store_true", help="stop all generated jobs")
    p.add_argument("--status", action="store_true", help="how many are alive")
    args = p.parse_args(argv)

    if args.stop:
        n = stop_all()
        print(f"stopped {n} background job(s).")
        return 0
    if args.status:
        print(f"{count_alive()} background job(s) alive.")
        return 0

    print(f"spawning {args.count} background jobs "
          f"(cpu/io/interactive = {args.cpu}/{args.io}/{args.interactive}), "
          f"~{args.duration:.0f}s each ...")
    spawned = spawn_background(
        count=args.count,
        duration=args.duration,
        mix=(args.cpu, args.io, args.interactive),
        stagger=args.stagger,
        seed=args.seed,
    )
    print(f"spawned {len(spawned)} jobs. State: {_STATE_FILE}")
    print("stop them with:  python -m trainos.workloads.generator --stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
