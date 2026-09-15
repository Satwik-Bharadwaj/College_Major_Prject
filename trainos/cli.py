"""TrainOS command line interface.

Subcommands:
  train      Train + select the best model on synthetic (or real) data.
  collect    Collect a real labelled/unlabelled trace on this machine.
  run        Start the adaptive scheduler (use --dry-run first!).
  benchmark  Compare default scheduler vs TrainOS (simulation).
  selftest   End-to-end sanity check that everything works.
  info       Show environment + model status.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from . import __version__, FEATURE_NAMES
from . import dataset as ds

# Default model path is needed by the argparse defaults, but importing the full
# model module (sklearn/joblib) should be deferred until a command actually
# needs it. Keep the parser importable in minimal environments.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "artifacts", "trainos_rf.joblib"
)


def _cmd_train(args):
    from . import model as m
    if args.csv:
        print(f"Loading dataset from {args.csv}")
        X, y = ds.load_csv(args.csv)
    else:
        print(f"Generating synthetic dataset ({args.n} per class)")
        X, y = ds.generate_synthetic(n_per_class=args.n, seed=args.seed)
        if args.save_dataset:
            ds.save_csv(args.save_dataset, X, y)
            print(f"Saved dataset -> {args.save_dataset}")

    print("Training candidate models and selecting best by macro-F1 ...")
    best, report = m.train_and_select(X, y)
    m.save(best, report, args.out)

    print("\n=== Model comparison ===")
    for name, sc in report["comparison"].items():
        marker = "  <== chosen" if name == report["chosen_model"] else ""
        print(f"  {name:16s} acc={sc['accuracy']:.4f} f1={sc['macro_f1']:.4f}{marker}")

    if "feature_importances" in report:
        print("\n=== Feature importances (leakage sanity check) ===")
        for k, v in sorted(report["feature_importances"].items(),
                           key=lambda kv: -kv[1]):
            print(f"  {k:22s} {v:.4f}")
        top = max(report["feature_importances"].values())
        if top > 0.85:
            print("  WARNING: one feature dominates (>0.85). Possible label leakage.")

    print(f"\nSaved model -> {args.out}")
    print(f"Saved report -> {args.out}.report.json")


def _cmd_collect(args):
    from .features import MetricCollector
    print(f"Collecting {args.ticks} samples every {args.interval}s ...")
    col = MetricCollector(window=args.window)
    col.sample()
    time.sleep(1.0)
    rows = []
    for _ in range(args.ticks):
        col.sample()
        for pid, tr in col.ready_trackers().items():
            f = tr.features()
            rows.append((tr.name, f))
        time.sleep(args.interval)

    import csv
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name"] + list(FEATURE_NAMES))
        for name, f in rows:
            w.writerow([name] + [f[n] for n in FEATURE_NAMES])
    print(f"Wrote {len(rows)} samples -> {args.out}")
    print("Label the 'label' column manually or with a rule, then `train --csv`.")


def _cmd_run(args):
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run `trainos train` first.",
              file=sys.stderr)
        return 1
    from .model import Classifier
    from .scheduler import TrainOSScheduler

    clf = Classifier(args.model)  # noqa: F841 (used below)
    sched = TrainOSScheduler(
        classifier=clf,
        interval=args.interval,
        window=args.window,
        min_runtime=args.min_runtime,
        dry_run=args.dry_run,
        manage_affinity=args.affinity,
        verbose=not args.quiet,
    )
    mode = "DRY-RUN (no changes)" if args.dry_run else "LIVE"
    if platform.system() != "Linux" and not args.dry_run:
        mode += " [non-Linux: actions simulated]"
    print(f"TrainOS scheduler starting — mode: {mode}. Ctrl-C to stop.")
    stats = sched.run(max_ticks=args.ticks)
    print("\n=== Session summary ===")
    print(f"  ticks:            {stats.ticks}")
    print(f"  processes managed:{stats.processes_seen}")
    print(f"  actions applied:  {stats.actions_applied}")
    print(f"  class breakdown:  {stats.classifications}")
    return 0


def _cmd_benchmark(args):
    from .benchmark import run_simulation
    res = run_simulation()
    print("=== Default scheduler (fair share) ===")
    print(json.dumps(res["baseline_default_scheduler"], indent=2))
    print("\n=== TrainOS (class-aware) ===")
    print(json.dumps(res["trainos"], indent=2))
    print("\n=== Improvement (positive = TrainOS better) ===")
    for k, v in res["improvement_percent"].items():
        print(f"  {k:32s} {v:+.2f}%")


def _cmd_selftest(args):
    from . import model as m
    print("TrainOS self-test")
    print("-" * 40)
    ok = True

    # 1. dataset
    X, y = ds.generate_synthetic(n_per_class=500, seed=1)
    assert len(X) == 1500 and len(X[0]) == len(FEATURE_NAMES)
    print("[1/5] dataset generation           OK")

    # 2. training + selection
    best, report = m.train_and_select(X, y)
    acc = report["comparison"][report["chosen_model"]]["accuracy"]
    assert acc > 0.8, f"accuracy too low: {acc}"
    print(f"[2/5] train + select ({report['chosen_model']}, acc={acc:.3f})  OK")

    # 3. save + load
    tmp = os.path.join(os.path.dirname(__file__), "artifacts", "_selftest.joblib")
    m.save(best, report, tmp)
    clf = m.Classifier(tmp)
    pred = clf.predict(X[0])
    assert pred in ("cpu_bound", "io_bound", "interactive")
    print(f"[3/5] save/load + predict ({pred})   OK")

    # 4. metric collection on this machine
    from .features import MetricCollector
    col = MetricCollector(window=3)
    col.sample(); time.sleep(1.1); col.sample()
    n = len(col.ready_trackers())
    print(f"[4/5] live metric collection ({n} procs)  OK")

    # 5. one scheduler tick in dry-run
    from .scheduler import TrainOSScheduler
    sched = TrainOSScheduler(clf, dry_run=True, min_runtime=0.0, verbose=False)
    sched.collector.sample(); time.sleep(1.1)
    actions = sched.tick()
    print(f"[5/5] scheduler dry-run tick ({len(actions)} actions)  OK")

    os.remove(tmp)
    if os.path.exists(tmp + ".report.json"):
        os.remove(tmp + ".report.json")

    print("-" * 40)
    print("ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌")
    return 0 if ok else 1


def _cmd_info(args):
    print(f"TrainOS v{__version__}")
    print(f"  platform : {platform.system()} {platform.release()}")
    print(f"  python   : {platform.python_version()}")
    print(f"  cpus     : {os.cpu_count()}")
    model_ok = os.path.exists(_DEFAULT_MODEL_PATH)
    print(f"  model    : {'present' if model_ok else 'MISSING (run train)'}")
    print(f"  features : {', '.join(FEATURE_NAMES)}")
    if platform.system() != "Linux":
        print("  note     : scheduling actions are SIMULATED off Linux.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trainos", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train + select best model")
    t.add_argument("--csv", help="use a real labelled CSV instead of synthetic")
    t.add_argument("--n", type=int, default=4000, help="synthetic rows per class")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out", default=_DEFAULT_MODEL_PATH)
    t.add_argument("--save-dataset", help="also write the synthetic dataset here")
    t.set_defaults(func=_cmd_train)

    c = sub.add_parser("collect", help="collect a real trace on this machine")
    c.add_argument("--ticks", type=int, default=20)
    c.add_argument("--interval", type=float, default=2.0)
    c.add_argument("--window", type=int, default=5)
    c.add_argument("--out", default="trace.csv")
    c.set_defaults(func=_cmd_collect)

    r = sub.add_parser("run", help="run the adaptive scheduler")
    r.add_argument("--model", default=_DEFAULT_MODEL_PATH)
    r.add_argument("--interval", type=float, default=3.0)
    r.add_argument("--window", type=int, default=5)
    r.add_argument("--min-runtime", type=float, default=2.0)
    r.add_argument("--ticks", type=int, default=None, help="stop after N ticks")
    r.add_argument("--dry-run", action="store_true", help="observe only")
    r.add_argument("--affinity", action="store_true", help="also manage CPU affinity")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=_cmd_run)

    b = sub.add_parser("benchmark", help="default vs TrainOS (simulation)")
    b.set_defaults(func=_cmd_benchmark)

    s = sub.add_parser("selftest", help="end-to-end sanity check")
    s.set_defaults(func=_cmd_selftest)

    i = sub.add_parser("info", help="environment + model status")
    i.set_defaults(func=_cmd_info)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
