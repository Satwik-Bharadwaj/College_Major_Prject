"""A real ML training workload (the job TrainOS is meant to protect).

This runs an actual iterative scikit-learn training loop -- repeatedly fitting
a model on a synthetic dataset for a number of rounds. It is genuinely
CPU+memory heavy and sustained, so it exhibits the `heavy_compute` signature
and stands in for "the demanding job you want to run on a modest machine".

Crucially it reports its own progress so an evaluation harness can measure it:
each completed round it appends `round,elapsed_s` to a progress file, and on
exit it writes a small JSON summary (rounds, wall time, rounds/sec). That gives
us a real throughput/completion measurement rather than a guess.

    python -m trainos.workloads.ml_train --rounds 40 --progress /tmp/ml.progress
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=40,
                   help="number of training rounds (0 = until killed)")
    p.add_argument("--samples", type=int, default=4000, help="dataset rows")
    p.add_argument("--features", type=int, default=40, help="dataset columns")
    p.add_argument("--progress", help="append 'round,elapsed' lines here")
    p.add_argument("--summary", help="write JSON summary here on exit")
    return p.parse_args()


def _set_name():
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle("trainos-ml-train")
    except Exception:
        pass


def main() -> int:
    args = _parse()
    _set_name()
    pid = os.getpid()
    print(f"[workload:ml_train] pid={pid} rounds={args.rounds}. Stop with: kill {pid}",
          flush=True)

    # Build the dataset once; refit repeatedly to create a sustained load.
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=max(2, args.features // 2),
        random_state=0,
    )

    if args.progress:
        # truncate any stale progress file
        try:
            open(args.progress, "w").close()
        except OSError:
            pass

    start = time.time()
    rounds_done = 0
    try:
        while args.rounds == 0 or rounds_done < args.rounds:
            clf = RandomForestClassifier(
                n_estimators=60, max_depth=None, n_jobs=1, random_state=rounds_done
            )
            clf.fit(X, y)
            clf.predict(X)  # a little inference too
            rounds_done += 1
            elapsed = time.time() - start
            if args.progress:
                try:
                    with open(args.progress, "a") as fh:
                        fh.write(f"{rounds_done},{elapsed:.4f}\n")
                except OSError:
                    pass
    except KeyboardInterrupt:
        pass

    wall = time.time() - start
    rate = rounds_done / wall if wall > 0 else 0.0
    summary = {
        "pid": pid,
        "rounds_completed": rounds_done,
        "wall_seconds": round(wall, 4),
        "rounds_per_sec": round(rate, 4),
    }
    print(f"[workload:ml_train] done {summary}", flush=True)
    if args.summary:
        try:
            with open(args.summary, "w") as fh:
                json.dump(summary, fh)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
