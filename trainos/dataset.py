"""Synthetic + real dataset construction for the workload classifier.

Two ways to get training data:

1. `generate_synthetic(...)`  -> reproducible labelled data based on the known
   behavioural signatures of each workload class. Lets anyone retrain the model
   with no data-collection step. Good for reproducibility in the report.

2. `load_csv(path)` -> load a real trace you collected on Linux with
   `trainos collect`. Preferred for the final results.

Class signatures (behaviour-based, scheduler-relevant):

    heavy_compute : sustained high CPU, low variance, high memory, long
                    runtime, high throughput stability. This is what an ML
                    training run looks like -- and also what any other
                    sustained compute job looks like. We prioritise it.
    normal        : everything else. A broad mixture of ordinary background
                    behaviour -- bursty/interactive spikes, mostly-idle
                    daemons, and I/O-waiting jobs -- i.e. low-to-moderate
                    average CPU, higher variance, shorter or idle periods,
                    and/or notable I/O wait. We de-prioritise it under load.

The two classes are separated by real signal in the features (sustained-vs-
bursty CPU, memory footprint, runtime, stability), not by an artificial tag.
An ML job and a non-ML CPU job that both sustain the machine will *both* be
heavy_compute -- that is intentional and honest (see PRIORITY_CLASS): the
scheduler protects sustained heavy compute, of which ML training is the
target case.
"""

from __future__ import annotations

import csv
import random
from typing import List, Tuple

from . import FEATURE_NAMES, WORKLOAD_CLASSES


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _sample_heavy_compute(rng: random.Random) -> List[float]:
    """Sustained heavy compute: an ML training run or similar.

    High steady CPU, low variance, sizeable memory footprint, long-lived,
    and very stable throughput -- the process just keeps grinding.
    """
    cpu = rng.gauss(90, 5)
    return [
        _clip(cpu),                           # cpu_usage (high)
        _clip(rng.gauss(6, 3), 0, 60),        # cpu_variance (low -> steady)
        _clip(rng.gauss(55, 18)),             # memory_usage (high)
        _clip(rng.gauss(0.03, 0.03), 0, 1),   # io_wait (low)
        _clip(rng.gauss(300, 150), 0, 3600),  # runtime (long jobs)
        _clip(rng.gauss(0.92, 0.06), 0, 1),   # throughput_stability (high)
    ]


# `normal` is a MIXTURE of ordinary background behaviours. We sample from a
# few sub-profiles so the classifier learns "not sustained heavy compute"
# robustly rather than one narrow shape.

def _sample_normal_interactive(rng: random.Random) -> List[float]:
    """Bursty, latency-sensitive foreground app: high variance, low avg CPU."""
    return [
        _clip(rng.gauss(18, 15)),             # cpu_usage (low avg)
        _clip(rng.gauss(55, 18), 0, 100),     # cpu_variance (high -> bursty)
        _clip(rng.gauss(25, 15)),             # memory_usage (modest)
        _clip(rng.gauss(0.1, 0.08), 0, 1),    # io_wait (low)
        _clip(rng.gauss(30, 25), 0, 3600),    # short-lived / idle
        _clip(rng.gauss(0.4, 0.2), 0, 1),     # unstable
    ]


def _sample_normal_io(rng: random.Random) -> List[float]:
    """I/O-waiting job: low/medium CPU, high io_wait, bursty throughput."""
    return [
        _clip(rng.gauss(22, 12)),             # cpu_usage (low/medium)
        _clip(rng.gauss(30, 12), 0, 100),     # cpu_variance (medium)
        _clip(rng.gauss(30, 15)),             # memory_usage
        _clip(rng.gauss(0.55, 0.18), 0, 1),   # io_wait (high)
        _clip(rng.gauss(90, 70), 0, 3600),    # runtime
        _clip(rng.gauss(0.45, 0.18), 0, 1),   # lower stability
    ]


def _sample_normal_idle(rng: random.Random) -> List[float]:
    """Mostly-idle daemon: near-zero CPU, tiny footprint."""
    return [
        _clip(rng.gauss(3, 3)),               # cpu_usage (near idle)
        _clip(rng.gauss(5, 5), 0, 100),       # cpu_variance (low, but low cpu)
        _clip(rng.gauss(12, 8)),              # memory_usage (small)
        _clip(rng.gauss(0.08, 0.08), 0, 1),   # io_wait (low)
        _clip(rng.gauss(200, 150), 0, 3600),  # can be long-lived but idle
        _clip(rng.gauss(0.6, 0.25), 0, 1),
    ]


_NORMAL_SUBPROFILES = (
    _sample_normal_interactive,
    _sample_normal_io,
    _sample_normal_idle,
)


def _sample_heavy(rng: random.Random) -> List[float]:
    return _sample_heavy_compute(rng)


def _sample_normal(rng: random.Random) -> List[float]:
    # pick one of the background sub-profiles at random
    return rng.choice(_NORMAL_SUBPROFILES)(rng)


_SAMPLERS = {
    "heavy_compute": _sample_heavy,
    "normal": _sample_normal,
}


def generate_synthetic(
    n_per_class: int = 4000, seed: int = 42
) -> Tuple[List[List[float]], List[str]]:
    """Return (X, y) with roughly balanced classes."""
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[str] = []
    for cls in WORKLOAD_CLASSES:
        sampler = _SAMPLERS[cls]
        for _ in range(n_per_class):
            X.append(sampler(rng))
            y.append(cls)
    # shuffle together
    idx = list(range(len(X)))
    rng.shuffle(idx)
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]
    return X, y


def save_csv(path: str, X: List[List[float]], y: List[str]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(FEATURE_NAMES) + ["label"])
        for row, label in zip(X, y):
            writer.writerow(list(row) + [label])


def load_csv(path: str) -> Tuple[List[List[float]], List[str]]:
    X: List[List[float]] = []
    y: List[str] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            X.append([float(r[name]) for name in FEATURE_NAMES])
            y.append(r["label"])
    return X, y
