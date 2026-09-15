"""Synthetic + real dataset construction for the workload classifier.

Two ways to get training data:

1. `generate_synthetic(...)`  -> reproducible labelled data based on the known
   behavioural signatures of each workload class. Lets anyone retrain the model
   with no data-collection step. Good for reproducibility in the report.

2. `load_csv(path)` -> load a real trace you collected on Linux with
   `trainos collect`. Preferred for the final results.

Class signatures (behaviour-based, scheduler-relevant):

    cpu_bound     : high CPU, low variance, low I/O wait, high stability
    io_bound      : low/medium CPU, high I/O wait, bursty, lower stability
    interactive   : bursty CPU (high variance), short/idle periods, low I/O wait
"""

from __future__ import annotations

import csv
import random
from typing import List, Tuple

from . import FEATURE_NAMES, WORKLOAD_CLASSES


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _sample_cpu_bound(rng: random.Random) -> List[float]:
    cpu = rng.gauss(88, 6)
    return [
        _clip(cpu),                          # cpu_usage
        _clip(rng.gauss(8, 4), 0, 60),       # cpu_variance (low)
        _clip(rng.gauss(45, 20)),            # memory_usage
        _clip(rng.gauss(0.03, 0.03), 0, 1),  # io_wait (low)
        _clip(rng.gauss(120, 80), 0, 3600),  # runtime (longer jobs)
        _clip(rng.gauss(0.9, 0.08), 0, 1),   # throughput_stability (high)
    ]


def _sample_io_bound(rng: random.Random) -> List[float]:
    cpu = rng.gauss(22, 12)
    return [
        _clip(cpu),
        _clip(rng.gauss(30, 12), 0, 100),    # medium variance
        _clip(rng.gauss(35, 18)),
        _clip(rng.gauss(0.55, 0.18), 0, 1),  # io_wait (high)
        _clip(rng.gauss(90, 70), 0, 3600),
        _clip(rng.gauss(0.45, 0.18), 0, 1),  # bursty -> lower stability
    ]


def _sample_interactive(rng: random.Random) -> List[float]:
    cpu = rng.gauss(18, 15)
    return [
        _clip(cpu),
        _clip(rng.gauss(55, 18), 0, 100),    # high variance (bursty)
        _clip(rng.gauss(25, 15)),
        _clip(rng.gauss(0.1, 0.08), 0, 1),   # low io_wait
        _clip(rng.gauss(30, 25), 0, 3600),   # shorter lived / idle
        _clip(rng.gauss(0.4, 0.2), 0, 1),
    ]


_SAMPLERS = {
    "cpu_bound": _sample_cpu_bound,
    "io_bound": _sample_io_bound,
    "interactive": _sample_interactive,
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
