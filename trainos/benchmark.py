"""Baseline (default scheduler) vs TrainOS comparison.

Two modes:

  simulate  : a portable, reproducible event-driven simulation of a mixed
              workload under (a) round-robin/CFS-like fair sharing and
              (b) TrainOS class-aware prioritisation. Runs anywhere, used to
              demonstrate the *mechanism* and produce report numbers.

  live      : (Linux only) launches real stress processes, runs them once with
              the default scheduler and once under TrainOS, and compares wall
              clock + interactive latency. Requires stress-ng or falls back to
              python busy loops.

The simulation is intentionally simple and transparent so it can be explained
in a viva: interactive jobs value low latency, cpu_bound jobs value throughput.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Job:
    jid: int
    cls: str          # cpu_bound | io_bound | interactive
    work: float       # total work units required
    remaining: float = 0.0
    wait_time: float = 0.0
    completion: float = 0.0

    def __post_init__(self):
        self.remaining = self.work


def _make_jobs(seed: int = 7) -> List[Job]:
    rng = random.Random(seed)
    jobs = []
    jid = 0
    for _ in range(4):
        jobs.append(Job(jid, "cpu_bound", rng.uniform(80, 120))); jid += 1
    for _ in range(3):
        jobs.append(Job(jid, "io_bound", rng.uniform(30, 60))); jid += 1
    for _ in range(5):
        jobs.append(Job(jid, "interactive", rng.uniform(5, 15))); jid += 1
    return jobs


def _weights(class_aware: bool) -> Dict[str, float]:
    if class_aware:
        # TrainOS: boost interactive, de-prioritise cpu_bound (mirrors POLICY_TABLE)
        return {"interactive": 3.0, "io_bound": 1.5, "cpu_bound": 1.0}
    # default fair share: everyone equal
    return {"interactive": 1.0, "io_bound": 1.0, "cpu_bound": 1.0}


def _simulate(class_aware: bool, quantum: float = 1.0, capacity: float = 4.0):
    jobs = _make_jobs()
    weights = _weights(class_aware)
    t = 0.0
    done: List[Job] = []
    active = list(jobs)

    while active:
        total_w = sum(weights[j.cls] for j in active)
        for j in active:
            share = capacity * quantum * (weights[j.cls] / total_w)
            progressed = min(j.remaining, share)
            j.remaining -= progressed
        for j in active:
            if j.remaining > 1e-6:
                j.wait_time += quantum
        t += quantum
        finished = [j for j in active if j.remaining <= 1e-6]
        for j in finished:
            j.completion = t
        done.extend(finished)
        active = [j for j in active if j.remaining > 1e-6]
        if t > 100000:  # safety
            break

    return done, t


def _metrics(jobs: List[Job], makespan: float) -> Dict[str, float]:
    inter = [j.completion for j in jobs if j.cls == "interactive"]
    return {
        "makespan": round(makespan, 3),
        "avg_completion": round(sum(j.completion for j in jobs) / len(jobs), 3),
        "avg_interactive_completion": round(sum(inter) / len(inter), 3) if inter else 0.0,
        "max_interactive_completion": round(max(inter), 3) if inter else 0.0,
    }


def run_simulation() -> Dict[str, Dict[str, float]]:
    base_jobs, base_span = _simulate(class_aware=False)
    trainos_jobs, trainos_span = _simulate(class_aware=True)
    base = _metrics(base_jobs, base_span)
    trainos = _metrics(trainos_jobs, trainos_span)

    improvement = {
        k: round((base[k] - trainos[k]) / base[k] * 100.0, 2) if base[k] else 0.0
        for k in base
    }
    return {
        "baseline_default_scheduler": base,
        "trainos": trainos,
        "improvement_percent": improvement,
    }
