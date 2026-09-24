"""Illustrative simulation: default scheduler vs TrainOS for a heavy job.

This is a small, transparent, event-driven simulation of the target scenario:
one demanding `heavy_compute` job (e.g. an ML training run) sharing an
oversubscribed machine with many ordinary `normal` background jobs. It compares
how long the heavy job takes to finish under:

  (a) default fair sharing  -- every runnable job gets an equal CPU slice
  (b) TrainOS class-aware    -- the heavy job is boosted, background jobs yield

It exists to explain the *mechanism* and produce a quick illustrative number.
It is NOT a measurement of the real system: the weights mirror the policy
table, so the direction of the result is built in. For real, measured evidence
(actual processes, real scheduling controls, measured completion time and
throughput), use `trainos evaluate`, which runs the true A/B experiment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Job:
    jid: int
    cls: str          # heavy_compute | normal
    work: float       # total work units required
    remaining: float = 0.0
    wait_time: float = 0.0
    completion: float = 0.0

    def __post_init__(self):
        self.remaining = self.work


def _make_jobs(seed: int = 7) -> List[Job]:
    """One big heavy_compute job amid many smaller normal background jobs."""
    rng = random.Random(seed)
    jobs = []
    jid = 0
    # the demanding job we care about
    jobs.append(Job(jid, "heavy_compute", rng.uniform(280, 320))); jid += 1
    # a crowd of ordinary background jobs contending for the CPU
    for _ in range(20):
        jobs.append(Job(jid, "normal", rng.uniform(20, 80))); jid += 1
    return jobs


def _weights(class_aware: bool) -> Dict[str, float]:
    if class_aware:
        # TrainOS: boost heavy_compute, de-prioritise normal (mirrors POLICY_TABLE)
        return {"heavy_compute": 4.0, "normal": 1.0}
    # default fair share: everyone equal
    return {"heavy_compute": 1.0, "normal": 1.0}


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
    heavy = [j.completion for j in jobs if j.cls == "heavy_compute"]
    return {
        "makespan": round(makespan, 3),
        "avg_completion": round(sum(j.completion for j in jobs) / len(jobs), 3),
        "heavy_completion": round(sum(heavy) / len(heavy), 3) if heavy else 0.0,
        "max_heavy_completion": round(max(heavy), 3) if heavy else 0.0,
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
