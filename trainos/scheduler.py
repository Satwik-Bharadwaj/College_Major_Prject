"""The TrainOS control loop: observe -> classify -> act, repeat.

This realises Figure 3.1 of the report as a running daemon.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import override
from .features import MetricCollector
from .model import Classifier
from .policy import PolicyEnforcer, Action


@dataclass
class LoopStats:
    ticks: int = 0
    processes_seen: int = 0
    classifications: Dict[str, int] = field(default_factory=dict)
    actions_applied: int = 0

    def record(self, actions: List[Action]) -> None:
        self.ticks += 1
        self.processes_seen += len(actions)
        for a in actions:
            self.classifications[a.workload_class] = (
                self.classifications.get(a.workload_class, 0) + 1
            )
            if a.applied:
                self.actions_applied += 1


class TrainOSScheduler:
    def __init__(
        self,
        classifier: Classifier,
        interval: float = 3.0,
        window: int = 5,
        min_runtime: float = 2.0,
        dry_run: bool = False,
        manage_affinity: bool = False,
        min_cpu_to_manage: float = 1.0,
        verbose: bool = True,
    ):
        self.classifier = classifier
        self.interval = interval
        self.collector = MetricCollector(window=window, min_runtime=min_runtime)
        self.enforcer = PolicyEnforcer(dry_run=dry_run, manage_affinity=manage_affinity)
        self.min_cpu_to_manage = min_cpu_to_manage
        self.verbose = verbose
        self.stats = LoopStats()
        self._running = False

    def stop(self, *_):
        self._running = False

    def tick(self) -> List[Action]:
        """Run one observe->classify->act cycle."""
        self.collector.sample()
        trackers = self.collector.ready_trackers()

        # Load user overrides once per tick (cheap file read). A flagged PID or
        # name-pattern forces a class regardless of what the classifier infers,
        # and is never filtered out as idle -- the user asked for it explicitly.
        forced: Dict[int, str] = {}

        vectors, meta = [], []
        override_meta = []  # (pid, name, forced_class)
        for pid, tracker in trackers.items():
            f = tracker.features()
            forced_cls = override.lookup(pid, tracker.name)
            if forced_cls is not None:
                forced[pid] = forced_cls
                override_meta.append((pid, tracker.name, forced_cls))
                continue
            # ignore idle/near-dead processes to avoid churn
            if f["cpu_usage"] < self.min_cpu_to_manage and f["io_wait"] < 0.05:
                continue
            vectors.append([f[n] for n in self.classifier.feature_names])
            meta.append((pid, tracker.name))

        classes = self.classifier.predict_batch(vectors)

        actions: List[Action] = []
        # user-forced processes first (override wins)
        for pid, name, cls in override_meta:
            actions.append(self.enforcer.enforce(pid, name, cls))
        # classifier-decided processes
        for (pid, name), cls in zip(meta, classes):
            actions.append(self.enforcer.enforce(pid, name, cls))

        self.stats.record(actions)
        if self.verbose:
            self._log(actions)
        return actions

    def _log(self, actions: List[Action]) -> None:
        applied = [a for a in actions if a.applied]
        summary = {}
        for a in actions:
            summary[a.workload_class] = summary.get(a.workload_class, 0) + 1
        stamp = time.strftime("%H:%M:%S")
        print(
            f"[{stamp}] tick={self.stats.ticks} managed={len(actions)} "
            f"applied={len(applied)} classes={summary}"
        )

    def run(self, max_ticks: Optional[int] = None) -> LoopStats:
        self._running = True
        # prime psutil cpu_percent (first call always returns 0.0)
        self.collector.sample()
        time.sleep(min(1.0, self.interval))

        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        while self._running:
            start = time.time()
            self.tick()
            if max_ticks is not None and self.stats.ticks >= max_ticks:
                break
            elapsed = time.time() - start
            time.sleep(max(0.0, self.interval - elapsed))

        return self.stats
