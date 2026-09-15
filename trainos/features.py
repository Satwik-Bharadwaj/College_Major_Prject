"""Runtime metric collection and feature extraction.

Uses psutil to sample per-process metrics. Works on Linux (full metrics) and
degrades gracefully on macOS/Windows (io_wait may be 0.0) so the pipeline is
testable off Linux.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Optional

try:  # psutil is required at runtime, but the pure-Python helpers below
    import psutil  # (variance/mean/io_wait) stay importable without it so the
except ImportError:  # logic can be unit-tested in minimal environments.
    psutil = None  # type: ignore

from . import FEATURE_NAMES


@dataclass
class ProcessSample:
    """A single point-in-time reading for one process."""

    pid: int
    cpu_percent: float
    memory_percent: float
    num_threads: int
    io_read_bytes: int
    io_write_bytes: int
    ctx_switches: int
    timestamp: float


@dataclass
class ProcessTracker:
    """Accumulates samples for one process and derives feature vectors.

    We keep a short rolling window of samples so we can compute variance and
    throughput stability, which need more than one reading.
    """

    pid: int
    name: str = ""
    window: int = 5
    _cpu_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    _io_hist: Deque[int] = field(default_factory=lambda: deque(maxlen=5))
    _first_seen: float = field(default_factory=time.time)
    _last_io_total: int = 0

    def add(self, sample: ProcessSample) -> None:
        self._cpu_hist.append(sample.cpu_percent)
        io_total = sample.io_read_bytes + sample.io_write_bytes
        # store per-interval delta so throughput stability reflects rate changes
        delta = max(0, io_total - self._last_io_total) if self._last_io_total else 0
        self._last_io_total = io_total
        self._io_hist.append(delta)

    def ready(self) -> bool:
        return len(self._cpu_hist) >= min(2, self.window)

    def features(self) -> Dict[str, float]:
        cpu_vals = list(self._cpu_hist)
        cpu_mean = _mean(cpu_vals)
        cpu_var = _variance(cpu_vals, cpu_mean)

        io_vals = list(self._io_hist)
        io_mean = _mean(io_vals)
        # throughput stability: high when io deltas are consistent (low rel. std)
        if io_mean > 0:
            io_std = _variance(io_vals, io_mean) ** 0.5
            stability = max(0.0, 1.0 - min(1.0, io_std / (io_mean + 1e-9)))
        else:
            # no I/O activity -> treat as perfectly stable (pure compute)
            stability = 1.0

        # io_wait proxy: share of samples where the process did I/O but low CPU
        io_wait = _io_wait_proxy(cpu_vals, io_vals)

        mem = 0.0
        if psutil is not None:
            try:
                mem = psutil.Process(self.pid).memory_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                mem = 0.0

        return {
            "cpu_usage": round(cpu_mean, 4),
            "cpu_variance": round(cpu_var, 4),
            "memory_usage": round(mem, 4),
            "io_wait": round(io_wait, 4),
            "runtime_duration": round(time.time() - self._first_seen, 4),
            "throughput_stability": round(stability, 4),
        }

    def feature_vector(self) -> List[float]:
        f = self.features()
        return [f[name] for name in FEATURE_NAMES]


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _variance(vals: List[float], mean: Optional[float] = None) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals) if mean is None else mean
    return sum((v - m) ** 2 for v in vals) / len(vals)


def _io_wait_proxy(cpu_vals: List[float], io_vals: List[float]) -> float:
    """Estimate I/O wait as fraction of intervals with I/O but little CPU.

    A process doing lots of I/O while its CPU% is low is likely I/O-bound and
    spending time in iowait. This is a portable proxy that does not need
    /proc/<pid>/stat parsing (which is Linux-only).
    """
    if not io_vals:
        return 0.0
    n = min(len(cpu_vals), len(io_vals))
    if n == 0:
        return 0.0
    waiting = sum(
        1 for i in range(n) if io_vals[i] > 0 and cpu_vals[i] < 30.0
    )
    return waiting / n


class MetricCollector:
    """Discovers processes and maintains a tracker per PID."""

    def __init__(self, window: int = 5, min_runtime: float = 0.0):
        self.window = window
        self.min_runtime = min_runtime
        self._trackers: Dict[int, ProcessTracker] = {}

    def sample(self) -> Dict[int, ProcessTracker]:
        """Take one sample of every accessible process."""
        seen = set()
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent",
             "num_threads", "io_counters", "num_ctx_switches"]
        ):
            try:
                info = proc.info
                pid = info["pid"]
                seen.add(pid)
                io = info.get("io_counters")
                ctx = info.get("num_ctx_switches")
                sample = ProcessSample(
                    pid=pid,
                    cpu_percent=info.get("cpu_percent") or 0.0,
                    memory_percent=info.get("memory_percent") or 0.0,
                    num_threads=info.get("num_threads") or 1,
                    io_read_bytes=getattr(io, "read_bytes", 0) if io else 0,
                    io_write_bytes=getattr(io, "write_bytes", 0) if io else 0,
                    ctx_switches=(ctx.voluntary + ctx.involuntary) if ctx else 0,
                    timestamp=time.time(),
                )
                tracker = self._trackers.get(pid)
                if tracker is None:
                    tracker = ProcessTracker(
                        pid=pid, name=info.get("name") or "", window=self.window
                    )
                    tracker._cpu_hist = deque(maxlen=self.window)
                    tracker._io_hist = deque(maxlen=self.window)
                    self._trackers[pid] = tracker
                tracker.add(sample)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        # drop trackers for processes that have exited
        for dead in set(self._trackers) - seen:
            self._trackers.pop(dead, None)

        return self._trackers

    def ready_trackers(self) -> Dict[int, ProcessTracker]:
        return {
            pid: t for pid, t in self._trackers.items()
            if t.ready() and (time.time() - t._first_seen) >= self.min_runtime
        }
