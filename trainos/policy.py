"""Scheduling policy enforcement.

This is the part that turns the classifier into a *scheduler*. Given a workload
class for a process, it applies real Linux scheduling controls:

  - nice value        (priority hint to CFS/EEVDF)
  - scheduling policy (SCHED_OTHER / SCHED_BATCH / SCHED_IDLE)
  - CPU affinity      (optional pinning)

All actions are guarded so that:
  * on non-Linux the calls are simulated (no-op) so it stays testable
  * in --dry-run nothing is changed, only logged
  * we never touch kernel threads or PID 0/1 or our own process
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import psutil
except ImportError:  # keep POLICY_TABLE and helpers importable without psutil
    psutil = None  # type: ignore

IS_LINUX = platform.system() == "Linux"

# Per-class scheduling intent.
#   nice: -20 (highest prio) .. 19 (lowest). We only *lower* priority for
#         batch/compute work and *raise* it slightly for interactive work.
#   policy: SCHED_BATCH suits throughput CPU jobs; SCHED_OTHER for interactive.
POLICY_TABLE: Dict[str, Dict] = {
    "cpu_bound": {
        "nice": 10,
        "sched_policy": "SCHED_BATCH",
        "rationale": "throughput job, de-prioritise to protect responsiveness",
    },
    "io_bound": {
        "nice": 0,
        "sched_policy": "SCHED_OTHER",
        "rationale": "spends time waiting on I/O, keep default fairness",
    },
    "interactive": {
        "nice": -5,
        "sched_policy": "SCHED_OTHER",
        "rationale": "latency-sensitive, give a small priority boost",
    },
}

# PIDs we must never touch.
PROTECTED_PIDS = {0, 1}


@dataclass
class Action:
    pid: int
    name: str
    workload_class: str
    old_nice: Optional[int]
    new_nice: Optional[int]
    policy: str
    applied: bool
    note: str = ""


def _sched_policy_const(name: str) -> Optional[int]:
    return {
        "SCHED_OTHER": getattr(os, "SCHED_OTHER", None),
        "SCHED_BATCH": getattr(os, "SCHED_BATCH", None),
        "SCHED_IDLE": getattr(os, "SCHED_IDLE", None),
    }.get(name)


class PolicyEnforcer:
    def __init__(self, dry_run: bool = False, manage_affinity: bool = False):
        self.dry_run = dry_run
        self.manage_affinity = manage_affinity
        self._self_pid = os.getpid()
        self._ncpu = os.cpu_count() or 1

    def _is_safe_target(self, pid: int) -> bool:
        if pid in PROTECTED_PIDS or pid == self._self_pid:
            return False
        return True

    def enforce(self, pid: int, name: str, workload_class: str) -> Action:
        plan = POLICY_TABLE.get(workload_class, POLICY_TABLE["io_bound"])
        target_nice = plan["nice"]
        policy_name = plan["sched_policy"]

        if not self._is_safe_target(pid):
            return Action(pid, name, workload_class, None, None, policy_name,
                          applied=False, note="protected pid, skipped")

        if psutil is None:
            return Action(pid, name, workload_class, None, target_nice,
                          policy_name, applied=False, note="psutil unavailable")

        try:
            proc = psutil.Process(pid)
            old_nice = proc.nice()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return Action(pid, name, workload_class, None, None, policy_name,
                          applied=False, note="no access / gone")

        if self.dry_run:
            return Action(pid, name, workload_class, old_nice, target_nice,
                          policy_name, applied=False, note="dry-run")

        if not IS_LINUX:
            # Simulate on non-Linux dev machines so the loop is exercisable.
            return Action(pid, name, workload_class, old_nice, target_nice,
                          policy_name, applied=False, note="simulated (non-linux)")

        applied = True
        note_parts = []
        # 1) nice value
        try:
            proc.nice(target_nice)
        except (psutil.AccessDenied, psutil.NoSuchProcess) as exc:
            applied = False
            note_parts.append(f"nice failed: {type(exc).__name__}")

        # 2) scheduling policy (best effort; needs privileges for some policies)
        const = _sched_policy_const(policy_name)
        if const is not None and hasattr(os, "sched_setscheduler"):
            try:
                os.sched_setscheduler(pid, const, os.sched_param(0))
            except (PermissionError, OSError) as exc:
                note_parts.append(f"policy skipped: {type(exc).__name__}")

        # 3) optional affinity: pin cpu_bound jobs off core 0 to protect it
        if self.manage_affinity and hasattr(os, "sched_setaffinity") and self._ncpu > 1:
            try:
                if workload_class == "cpu_bound":
                    cores = set(range(1, self._ncpu))  # avoid core 0
                else:
                    cores = set(range(self._ncpu))
                os.sched_setaffinity(pid, cores)
            except (OSError, AttributeError) as exc:
                note_parts.append(f"affinity skipped: {type(exc).__name__}")

        return Action(pid, name, workload_class, old_nice, target_nice,
                      policy_name, applied=applied, note="; ".join(note_parts))
