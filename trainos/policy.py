"""Scheduling policy enforcement.

This is the part that turns the classifier into a *scheduler*. Given a workload
class for a process, it applies real Linux scheduling controls:

  - nice value        (priority hint to CFS/EEVDF)
  - scheduling policy (SCHED_OTHER / SCHED_BATCH / SCHED_IDLE)
  - CPU affinity      (optional pinning)

The intent: boost the demanding heavy_compute job (e.g. an ML training run)
and de-prioritise ordinary background work so the heavy job runs well even on
a machine not designed for it.

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
#   nice: -20 (highest prio) .. 19 (lowest). We *raise* priority for the
#         demanding heavy-compute job and *lower* it for ordinary background
#         work, so the heavy job gets the CPU on an oversubscribed machine.
#   policy: SCHED_OTHER for the priority job (fully schedulable), SCHED_BATCH
#           for background work (yields to interactive/normal timeslices).
POLICY_TABLE: Dict[str, Dict] = {
    "heavy_compute": {
        "nice": -10,
        "sched_policy": "SCHED_OTHER",
        "rationale": "priority workload (e.g. ML training); give it the CPU",
    },
    "normal": {
        "nice": 10,
        "sched_policy": "SCHED_BATCH",
        "rationale": "ordinary background work, step aside for the heavy job",
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
        # Unknown classes fall back to `normal` (treated as background work).
        plan = POLICY_TABLE.get(workload_class, POLICY_TABLE["normal"])
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

        # 3) optional affinity: keep ordinary background work off core 0 so the
        #    heavy_compute job always has at least one uncontended core. The
        #    heavy job itself is free to use every core.
        if self.manage_affinity and hasattr(os, "sched_setaffinity") and self._ncpu > 1:
            try:
                if workload_class == "normal":
                    cores = set(range(1, self._ncpu))  # confine off core 0
                else:
                    cores = set(range(self._ncpu))     # heavy job: all cores
                os.sched_setaffinity(pid, cores)
            except (OSError, AttributeError) as exc:
                note_parts.append(f"affinity skipped: {type(exc).__name__}")

        return Action(pid, name, workload_class, old_nice, target_nice,
                      policy_name, applied=applied, note="; ".join(note_parts))
