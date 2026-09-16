"""CPU-bound demo workload.

A tight arithmetic loop with no sleeping and no I/O. This produces the
signature the classifier learned for `cpu_bound`: high, steady CPU
utilisation, low variance, ~zero I/O wait, high throughput stability.

TrainOS should classify this as `cpu_bound` and de-prioritise it
(nice 10, SCHED_BATCH) so it stops hurting interactive responsiveness.

    python -m trainos.workloads.cpu_bound            # run 120s
    python -m trainos.workloads.cpu_bound --duration 0   # until killed
"""

from __future__ import annotations

import math

from ._common import (
    banner,
    deadline_from,
    expired,
    install_sigterm_handler,
    parse_args,
    set_process_name,
)


def main() -> int:
    args = parse_args()
    install_sigterm_handler()
    set_process_name("trainos-cpu-bound")
    banner("cpu_bound", args.duration)

    deadline = deadline_from(args.duration)
    x = 0.0
    # Pure compute: no sleep, no I/O. Keeps a core pinned near 100%.
    while not expired(deadline):
        # a chunk of work between deadline checks so we don't check the clock
        # too often (which would add jitter and lower the CPU signature).
        for i in range(1, 200_000):
            x += math.sqrt(i) * math.sin(i)
        # prevent the optimiser/GC from discarding the result
        if x == float("inf"):
            x = 0.0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
