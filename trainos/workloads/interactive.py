"""Interactive demo workload.

Simulates a latency-sensitive interactive app: it does a short burst of CPU
work (like reacting to a keystroke or rendering a frame) then sleeps (waiting
for the next event). Over a sampling window this looks bursty -- high CPU
variance, low I/O wait, short active periods -- which is the `interactive`
signature the classifier learned.

TrainOS should classify this as `interactive` and give it a small priority
boost (nice -5) so it stays responsive even when cpu_bound jobs are running.

    python -m trainos.workloads.interactive
    python -m trainos.workloads.interactive --duration 0   # until killed
"""

from __future__ import annotations

import math
import random
import time

from ._common import (
    banner,
    deadline_from,
    expired,
    install_sigterm_handler,
    parse_args,
    set_process_name,
)


def _burst(work_units: int) -> float:
    x = 0.0
    for i in range(1, work_units):
        x += math.sqrt(i) * math.cos(i)
    return x


def main() -> int:
    args = parse_args()
    install_sigterm_handler()
    set_process_name("trainos-interactive")
    banner("interactive", args.duration)

    deadline = deadline_from(args.duration)
    rng = random.Random(0)
    while not expired(deadline):
        # short, variable CPU burst (the "reacting to an event" part)
        _burst(rng.randint(150_000, 400_000))
        # idle wait for the "next event" -> creates the bursty, high-variance
        # profile that distinguishes interactive from steady cpu_bound work.
        time.sleep(rng.uniform(0.3, 0.9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
