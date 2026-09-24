# TrainOS — Workload-Aware CPU Scheduling Layer

TrainOS is a **user-space adaptive CPU scheduling layer** for Linux. Its goal is
to let a **demanding job -- an ML training run or any sustained heavy-compute
workload -- run well on a machine that was not designed for it.** It observes
running processes, classifies each one with a trained **Random Forest** model as
`heavy_compute` (the demanding job to protect) or `normal` (ordinary background
work), and applies real scheduling controls (priority, CPU affinity, scheduling
policy) on top of the Linux scheduler (CFS/EEVDF) to give the heavy job the CPU.

It is **not** a kernel patch. It runs as a normal background process/daemon and
uses standard Linux syscalls, so it ships to any Ubuntu machine and runs after a
one-line install.

```
┌──────────────┐   metrics   ┌───────────────┐  class   ┌────────────────┐
│  Processes   │ ──────────▶ │ Random Forest │ ───────▶ │ Policy Enforcer │
│ (via psutil) │             │  classifier   │          │ nice/affinity  │
└──────────────┘             └───────────────┘          └────────────────┘
        ▲                            ▲                           │
        │                    user flag override                  │
        └────────────────── control loop (every N sec) ──────────┘
```

## What it does

1. **Collects** runtime metrics per process (CPU %, CPU variance, memory %,
   I/O wait, runtime, throughput stability).
2. **Classifies** each process as `heavy_compute` (sustained high CPU + memory,
   steady — an ML training run is the target case) or `normal` (everything
   else). A **user flag** can force a job's class when the classifier is unsure.
3. **Acts** by boosting the heavy job (higher priority) and de-prioritising
   background `normal` work (lower priority, SCHED_BATCH, kept off core 0) —
   the part that makes it a scheduler, not just a classifier.

Honest scope: the classifier separates *sustained heavy compute* from ordinary
work — an ML job and a non-ML CPU job that both saturate the machine are both
`heavy_compute`. To measure the benefit on real processes, use
`trainos evaluate` (a real A/B, not a simulation).

## Quick start (Ubuntu)

```bash
./install.sh                      # installs deps into a venv
sudo ./venv/bin/python -m trainos.cli run --dry-run   # observe only, no changes
sudo ./venv/bin/python -m trainos.cli run             # real scheduling
```

## Testing anywhere (macOS/Linux, no root)

```bash
./install.sh
./venv/bin/python -m trainos.cli selftest    # end-to-end check
./venv/bin/python -m trainos.cli train       # (re)train the model
./venv/bin/python -m trainos.cli benchmark   # simulated CFS vs TrainOS comparison
```

See `docs/USAGE.md` for the full guide and `docs/PHASE3.md` for how this maps to
your project report.
