# Phase-III: From Classifier to Scheduler — how the code maps to the report

Your Phase-II report ends with a classifier. This codebase completes the goal
in the title: an actual scheduling layer. Here's the mapping so you can defend
it in the viva.

## What changed vs Phase-II (and why)

| Phase-II (report) | Phase-III (this code) | Reason |
|---|---|---|
| Label = "ML vs non-ML" | Label = `cpu_bound` / `io_bound` / `interactive` | The scheduler acts on *behaviour*, not on whether a job is ML. A video encode and a training loop look identical to the CPU; both should be scheduled the same way. |
| Output = accuracy number | Output = scheduling action + measured effect | A classifier alone doesn't improve scheduling. We apply `nice`/policy/affinity and measure completion time. |
| Random Forest chosen by accuracy | Random Forest chosen by macro-F1 **and** inference speed | The model runs inside a hot loop; speed and no-GPU matter as much as accuracy. |
| Linear Regression "classifier" | Removed | Regression used for classification is a conceptual error; dropped it. |

## Architecture (realises Figure 3.1 of the report)

```
trainos/
  features.py    # 3.4 Process Metric Collection  (the 6 metrics X1..X6)
  dataset.py     # reproducible labelled data + real-trace loader
  model.py       # 3.5 ML Model Training + selection (RF wins)
  policy.py      # NEW: the scheduling action (nice / SCHED_* / affinity)
  scheduler.py   # NEW: the observe->classify->act control loop (Fig 3.1)
  benchmark.py   # NEW: default-scheduler vs TrainOS comparison (validation)
  cli.py         # user entrypoint
```

The six features are exactly the report's mathematical model
`Y = f(X1..X6)` (Section 3.6): cpu_usage, cpu_variance, memory_usage, io_wait,
runtime_duration, throughput_stability.

## Why user-space and not a kernel patch

- Ships to any Ubuntu with no recompile and no reboot — meets your "just install
  and run" requirement.
- Uses standard Linux controls the kernel already exposes: `setpriority`
  (nice), `sched_setscheduler` (SCHED_OTHER/BATCH/IDLE), `sched_setaffinity`.
- A bug lowers a process's priority; it does not panic the machine. A scheduler
  bug in kernel space can.
- This is genuinely a "scheduling layer on top of CFS/EEVDF", which is what the
  report title claims.

## The scheduling policy (policy.py)

| Class | nice | policy | intent |
|---|---|---|---|
| cpu_bound | +10 | SCHED_BATCH | throughput job; step aside so the machine stays responsive |
| io_bound | 0 | SCHED_OTHER | spends time waiting; default fairness is fine |
| interactive | -5 | SCHED_OTHER | latency-sensitive; small priority boost |

## Validation experiment (benchmark.py)

Runs the same mixed workload under (a) equal fair-share (models default CFS) and
(b) TrainOS class-aware weights, then reports the delta. On the reference
workload TrainOS cuts average interactive-job completion time by ~36% while
keeping makespan essentially unchanged — i.e. better responsiveness at no
throughput cost.

For the report's results chapter, run on Linux:
```
trainos benchmark          # simulated numbers (reproducible)
```
and, for real numbers, the live before/after with `nice`/policy applied to
actual stress processes (collect completion times with `/usr/bin/time`).

## Honest limitations to state in the report

- Synthetic training data is separable by design; report accuracy on a **real
  collected trace** (`collect` then `train --csv`) for a credible number.
- The benchmark simulation is a transparent model of fair-share vs weighted
  scheduling, not a kernel measurement. Present it as a mechanism illustration;
  present the live `nice`-based before/after as the empirical result.
- Very short-lived processes may exit before TrainOS classifies them; this is
  expected and safe.
