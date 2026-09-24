# TrainOS — Live Demo & Testing Guide (for the presentation)

A step-by-step script for demonstrating TrainOS in front of reviewers. Follow
it top to bottom. Every command is copy-paste ready. Each step says **what to
run**, **what the audience should see**, and **what to say**.

> **Golden rule: run the real demo on Linux (Ubuntu), ideally with `sudo`.**
> Only on Linux do `nice` / `SCHED_BATCH` / CPU affinity actually change
> scheduling. On macOS the scheduler *simulates* its actions, so the two
> phases look identical — fine for a dry rehearsal, useless as evidence.

---

## 0. Before the presentation (setup, do this in advance)

Do this once, on the machine you'll present from, so nothing is downloading
live in front of the room.

```bash
cd ~/Desktop/College_Major_Prject      # or wherever you cloned it
./install.sh                            # venv + deps + trains the model
./venv/bin/python -m trainos.cli info   # sanity: platform Linux, model present
```

Expected `info` output: platform `Linux`, a CPU count, `model : present`, and
the six feature names. If it says `model : MISSING`, run
`./venv/bin/python -m trainos.cli train` once.

Open **two terminals** side by side and `cd` both into the project directory.
Terminal 1 = "control", Terminal 2 = "observe". Have this file open on a phone
or second screen so you can read the talking points.

**Timing:** the full demo below is ~6–8 minutes. If you have less time, do
Steps 1, 2, and 5 only (the model, the mechanism, and the measured result).

---

## 1. Show the model is real (30 seconds)

**Run (Terminal 1):**
```bash
./venv/bin/python -m trainos.cli selftest
```

**Audience sees:** five checks, ending in `ALL CHECKS PASSED ✅`.

**Say:** "This proves the whole pipeline works on this machine end to end —
it generates data, trains and selects the best model, saves and reloads it,
collects live metrics from real processes, and runs one scheduling cycle."

Optionally show the model-selection numbers:
```bash
./venv/bin/python -m trainos.cli train
```
Point at the comparison table (RandomForest chosen) and the feature
importances. **Say:** "No single feature dominates, so there's no label
leakage — the classifier is learning a real boundary."

---

## 2. Explain what it classifies (30 seconds, no command)

**Say:** "TrainOS sorts every running process into one of two classes:
`heavy_compute` — a sustained, CPU- and memory-heavy job, which is what an ML
training run looks like — and `normal`, ordinary background work. The goal is
to let that heavy job run well even on a machine that wasn't built for it, by
getting the background churn out of its way."

Be honest and pre-empt the obvious question (see FAQ Q1): the classifier
detects *sustained heavy compute*, of which ML training is the target case.

---

## 3. Show live classification + scheduling (2 minutes)

Generate background load, then watch TrainOS classify and act on it.

**Terminal 1 — create load:**
```bash
./venv/bin/python -m trainos.cli generate --count 60 --duration 120
```
**Audience sees:** "spawned 60 jobs".

**Terminal 2 — start a heavy job and flag it, then run the scheduler:**
```bash
# start a real ML training job in the background
./venv/bin/python -m trainos.workloads.ml_train --rounds 0 &
# note the pid it prints, then flag it as the priority job
./venv/bin/python -m trainos.cli flag --name ml_train --class heavy_compute
# run the scheduler (sudo so boosting takes full effect)
sudo ./venv/bin/python -m trainos.cli run --interval 2 --affinity
```

**Audience sees:** a log line each tick, e.g.
`tick=3 managed=48 applied=45 classes={'normal': 47, 'heavy_compute': 1}`.

**Say:** "It's seeing ~60 processes every two seconds, classifying almost all
of them as `normal`, and the one training job as `heavy_compute`. On Linux with
root it's now actively re-nicing them — the training job up, the rest down."

Press **Ctrl-C** to stop the scheduler. It prints a session summary (ticks,
processes managed, actions applied, class breakdown).

**Clean up:**
```bash
./venv/bin/python -m trainos.cli generate --stop
./venv/bin/python -m trainos.cli flag --clear-all
kill %1 2>/dev/null    # stop the ml_train job if still running
```

---

## 4. Show the user override (30 seconds)

**Say:** "If the operator already knows a job matters — before it even builds
up a behavioural signature — they can flag it, and that always wins over the
classifier."

```bash
./venv/bin/python -m trainos.cli flag --pid 12345 --class heavy_compute
./venv/bin/python -m trainos.cli flag --list
./venv/bin/python -m trainos.cli flag --clear-all
```

(Use any real PID from `ps` if you want `--list` to look concrete.)

---

## 5. The measured result — the headline (2–3 minutes)

This is the part that proves the point. It runs the ML job under heavy load
**twice** — once with the default OS scheduler, once with TrainOS — and reports
the difference.

**Run (Terminal 1), on Linux under sudo:**
```bash
sudo ./venv/bin/python -m trainos.cli evaluate --count 100 --rounds 20
```

**Audience sees:** two phases run, then a table:

```
metric                               baseline      trainos
------------------------------------------------------------
rounds completed                           20           20
wall seconds (lower better)             XX.XX        YY.YY
rounds/sec (higher better)              A.AAA        B.BBB
------------------------------------------------------------
completion time reduction: +Z.ZZ%  (positive = TrainOS finished the ML job faster)
```

**Say:** "Same machine, same 100 background jobs, same training job. Under the
default scheduler it took XX seconds; under TrainOS, YY. TrainOS finished the
demanding job Z percent faster by de-prioritising the background work. This is
a real before/after on real processes, not a simulation."

**Tuning for the room:**
- Bigger, clearer gap: raise `--count` (more contention) — e.g. `--count 150`.
- Faster run: lower `--rounds` (e.g. `--rounds 10`) and `--count` (e.g. `--count 60`).
- Always keep `--count` background jobs > number of CPU cores, or there's no
  contention to resolve and the gap shrinks to nothing.

---

## 6. Optional — the illustrative simulation (skip if short on time)

```bash
./venv/bin/python -m trainos.cli benchmark
```

**Say, honestly:** "This is a simple simulation that explains the *mechanism* —
it's not a measurement. The real evidence is the `evaluate` run you just saw."
Do not present this as proof; a sharp reviewer will note the weights are
assigned. Lead with Step 5, keep this as a mechanism illustration only.

---

## FAQ — likely reviewer questions and honest answers

**Q1. "Does it actually detect ML, or just heavy CPU jobs?"**
It detects *sustained heavy compute* — high steady CPU + memory over time. An
ML training job fits that profile, but so would video encoding or a big
simulation. We're honest about that: the classifier protects sustained heavy
compute, of which ML training is the target case. When the operator needs a
*specific* job prioritised, the `flag` override names it explicitly.

**Q2. "Isn't the benchmark rigged?"**
The `benchmark` command is a simulation and yes, its weights are assigned — it
only illustrates the mechanism. That's why the headline result is `evaluate`,
which measures real processes with real scheduling controls and no assumed
advantage.

**Q3. "Why does it need root / sudo?"**
Raising a process's priority (negative nice) and changing scheduling policy
require privileges on Linux. Without root, TrainOS can still *lower* the
priority of background jobs (positive nice needs no privilege), which is the
larger lever, so you still see a benefit — just a smaller one.

**Q4. "Is this a kernel change?"**
No. It's a user-space daemon using standard Linux syscalls (`nice`,
`sched_setscheduler`, `sched_setaffinity`) on top of the existing scheduler.
No kernel patch, no reboot. All actions are reversible.

**Q5. "What if it misclassifies something important?"**
Two safeguards: it never touches PID 0/1 or itself, and the operator can
override any process's class with `flag`. `nice` changes reset when a process
exits or when TrainOS is stopped.

**Q6. "Does it work on this laptop (macOS)?"**
The logic runs anywhere, but macOS hides per-process I/O counters and only
*simulates* the scheduling actions, so the measured gap disappears. The real
numbers come from Linux — that's where it's meant to run.

---

## Emergency fallbacks

- **A command hangs:** Ctrl-C. The scheduler and workloads all stop cleanly on
  Ctrl-C / SIGTERM.
- **Leftover jobs after a crash:** `./venv/bin/python -m trainos.cli generate --stop`
  and `./venv/bin/python -m trainos.cli flag --clear-all`.
- **Machine feels frozen under load:** you set `--count` too high for the core
  count; stop the jobs (above) and rerun with a smaller `--count`.
- **No Linux box available live:** rehearse and screen-record the `evaluate`
  run on Linux beforehand, and play the recording — state clearly it's a
  captured run, then show the code live.

---

## One-screen cheat sheet

```bash
# setup (once, in advance)
./install.sh
./venv/bin/python -m trainos.cli selftest        # expect ALL CHECKS PASSED

# live classification
./venv/bin/python -m trainos.cli generate --count 60 --duration 120
./venv/bin/python -m trainos.workloads.ml_train --rounds 0 &
./venv/bin/python -m trainos.cli flag --name ml_train --class heavy_compute
sudo ./venv/bin/python -m trainos.cli run --interval 2 --affinity    # Ctrl-C to stop

# the measured result (headline)
sudo ./venv/bin/python -m trainos.cli evaluate --count 100 --rounds 20

# cleanup
./venv/bin/python -m trainos.cli generate --stop
./venv/bin/python -m trainos.cli flag --clear-all
```
