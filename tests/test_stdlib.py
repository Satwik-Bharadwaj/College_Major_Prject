"""Dependency-free tests (no psutil/sklearn needed).

Run with:  python3 -m tests.test_stdlib      (from project root)
or:        python3 tests/test_stdlib.py

Verifies dataset generation, feature math, benchmark simulation, CLI parser
wiring and policy-table integrity. Use `trainos selftest` for the full,
dependency-backed end-to-end check once installed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        check.failed = True
check.failed = False


def main():
    print("TrainOS stdlib verification")
    print("=" * 50)

    print("\n1. Synthetic dataset")
    from trainos import dataset as ds
    from trainos import FEATURE_NAMES, WORKLOAD_CLASSES
    X, y = ds.generate_synthetic(n_per_class=300, seed=1)
    check("row count == 900", len(X) == 900)
    check("feature width == 6", len(X[0]) == len(FEATURE_NAMES))
    check("all 3 classes present", set(y) == set(WORKLOAD_CLASSES))

    cpu_idx = FEATURE_NAMES.index("cpu_usage")
    cpu = [X[i][cpu_idx] for i in range(len(X)) if y[i] == "cpu_bound"]
    inter = [X[i][cpu_idx] for i in range(len(X)) if y[i] == "interactive"]
    check("cpu_bound cpu > interactive cpu",
          sum(cpu) / len(cpu) > sum(inter) / len(inter))

    io_idx = FEATURE_NAMES.index("io_wait")
    io = [X[i][io_idx] for i in range(len(X)) if y[i] == "io_bound"]
    check("io_bound io_wait high (>0.3)", sum(io) / len(io) > 0.3)

    print("\n2. Feature math helpers")
    import trainos.features as F
    check("variance of constant == 0", F._variance([5, 5, 5]) == 0.0)
    check("variance of [0,10] == 25", abs(F._variance([0, 10]) - 25.0) < 1e-9)
    check("mean of [2,4,6] == 4", F._mean([2, 4, 6]) == 4.0)
    check("io_wait proxy high (low cpu + io)",
          F._io_wait_proxy([5, 10, 8], [100, 200, 150]) > 0.9)
    check("io_wait proxy low (high cpu + io)",
          F._io_wait_proxy([90, 95, 92], [100, 200, 150]) < 0.1)

    print("\n3. Benchmark simulation (CFS vs TrainOS)")
    from trainos.benchmark import run_simulation
    res = run_simulation()
    base = res["baseline_default_scheduler"]
    trainos = res["trainos"]
    check("baseline has makespan", base["makespan"] > 0)
    check("trainos improves interactive completion",
          trainos["avg_interactive_completion"] < base["avg_interactive_completion"])
    print(f"      improvement (interactive): "
          f"{res['improvement_percent']['avg_interactive_completion']:+.1f}%")

    print("\n4. CLI parser wiring")
    from trainos.cli import build_parser
    parser = build_parser()
    for cmd in ["train", "collect", "run", "benchmark", "selftest", "info"]:
        argv = ["run", "--dry-run"] if cmd == "run" else [cmd]
        args = parser.parse_args(argv)
        check(f"subcommand '{cmd}' parses", hasattr(args, "func"))

    print("\n5. Policy table")
    from trainos.policy import POLICY_TABLE
    check("every class has a policy",
          all(c in POLICY_TABLE for c in WORKLOAD_CLASSES))
    check("interactive prioritised over cpu_bound",
          POLICY_TABLE["interactive"]["nice"] < POLICY_TABLE["cpu_bound"]["nice"])

    print("\n" + "=" * 50)
    if check.failed:
        print("RESULT: SOME CHECKS FAILED")
        return 1
    print("RESULT: ALL STDLIB CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
