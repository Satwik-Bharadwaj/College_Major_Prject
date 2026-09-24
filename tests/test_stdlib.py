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
    check("row count == 600", len(X) == 300 * len(WORKLOAD_CLASSES))
    check("feature width == 6", len(X[0]) == len(FEATURE_NAMES))
    check("both classes present", set(y) == set(WORKLOAD_CLASSES))
    check("classes are heavy_compute + normal",
          set(WORKLOAD_CLASSES) == {"heavy_compute", "normal"})

    cpu_idx = FEATURE_NAMES.index("cpu_usage")
    heavy = [X[i][cpu_idx] for i in range(len(X)) if y[i] == "heavy_compute"]
    normal = [X[i][cpu_idx] for i in range(len(X)) if y[i] == "normal"]
    check("heavy_compute cpu > normal cpu",
          sum(heavy) / len(heavy) > sum(normal) / len(normal))

    stab_idx = FEATURE_NAMES.index("throughput_stability")
    h_stab = [X[i][stab_idx] for i in range(len(X)) if y[i] == "heavy_compute"]
    check("heavy_compute stability high (>0.8)",
          sum(h_stab) / len(h_stab) > 0.8)

    print("\n2. Feature math helpers")
    import trainos.features as F
    check("variance of constant == 0", F._variance([5, 5, 5]) == 0.0)
    check("variance of [0,10] == 25", abs(F._variance([0, 10]) - 25.0) < 1e-9)
    check("mean of [2,4,6] == 4", F._mean([2, 4, 6]) == 4.0)
    check("io_wait proxy high (low cpu + io)",
          F._io_wait_proxy([5, 10, 8], [100, 200, 150]) > 0.9)
    check("io_wait proxy low (high cpu + io)",
          F._io_wait_proxy([90, 95, 92], [100, 200, 150]) < 0.1)

    print("\n3. Benchmark simulation (default vs TrainOS)")
    from trainos.benchmark import run_simulation
    res = run_simulation()
    base = res["baseline_default_scheduler"]
    trainos = res["trainos"]
    check("baseline has makespan", base["makespan"] > 0)
    check("trainos speeds up heavy_compute completion",
          trainos["heavy_completion"] < base["heavy_completion"])
    print(f"      heavy_compute completion improvement: "
          f"{res['improvement_percent']['heavy_completion']:+.1f}%")

    print("\n4. CLI parser wiring")
    from trainos.cli import build_parser
    parser = build_parser()
    for cmd in ["train", "collect", "run", "benchmark", "selftest", "info",
                "flag", "generate", "evaluate"]:
        argv = ["run", "--dry-run"] if cmd == "run" else [cmd]
        args = parser.parse_args(argv)
        check(f"subcommand '{cmd}' parses", hasattr(args, "func"))

    print("\n5. Policy table")
    from trainos.policy import POLICY_TABLE
    check("every class has a policy",
          all(c in POLICY_TABLE for c in WORKLOAD_CLASSES))
    check("heavy_compute prioritised over normal",
          POLICY_TABLE["heavy_compute"]["nice"] < POLICY_TABLE["normal"]["nice"])

    print("\n6. User override store")
    import tempfile, os as _os
    _tmp = tempfile.mktemp(suffix=".json")
    _os.environ["TRAINOS_OVERRIDE_FILE"] = _tmp
    import importlib
    from trainos import override as _ov
    importlib.reload(_ov)
    _ov.clear_all()
    _ov.set_pid(4242, "heavy_compute")
    check("pid override looked up", _ov.lookup(4242) == "heavy_compute")
    _ov.set_name("train.py", "heavy_compute")
    check("name override matches substring",
          _ov.lookup(999, "python train.py --x") == "heavy_compute")
    check("pid wins over classifier default", _ov.lookup(4242, "whatever") == "heavy_compute")
    _ov.clear_pid(4242)
    check("cleared pid override returns None", _ov.lookup(4242) is None)
    _ov.clear_all()
    try:
        _os.remove(_tmp)
    except OSError:
        pass

    print("\n" + "=" * 50)
    if check.failed:
        print("RESULT: SOME CHECKS FAILED")
        return 1
    print("RESULT: ALL STDLIB CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
