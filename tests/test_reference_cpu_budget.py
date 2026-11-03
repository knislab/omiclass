import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.reference_preprocessing import cpu_budget as budget


def test_default_method_cpu_sets_are_disjoint_and_total_twenty(monkeypatch):
    monkeypatch.setattr(budget.os, "sched_getaffinity", lambda pid: set(range(24)))
    monkeypatch.setattr(budget, "core_key", lambda cpu: cpu)
    quantile = budget.method_cpus("frozen_quantile")
    fsva = budget.method_cpus("fsva_exact")
    assert quantile == list(range(10)) and fsva == list(range(10, 20))
    assert len(set(quantile + fsva)) == 20
    with pytest.raises(ValueError, match="at least"):
        budget.method_cpus("frozen_quantile", workers=3)
    monkeypatch.setattr(budget.os, "sched_getaffinity", lambda pid: set(range(10)))
    with pytest.raises(ValueError, match="outside"):
        budget.method_cpus("fsva_exact")


def test_smt_siblings_are_not_double_counted(monkeypatch):
    monkeypatch.setattr(budget, "core_key", lambda cpu: cpu % 2)
    with pytest.raises(ValueError, match="SMT"):
        budget.method_cpus("frozen_quantile")


@pytest.mark.parametrize("value", ["", "-1", "3-1", "0-1,1", "word"])
def test_invalid_cpu_lists_are_rejected(value):
    with pytest.raises(ValueError, match="CPU list"):
        budget.parse_cpu_list(value)


def test_explicit_cpu_ranges():
    assert budget.parse_cpu_list("0-4,10-14") == list(range(5)) + list(range(10, 15))


def test_bootstrap_affinity_is_inherited_by_dummy_descendants():
    cpus = budget.available_cpus()[:2]
    if len(cpus) < 2:
        pytest.skip("Needs two available cores")
    code = (
        "import sys,json,os,subprocess; from experiments.reference_preprocessing.cpu_budget import bootstrap; "
        "bootstrap(['run','--cores-per-seed','2','--cpu-list',sys.argv[1]]); "
        "probe='import os,json; print(json.dumps(dict(cpus=sorted(os.sched_getaffinity(0)),omp=os.environ[\"OMP_THREAD_LIMIT\"])))'; "
        "print(subprocess.check_output([sys.executable,'-c',probe],text=True))"
    )
    result = json.loads(subprocess.check_output([sys.executable, "-c", code, ",".join(map(str, cpus))], text=True))
    assert result == dict(cpus=cpus, omp="1")


def test_joblib_workers_and_nested_calls_respect_the_budget():
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_info
    import numpy as np
    from joblib.parallel import BACKENDS
    import threading
    original_thread_backend = BACKENDS["threading"]

    def nested_probe():
        from joblib import Parallel, delayed, effective_n_jobs
        import os
        import numpy as np
        from threadpoolctl import threadpool_info
        np.dot(np.ones((3, 3)), np.ones((3, 3)))
        nested = Parallel(n_jobs=9)(delayed(os.getpid)() for _ in range(3))
        return os.getpid(), nested, effective_n_jobs(9), threadpool_info()

    with budget.execution_limits(2):
        np.dot(np.ones((3, 3)), np.ones((3, 3)))
        assert all(p["num_threads"] == 1 for p in threadpool_info())
        result = Parallel(n_jobs=9)(delayed(nested_probe)() for _ in range(4))
        shared_threads = Parallel(n_jobs=9, require="sharedmem")(
            delayed(threading.get_ident)() for _ in range(20))
        assert len(set(shared_threads)) <= 2
    assert BACKENDS["threading"] is original_thread_backend
    assert len({r[0] for r in result}) <= 2
    for pid, nested, jobs, pools in result:
        assert set(nested) == {pid} and jobs == 1
        assert all(p["num_threads"] == 1 for p in pools)


def test_shell_defaults_without_launching_anything(tmp_path):
    # Replace only screen with a recorder; no experiment command is executed.
    script = Path(__file__).resolve().parents[1] / "experiments/reference_preprocessing/run_later.sh"
    recorder = tmp_path / "screen"
    recorder.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CPU_TEST_ARGUMENTS\"\n")
    recorder.chmod(0o755)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}")
    args_file = tmp_path / "args"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"], CPU_TEST_ARGUMENTS=str(args_file))
    subprocess.run(["bash", str(script), "frozen_quantile", "--screen", "dummy-record-only",
                    "--prepared", str(prepared), "--python", sys.executable, "--out", str(tmp_path / "new"),
                    "--cpus", "0-9"],
                    env=env, check=True, capture_output=True)
    args = args_file.read_text().splitlines()
    assert args[args.index("--workers") + 1] == "2"
    assert args[args.index("--cores-per-seed") + 1] == "5"
    assert args[args.index("--cpus") + 1] == "0-9"
    assert not (tmp_path / "new").exists()
