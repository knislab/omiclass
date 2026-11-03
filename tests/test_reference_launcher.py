import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import json

import pytest

from experiments.reference_preprocessing.launch_comparison import run_owned_process


def test_stopped_launcher_does_not_spawn(monkeypatch, tmp_path):
    stopped = threading.Event()
    stopped.set()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("spawned")))
    with (tmp_path / "log").open("w") as stream:
        assert run_owned_process(["unused"], cwd=tmp_path, env=os.environ.copy(), stream=stream, stopped=stopped) < 0


def test_cancellation_only_stops_owned_dummy_processes(tmp_path):
    # Dummy sleepers only: this test starts no preprocessing or classifiers.
    stopped = threading.Event()
    pid_file = tmp_path / "owned.pid"
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    errors = []
    result = []
    def run():
        try:
            with (tmp_path / "log").open("w") as stream:
                result.append(run_owned_process([sys.executable, "-c", parent_code, str(pid_file), child_code],
                    cwd=tmp_path, env=os.environ.copy(), stream=stream, stopped=stopped))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    try:
        thread.start()
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pid_file.exists()
        stopped.set()
        thread.join(timeout=15)
        assert not thread.is_alive() and not errors and result[0] != 0
        assert unrelated.poll() is None
        pid = int(pid_file.read_text())
        # A dead orphan may briefly be a zombie until PID 1 reaps it.
        proc_stat = Path(f"/proc/{pid}/stat")
        def still_running():
            try:
                return proc_stat.read_text().split()[2] not in {"Z", "X"}
            except (FileNotFoundError, ProcessLookupError):
                return False
        deadline = time.monotonic() + 3
        while still_running() and time.monotonic() < deadline:
            time.sleep(.02)
        assert not still_running()
    finally:
        stopped.set()
        thread.join(timeout=15)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_two_seed_queue_reuses_cpu_slots_and_isolates_a_failed_seed(tmp_path, monkeypatch):
    import experiments.reference_preprocessing.launch_comparison as launcher
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: None)
    lock = threading.Lock()
    active = set()
    visited = []
    peaks = []

    def fake_process(command, *, env, stream, **kwargs):
        if "preprocess" in command:
            return 0  # No preprocessing, models, or real experiment subprocesses.
        seed = int(command[command.index("--seed") + 1])
        cpus = frozenset(map(int, command[command.index("--cpu-list") + 1].split(",")))
        assert len(cpus) == 5
        assert Path(stream.name).name == f"seed{seed}.log"
        assert f"seed{seed}/" in env["MPLCONFIGDIR"]
        assert f"seed{seed}/" in env["JOBLIB_TEMP_FOLDER"]
        with lock:
            assert all(not cpus & other for other in active)
            active.add(cpus)
            peaks.append(len(active))
            visited.append(seed)
        try:
            time.sleep(.03)
            if seed == 11:
                raise OSError("simulated per-seed launch failure")
            return 0
        finally:
            with lock:
                active.remove(cpus)

    monkeypatch.setattr(launcher, "run_owned_process", fake_process)
    args = SimpleNamespace(out=tmp_path / "runs", prepared=tmp_path / "input",
        workers=2, cores_per_seed=5,
        method="frozen_quantile", n_factors=1, r_library=None, rscript=None)
    with pytest.raises(SystemExit) as error:
        launcher.launch(args, list(range(10)))
    assert error.value.code == 1
    assert sorted(visited) == [11, 61, 73, 80, 89] and max(peaks) == 2
    progress = json.loads((args.out / "progress.json").read_text())
    assert progress["seeds"]["11"] == "failed:1"
    assert all(progress["seeds"][str(s)] == "complete" for s in [61, 73, 80, 89])
