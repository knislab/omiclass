"""Run a shared preprocessing fit followed by the five unmodified manuscript runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

from experiments.reference_preprocessing.cpu_budget import bind_cpus, method_cpus, thread_environment


def run_owned_process(command, *, cwd, env, stream, stopped):
    """Own a process group so stopping a launcher cannot leave its fits behind."""
    if stopped.is_set():
        return -signal.SIGTERM
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            if stopped.is_set():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    return process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return process.wait()
                finally:
                    # Some multiprocessing trackers ignore SIGTERM. Reap any
                    # surviving descendants even if the group leader exited.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                return process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        # Retire orphaned trackers/workers belonging to this completed child.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=["frozen_quantile", "fsva_exact"], required=True)
    parser.add_argument("--r-library", type=Path)
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--n-factors", type=int, default=1)
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--cores-per-seed", type=int, default=5)
    parser.add_argument("--cpus", help="Method CPU set (quantile: 0-9; exact-fSVA: 10-19)")
    args = parser.parse_args()
    os.environ.update(thread_environment(args.cores_per_seed))
    cpus = method_cpus(args.method, args.workers, args.cores_per_seed, args.cpus)
    bind_cpus(cpus)
    launch(args, cpus)


def launch(args, cpus):
    # Thread settings are established before importing numpy or the runner.
    from experiments.reference_preprocessing.run import ROOT, SEEDS, digest
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(thread_environment(args.cores_per_seed), PYTHONUNBUFFERED="1")
    slots = queue.Queue()
    for i in range(args.workers):
        slots.put(cpus[i * args.cores_per_seed:(i + 1) * args.cores_per_seed])
    (out / "cpu_allocation.json").write_text(json.dumps(dict(
        workers=args.workers, cores_per_seed=args.cores_per_seed, method_cores=len(cpus), cpu_ids=cpus,
        thread_environment=thread_environment(args.cores_per_seed)), indent=2))
    code = list((ROOT / "omiclass").rglob("*.py")) + list((ROOT / "omiclass").rglob("*.R"))
    code += list(Path(__file__).parent.glob("*.py"))
    (out / "code_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)): digest(p) for p in code}, indent=2))
    common = [sys.executable, "-u", "-m", "experiments.reference_preprocessing.run"]
    options = ["--prepared", str(args.prepared.resolve()), "--method", args.method,
               "--n-factors", str(args.n_factors), "--cores-per-seed", str(args.cores_per_seed)]
    if args.r_library:
        options += ["--r-library", str(args.r_library.resolve())]
    if args.rscript:
        options += ["--rscript", str(args.rscript.resolve())]
    progress = dict(method=args.method, stage="preprocessing", seeds={str(s): "pending" for s in SEEDS}, seed_cpu_ids={})
    progress_lock = threading.RLock()
    stopped = threading.Event()
    stop_signal = [signal.SIGINT]

    def request_stop(signum, frame):
        stop_signal[0] = signum
        stopped.set()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)

    def record():
        with progress_lock:
            temporary = out / "progress.tmp"
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(out / "progress.json")

    def update_seed(seed, status, cpus=None):
        with progress_lock:
            progress["seeds"][str(seed)] = status
            if cpus is not None:
                progress["seed_cpu_ids"][str(seed)] = cpus
            record()

    record()
    with (out / "preprocessing.log").open("w") as stream:
        preprocessing_cpus = cpus[:args.cores_per_seed]
        pre_env = dict(env, MPLCONFIGDIR=str(out / "shared/matplotlib"),
                       JOBLIB_TEMP_FOLDER=str(out / "shared/joblib"))
        returncode = run_owned_process(common + ["preprocess"] + options + ["--out", str(out / "shared"),
            "--cpu-list", ",".join(map(str, preprocessing_cpus))],
            cwd=ROOT, env=pre_env, stream=stream, stopped=stopped)
    if returncode or stopped.is_set():
        progress.update(stage="stopped" if stopped.is_set() else "preprocessing_failed", returncode=returncode)
        record()
        raise SystemExit(128 + stop_signal[0] if stopped.is_set() else returncode)
    progress["stage"] = "classification"
    progress["seeds"] = {str(s): "queued" for s in SEEDS}
    record()

    def train(seed):
        cpus = slots.get()
        try:
            if stopped.is_set():
                return -signal.SIGTERM
            update_seed(seed, "running", cpus)
            seed_env = dict(env, MPLCONFIGDIR=str(out / f"seed{seed}/matplotlib"),
                            JOBLIB_TEMP_FOLDER=str(out / f"seed{seed}/joblib"))
            with (out / f"seed{seed}.log").open("w") as stream:
                return run_owned_process(common + ["run"] + options + ["--seed", str(seed),
                    "--preprocessing-cache", str(out / "shared"), "--out", str(out / f"seed{seed}"),
                    "--cpu-list", ",".join(map(str, cpus))],
                    cwd=ROOT, env=seed_env, stream=stream, stopped=stopped)
        except Exception as exc:
            (out / f"seed{seed}.launch_error.json").write_text(json.dumps(dict(error=type(exc).__name__, message=str(exc))))
            return 1
        finally:
            slots.put(cpus)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(train, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed, returncode = futures[future], future.result()
            update_seed(seed, "complete" if returncode == 0 else (
                "stopped" if stopped.is_set() else f"failed:{returncode}"))
            print(args.method, seed, progress["seeds"][str(seed)], flush=True)
    success = all(v == "complete" for v in progress["seeds"].values())
    progress["stage"] = "stopped" if stopped.is_set() else ("complete" if success else "classification_failed")
    record()
    if stopped.is_set():
        raise SystemExit(128 + stop_signal[0])
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
