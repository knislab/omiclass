"""Runner-only CPU affinity and thread limits; no numerical imports at startup."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path


def thread_environment(cores):
    if cores < 1:
        raise ValueError("cores must be positive")
    result = {name: "1" for name in (
        "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
        "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS",
        "RCPP_PARALLEL_NUM_THREADS", "MC_CORES", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
    result.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", LOKY_MAX_CPU_COUNT=str(cores))
    return result


def core_key(cpu):
    """Avoid assigning two SMT siblings as two physical cores."""
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        if package >= 0 and core >= 0:
            return package, core
    except (OSError, ValueError):
        pass
    return "logical", cpu


def available_cpus():
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("CPU-budgeted experiments require Linux sched_getaffinity")
    result, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        key = core_key(cpu)
        if key not in seen:
            result.append(cpu)
            seen.add(key)
    return result


def parse_cpu_list(value):
    cpus = []
    try:
        for part in value.split(","):
            if "-" in part:
                first, last = map(int, part.split("-"))
                if first < 0 or last < first:
                    raise ValueError
                cpus.extend(range(first, last + 1))
            else:
                cpus.append(int(part))
        if not cpus or min(cpus) < 0 or len(set(cpus)) != len(cpus):
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("CPU list must contain unique nonnegative IDs/ranges, e.g. 0-9 or 0-4,10-14") from None
    return cpus


def method_cpus(method, workers=2, cores=5, cpu_list=None):
    cpus = parse_cpu_list(cpu_list or ("0-9" if method == "frozen_quantile" else "10-19"))
    needed = workers * cores
    if cores < 1 or workers < 1 or len(cpus) < needed:
        raise ValueError("--cpus must provide at least workers * cores-per-seed CPU IDs")
    cpus = cpus[:needed]
    if len(set(map(core_key, cpus))) != needed:
        raise ValueError("Choose distinct physical cores, not SMT siblings")
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("--cpus includes CPUs outside the inherited server/scheduler affinity")
    return cpus


def bind_cpus(cpus):
    cpus = set(cpus)
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise ValueError("Requested CPUs are outside the inherited affinity allocation")
    os.sched_setaffinity(0, cpus)


def bootstrap(argv):
    """Called before numpy/joblib/R imports, including for direct runner use."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cores-per-seed", type=int, default=5)
    parser.add_argument("--cpu-list")
    args, _ = parser.parse_known_args(argv)
    os.environ.update(thread_environment(args.cores_per_seed))
    if "--help" in argv or "-h" in argv or not argv or argv[0] not in {"run", "preprocess", "validate", "evaluate"}:
        return
    cpus = parse_cpu_list(args.cpu_list) if args.cpu_list else available_cpus()[:args.cores_per_seed]
    if len(cpus) != args.cores_per_seed or len(set(map(core_key, cpus))) != len(cpus):
        raise ValueError("Insufficient distinct allocated cores for --cores-per-seed")
    bind_cpus(cpus)


@contextmanager
def execution_limits(cores):
    """Limit loaded native pools and one joblib level without altering models."""
    from threadpoolctl import threadpool_limits
    from joblib import parallel_backend
    from joblib._parallel_backends import ThreadingBackend, SequentialBackend

    class SeedThreads(ThreadingBackend):
        def effective_n_jobs(self, n_jobs):
            return min(cores, super().effective_n_jobs(n_jobs))

        def get_nested_backend(self):
            return SequentialBackend(nesting_level=self.nesting_level + 1), 1

    os.environ.update(thread_environment(cores))
    with threadpool_limits(limits=1), parallel_backend(SeedThreads(), n_jobs=cores):
        yield
