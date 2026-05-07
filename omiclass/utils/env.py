# omiclass/utils/env.py

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union
import warnings

def setup_logging():
    def _setdefault_env(name: str, value: str = "1"):
        if name not in os.environ:
            os.environ[name] = value

    # Generic BLAS/OMP caps (Linux/Windows/macOS; harmless if not present)
    for var in [
        "OMP_NUM_THREADS",         # OpenMP (many libs)
        "OPENBLAS_NUM_THREADS",    # OpenBLAS (NumPy/SciPy)
        "MKL_NUM_THREADS",         # Intel MKL (NumPy/SciPy on Intel builds)
        "BLIS_NUM_THREADS",        # BLIS (some NumPy builds)
        "NUMEXPR_NUM_THREADS",     # numexpr (sometimes pulled in by pandas)
        "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate (macOS)
        "KMP_INIT_AT_FORK",        # Intel OMP fork-safety; keep at 0
    ]:
        _setdefault_env(var, "1" if var != "KMP_INIT_AT_FORK" else "0")

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module="numpy.ma.core",
        message="invalid value encountered in cast",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="skopt.optimizer.optimizer",
        message="The objective has been evaluated at point",
    )
    warnings.filterwarnings(
        "ignore",
        module="lightgbm",
        message=".*stopped training because there are no more leaves.*"
    )
    warnings.filterwarnings(
        "ignore",
        module="lightgbm",
        message=".*No further splits with positive gain.*"
    )



class RedirectStd:
    """Context manager to redirect ALL stdout/stderr to a single log file."""
    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self._stdout = None
        self._stderr = None
        self._fh = None

    def __enter__(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._fh = open(self.log_file, "a")
        sys.stdout = self._fh
        sys.stderr = self._fh
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        if self._fh is not None:
            self._fh.close()

def setup_pipeline_log(log_dir: Union[Path, str, None] = None) -> Path:
    """
    Creates a timestamped log file path like:
        pipeline_20251116_094500.log
    If log_dir is None, uses the current working directory.
    """
    if log_dir is None:
        log_dir = Path.cwd()
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"pipeline_{ts}.log"

    return log_file

