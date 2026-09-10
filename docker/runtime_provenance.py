from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import platform
from contextlib import redirect_stdout
import shutil
import sys

import imblearn
import joblib
import lightgbm
import numpy
import scipy
import sklearn
import skopt
import xgboost


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render() -> str:
    numpy_config = io.StringIO()
    with redirect_stdout(numpy_config):
        numpy.show_config()

    manifest = Path("/opt/omiclass-docker/source.sha256")
    lines = [
        f"sys.executable: {sys.executable}",
        f"Python version: {platform.python_version()}",
        f"Python build: {sys.version}",
        f"numpy: {numpy.__version__}",
        f"scipy: {scipy.__version__}",
        f"sklearn: {sklearn.__version__}",
        f"skopt: {skopt.__version__}",
        f"joblib: {joblib.__version__}",
        f"imbalanced-learn: {imblearn.__version__}",
        f"lightgbm: {lightgbm.__version__}",
        f"xgboost: {xgboost.__version__}",
    ]
    if manifest.is_file():
        lines.extend(
            [
                f"source manifest: {manifest}",
                f"source manifest SHA256: {_sha256(manifest)}",
            ]
        )
    for input_path in (
        Path("/app/data/protein_expression_matrix.csv"),
        Path("/app/data/clinical_metadata.csv"),
    ):
        if input_path.is_file():
            lines.append(f"input SHA256 {input_path}: {_sha256(input_path)}")
    lines.append(f"R_HOME: {os.environ.get('R_HOME', '<unset>')}")
    lines.extend(f"{name}: {os.environ.get(name, '<unset>')}" for name in THREAD_VARIABLES)
    lines.extend(["numpy.show_config():", numpy_config.getvalue().rstrip(), ""])
    return "\n".join(lines)


def main() -> None:
    destination = Path(sys.argv[1])
    report = render()
    print(report, flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(f"Runtime provenance saved to: {destination}", flush=True)

    for source, filename in (
        (Path("/opt/omiclass-docker/source.sha256"), "docker_source.sha256"),
        (Path("/opt/omiclass-docker/conda-packages.json"), "docker_conda_packages.json"),
    ):
        if source.is_file():
            copied_to = destination.parent / filename
            shutil.copyfile(source, copied_to)
            print(f"Docker provenance saved to: {copied_to}", flush=True)


if __name__ == "__main__":
    main()
