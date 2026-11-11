# Installation and validation

Use a new, user-owned environment. The current baseline is Linux/Python 3.9, NumPy 1.26, pandas 2.2, SciPy 1.13, sklearn 1.5, imbalanced-learn 0.12, XGBoost 2.1 and LightGBM 4.5. Package bounds preserve this dependency family; they are not a guarantee of arbitrary binary/model compatibility. A fresh network dependency solve and Docker build require separate verification.

From the source root, `python -m pip install .` installs core dependencies. `python -m pip install -e .` supports development. To build an archive/wheel in a new environment, use `python -m pip wheel --no-deps .`; build isolation may obtain setuptools/wheel. Neither command downloads model checkpoints.

The build uses setuptools namespace-package discovery and keeps the existing top-level `api` module. R scripts are package resources in `omiclass.batch` and `omiclass.de`. The standalone BulkRNABert worker is included as Python source. Gene annotations, surface-marker data and old examples are deliberately omitted pending permissions; see [licensing](licensing.md).

## Dependency groups

| Group | Scope |
| --- | --- |
| Core | Declared NumPy/pandas/SciPy/sklearn, RF/XGBoost/LightGBM/skopt/imbalanced-learn, configuration, plotting, requests, statsmodels and lifelines dependencies. No R import is needed merely to enumerate the API catalogue. |
| `foundation` | Torch for existing CPU neural heads. This alone is not a complete encoder installation. |
| `bulkformer` | Torch, Torch Geometric, torch-sparse and performer-pytorch. Sparse wheels must match Torch/CUDA/platform; source compilation may be required. Encoder source/assets are separate. |
| `cellfm` | Common Torch dependency only. The exact compatible PyTorch port and its MindSpore/Scanpy/Numba dependencies must come from a verified port manifest; this remains an acquisition blocker. |
| `r` | Python rpy2 bridge only. Install compatible R and its packages separately. |
| `io` | Parquet/Excel dependencies (`pyarrow`, `openpyxl`). |
| `test` | pytest and wheel. Neural tests also need `foundation`. |

BulkRNABert runs in an explicitly configured separate interpreter with JAX, Haiku (`dm-haiku`), NumPy, pandas, joblib and the matching `multiomics_open_research` installation. Follow the pinned upstream installation requirements in that separate environment. No guessed JAX/MindSpore version is presented as verified. [Encoder setup](encoders.md) lists required assets and unresolved revision checks.

## R

`environment.yml` declares R 4.3, rpy2, jsonlite, limma, edgeR, sva and BiocParallel alongside core dependencies. Create it from the source root, activate `omiclass_env`, then `python -m pip install --no-deps -e .`. The file does not install the package automatically or reference a research-server environment.

RNA DE uses limma-voom/edgeR through rpy2. Transductive correction uses limma/jsonlite. Native exact fSVA uses a compatible `Rscript`, sva, jsonlite and BiocParallel; configure the executable/library explicitly and retain the fitted sva version for replay. Missing dependencies raise errors rather than install packages or change scientific settings. Native-R numerical validation is distinct from tests of dependency guards or mocked R bridges.

## Public validation

Run `python -m pytest --collect-only` before `python -m pytest`. The suite has no dependency on manuscript launchers or datasets. Set `PYTHONDONTWRITEBYTECODE=1`, `MPLCONFIGDIR`, `XDG_CACHE_HOME` and `TMPDIR` to writable validation-only paths, disable CUDA for CPU checks, and limit OMP/OpenBLAS/MKL threads. Tests write synthetic artifacts under pytest's temporary directory; do not point them at saved results.

`OMICLASS_TEST_SVA_LIB=/path/to/compatible/R/library python -m pytest -m native_r` enables the native-fSVA numerical integration check. A missing setting produces an explicit skip. A passing mocked-encoder API matrix, CPU head fit, or R dependency-guard test is not a real pretrained encoder/native-R scientific run.

Wheel validation must import from the installed artifact outside the checkout, verify required `.R` files, enumerate all 104 configurations, and replay a saved synthetic model. Separately test real encoder assets, GPU kernels, native correction and resource-dependent annotation paths when authorized assets/environments are available. No manuscript benchmark is needed for these bounded checks.

## Docker and compatibility

Docker uses `environment.yml`; it does not reference an internal Conda environment export. It includes both DE/correction R code and package metadata, excludes local validation/quarantine files and unapproved data, and writes provenance into the mounted results directory by default. It is a proposed core image until a fresh build/native-R smoke test passes. No claim is made that its default environment supplies all foundation encoders.

Saved estimator import paths are stable. Dependency-version compatibility is limited to the tested environment family. Foundation bundle signatures currently bind absolute encoder/resource paths as well as contents, so moving identical assets can be rejected. Keep the saved layout or treat relocation as unsupported pending a separately reviewed change; do not edit signed manifests or disable checks. Historical bundles retain their original input-reader and CellFM sampling conventions.

The proposed GitHub workflow runs CPU synthetic tests and a wheel build with read-only repository permissions. It has no publication/deployment job, checkpoint downloader or manuscript launcher. It is prepared configuration, not evidence of a completed remote CI run; it executes only if this candidate is later pushed after approval.

Core marker examples use prepared expression explicitly. The separately approved correction makes new raw-count/no-DE/no-batch training use the existing normalization and persist its lengths. Older runs that recorded ready-expression mode keep that historical scale. Prediction replays the saved mode, rejects explicit mismatches, and never silently normalizes an old ready-expression model.
