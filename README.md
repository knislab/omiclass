# Omiclass

Omiclass trains and evaluates the existing marker-based RF, LightGBM, XGBoost and logistic-regression stacked ensemble workflows. The opt-in foundation API exposes 104 pooled/token configurations for BulkFormer, CellFM and BulkRNABert while preserving their separate training protocols.

This is a **private publication candidate**. The software license, annotation/data redistribution permissions and some external encoder assets remain unresolved. No license is granted by this README. Version `0.0.0.dev0` is a packaging-validation placeholder, not a published release.

## Installation

The validation baseline is Linux with Python 3.9. Use a new environment owned by you; do not install into an experiment environment.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

The package retains `api.run`, `api.predict` and `omiclass.*` import paths. The numerical dependency bounds reflect the corrected implementation's tested family of versions; cross-version saved-model compatibility is not guaranteed. For R-backed DE/correction, create the Conda environment from the repository root, then install the source:

```bash
conda env create -f environment.yml
conda activate omiclass_env
python -m pip install --no-deps -e .
```

See [installation and validation](docs/installation.md), [encoder setup and required assets](docs/encoders.md), and [licensing/resource exclusions](docs/licensing.md). Core installation does not install encoder checkpoints, a CellFM port, JAX, or GPU binaries. CPU neural-head tests additionally require the `foundation` extra.

## Inputs and marker-based training

Expression/count matrices have **genes/features in rows and sample IDs in columns**. Metadata has **one row per sample** and a classification column. Core input tables require unique, nonempty, trimmed identifiers; duplicate CSV/TSV headers fail before pandas can rename them. Use CSV/TSV or the existing optional Parquet loader. Foundation expression input is CSV.

Supply your own permitted input files. The minimal marker example uses an already prepared expression matrix on its intended classification scale. Historical example data, annotations, screenshots and generated reports are excluded from the publication candidate pending permission review. No trained models are shipped.

```python
from pathlib import Path
from api import run

result = run(
    Path("inputs/expression.csv"),
    output_dir=Path("outputs"),
    meta_path=Path("inputs/metadata.csv"),
    classification_column="class",
    markers_path=Path("inputs/markers.txt"),
    raw_count=False,
    log_transform=False,
    feature_selection_enabled=False,
    run_de=False,
    **{"batch_correction.algorithm": "none"},
)
print(result.out_dir)  # Keep this actual timestamped run directory for prediction.
```

Markers are one feature ID per line. This minimal example keeps the supplied markers and prepared scale; it does not normalize raw counts or rerank markers. Raw-count DE workflows require verified gene lengths matching the measured universe and the existing normalization protocol. Without a marker list, the existing DE workflow requires compatible R/rpy2, limma, edgeR and jsonlite. Supervised selection and learned preprocessing fit within training/CV partitions. Native nested stacking/calibration needs sufficient samples per class and may reject small partitions rather than substitute in-sample predictions.

For marker/core prediction, supply expression on the same prepared scale and reuse the actual completed run directory returned above:

```python
prediction = run(
    Path("inputs/prediction_expression.csv"),
    output_dir=result.out_dir,
    meta_path=Path("inputs/prediction_metadata.csv"),
)
```

That directory must contain `leaderboard.csv`, `config.freeze.yaml` and fitted model artifacts. A folder containing only HTML is not a saved model. The existing core workflow writes prediction/report artifacts under the saved run. Foundation prediction instead uses a new output directory and `api.predict`; see below. [Runnable argument-based examples](examples/README.md) check the saved-run prerequisites.

## Foundation configurations

```python
import api
rows = api.foundation_configurations()
assert len(rows) == 104
```

There are 41 BulkFormer, 21 CellFM one-view, 21 CellFM twenty-view and 21 BulkRNABert configurations. The [foundation API guide](docs/foundation-api.md) defines the representations, classifiers, training protocols, external assets and prediction contract. Registration and mocked-encoder coverage do not establish real-checkpoint end-to-end validation for every configuration.

**CellFM token heads use sampled positions. A position is not guaranteed to represent the same gene across samples or views.** This limits positional interpretation and may affect generalization. The existing projection/flatten/dense architecture, dimensions and selection rule remain unchanged. CLS support is absent.

## Preprocessing and evaluation

Explicitly transductive limma correction still fits the supplied cohort and re-corrects the supplied prediction cohort. It is not an inductive/frozen transform. Frozen-reference alternatives on the core route fit within training partitions and apply fitted state to holdout/external data. These routes must not be described interchangeably; see [preprocessing methods and boundaries](docs/reference_preprocessing.md).

Ordinary estimator predictions remain primary, including SVM decisions. The core's threshold derivative is fitted from training OOF predictions, without holdout-based retention. The historical validation-ranked feature-reduction diagnostic is disabled. Standalone foundation heads retain their original fixed/nested/early-stopping evaluation protocols.

## Docker

The image provides the core Python/R runtime, not encoder assets or a certified GPU foundation environment. From the repository root:

```bash
docker build -t omiclass:dev .
docker run --rm \
  -v "$PWD/inputs:/app/data:ro" -v "$PWD/outputs:/app/results" \
  omiclass:dev \
  --override expression_file /app/data/expression.csv \
  --override meta_path /app/data/metadata.csv \
  --override classification_column class \
  --override output_dir /app/results \
  --override markers_path /app/data/markers.txt \
  --override raw_count false \
  --override run_de false \
  --override log_transform false \
  --override feature_selection_enabled false \
  --override batch_correction.algorithm none
```

Neither hg19 nor hg38 annotation is silently included while permission review is pending. Supply permitted gene lengths explicitly for raw-count DE workflows; the prepared-expression Docker example above does not perform normalization. Both can be added together only after approval; their omission is a disclosed resource limitation, not a change of normalization method. Runtime provenance is written to `/app/results/runtime_provenance.txt`; set `OMICLASS_RUNTIME_REPORT` if using another writable output mount.

## Tests and remaining publication decisions

```bash
python -m pip install '.[foundation,test]'
python -m pytest
```

Public tests use synthetic inputs. Encoder calls are mocked unless explicitly documented otherwise; some real sklearn/XGBoost/LightGBM/Torch fits run with small test budgets. Native fSVA needs the explicit `OMICLASS_TEST_SVA_LIB` setting. No tests launch manuscript analyses. See [installation](docs/installation.md) for isolated cache/output settings and [resource/licensing blockers](docs/licensing.md). The candidate remains unpublished until the owner approves licensing, resources, validation results and the publication destination.

**Raw-count compatibility:** new core training honors `raw_count=True` even when DE and batch correction are disabled, using the existing gene-length/TPM/log1p normalization and recording its fitted lengths. Supply verified `gene_lengths_path`. Historical runs that recorded ready-expression mode remain on that scale, including runs whose old raw flag was ignored. Prediction reuses the recorded mode; it does not silently reinterpret an old model as raw-count trained.

For new raw-count marker training, use the same API with `raw_count=True`, `gene_lengths_path=Path("inputs/gene_lengths.csv")`, `run_de=False` and the desired existing correction/feature-selection settings. The minimal example above intentionally uses prepared expression instead. For prediction on a new scoped run, omit a conflicting raw-count override and provide the scale recorded by that run.
