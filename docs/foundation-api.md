# Foundation models through api.py

The opt-in `foundation` block exposes the recovered manuscript protocols. Existing marker and legacy embedding calls remain on their original route. [Step 3 Stage 1](step3-stage1-report.md) adds configuration, selection and artifact reliability fixes; [Stage 2](step3-stage2-report.md) corrects split and preprocessing scopes; [Stage 3](step3-stage3-report.md) corrects fit-local weighting, RF calibration consistency and infeasible CV handling. The remaining methodological boundaries are identified below.

## Evidence and complete configuration set

The saved cancer/non-cancer protocol graph and classification/extraction adapters define the matrix below. The manuscript statistics audit references 820 distinct source roots; 52 available `config.freeze.yaml` files were inspected directly. Their active settings were BulkFormer `mean`/`all`, CellFM 1/20 views, BulkRNABert `bulk_rna_bert_gtex_encode`, four reduction strategies, supervised budget 200 and PCA budget 50. Standalone head scripts and the later full-token validation supply the other protocols. Source hashes and promotion details are in [SOURCE_PROVENANCE.md](../omiclass/foundation/SOURCE_PROVENANCE.md).

| `foundation.model` | Pooling | `foundation.representation` | `foundation.classifier` |
|---|---|---|---|
| `bulkformer` | `mean`, `max_mean_median` | `full`, `supervised`, `pca`, `supervised_pca` | `omiclass`, `lr`, `svm`, `mlp`, `mlp_smote` |
| `cellfm_1view`, `cellfm_20view` | `mean` | Same four pooled representations | Same five classifiers |
| `bulkrnabert` | `mean` | Same four pooled representations | Same five classifiers |
| `bulkformer`, `cellfm_1view`, `cellfm_20view` | `mean` (ignored as an aggregation for tokens) | `tokens` | `mlp` |
| `bulkrnabert` | `mean` | `tokens` | `mlp` — experimentally validated; no completed full-scale manuscript fits |

This is 104 explicit entries, enumerable without loading models:

```python
import api
for configuration in api.foundation_configurations():
    print(configuration)
```

CellFM `mean` is valid-gene pooling within each view, then the mean of view embeddings. Full-token heads train each view with equal loss weight and average **probabilities** over views at evaluation; they do not flatten all views into one sample.

BulkFormer `max_mean_median` maps to the historical `aggregate_type='all'`: **max + mean + median elementwise sum**, not concatenation. Existing max-only and median-only pooling remain available as `foundation.pooling='max'/'median'`, explicitly marked framework extensions. The `cellfm` model name accepts the existing positive `cellfm_num_views`; counts other than 1/20 are implemented extensions, not claims of matched manuscript evidence.

## Train one configuration

Inputs are gene-by-sample CSVs. Metadata has sample IDs in its first column and the named classification column. Paths below are placeholders for independently provisioned assets and new outputs. No assets or environments are downloaded/installed by this API.

```python
from pathlib import Path
import api

result = api.run(
    Path('/inputs/counts.csv'),
    meta_path=Path('/inputs/metadata.csv'),
    classification_column='label',
    output_dir=Path('/outputs/omiclass-public'),
    foundation={
        'enabled': True,
        'model': 'bulkformer',
        'pooling': 'mean',
        'representation': 'supervised_pca',
        'classifier': 'lr',
        'split_path': Path('/inputs/split.json'),
    },
    bulkformer_root_dir=Path('/assets/BulkFormer'),
    bulkformer_device='cpu',
    random_state=80,
)
print(result.out_dir)
```

Select any row of the matrix by changing the four `foundation` settings. Default reduction budgets are 200 supervised dimensions and 50 PCs, capped by the fitted training data. Configure budgets with existing dotted options, for example `**{'bulkformer.reduction.supervised_dimensions': 100}`. Do not provide a contradictory backend reduction strategy or aggregate type.

The split JSON must contain disjoint, unique, nonempty `train_index` and `test_index` lists; all requested samples need labels, and training must contain every canonical class. Without a split path, the existing holdout splitter creates a split using runtime settings. **To reproduce repeated head seeds on fixed manuscript partitions, always reuse the same split file.** `runtime.random_state` controls downstream training; `foundation.extraction_seed` controls BulkFormer extraction, while `cellfm_random_state` controls CellFM views. These are separate from split membership when a split file is supplied.

Resource options for each backend:

| Backend | Required resources/options |
|---|---|
| BulkFormer | `bulkformer_root_dir`; existing checkpoint, graph, graph weights, gene embedding, gene info, gene lengths and interested-gene paths can be overridden individually with their existing `bulkformer_*` aliases. Input is raw counts. |
| CellFM | `cellfm_root_dir`, `cellfm_checkpoint_path`, `cellfm_gene_info_path`, `cellfm_vendor_dir`, `cellfm_device`. Input is original raw counts. Aliases select 1/20 views; new foundation calls use the saved protocol's batch size 4 unless explicitly overridden. Legacy calls retain their prior defaults. |
| BulkRNABert | `bulkrnabert_root_dir`, `bulkrnabert_checkpoint_directory`, `bulkrnabert_vocabulary_path`, `bulkrnabert_python_executable`, `bulkrnabert_device`. The configured Python must already contain compatible JAX/Haiku/tokenizer dependencies. Set `foundation.input_scale='tpm'` for unlogged TPM CSV input. For counts, supply verified `gene_lengths_path` or an explicit `bulkrnabert_tpm_input_path`; no length imputation is performed. |

BulkRNABert uses frozen layer 4 and its original vocabulary/checkpoint shape (19,062 genes, 256 channels, four layers). All token positions, including zero-filled vocabulary genes, remain valid. Token export streams per-sample arrays; it does not retain a whole cohort tensor in memory. Only the observed `bulk_rna_bert_gtex_encode` checkpoint is claimed by this integration. Other encoder checkpoints/layers require separate validation.

A flat CLI invocation uses the same configuration loader:

```bash
python api.py \
  --override expression_file /inputs/counts.csv \
  --override meta_path /inputs/metadata.csv \
  --override classification_column label \
  --override output_dir /outputs/omiclass-public \
  --override foundation_enabled true \
  --override foundation_model cellfm_20view \
  --override foundation_classifier svm \
  --override foundation_representation pca \
  --override foundation_split_path /inputs/split.json \
  --override cellfm_root_dir /assets/CellFM/CellFM-torch \
  --override cellfm_device cpu
```

## Training and evaluation protocols

| Classifier | Protocol, final fitting and selection |
|---|---|
| `omiclass` (`foundation_classifier` alias) | Existing native algorithms: nested CV/tuning, calibration, imbalance dispatch and stacker. Every completed applicable candidate is saved and evaluated on holdout; selection uses the requested `runtime.scoring` training-CV score (default macro-F1), with the historical alphabetical tie break. RF outer CV and stacking base probabilities use its configured calibration. Weighted XGBoost calculates weights within each fit. Infeasible CV/calibration fails explicitly. Nested stacking evaluation corrections remain pending. |
| `lr` | Original L2/lbfgs LR, max_iter 5000; scaler and representation inside each inner/outer fit. Grid C = .01, .1, 1, 10, 100 and class_weight None/balanced; five outer/three inner folds by default, safely limited by class counts. Final grid search refits on all outer training data. `foundation.lr_c_values` supports small test grids. |
| `svm` | Fixed RBF SVC, C=1, gamma=scale, no class weights, probability=True, ovr decision function. Representation and scaler fit only on outer training data; libsvm's probability calibration remains training-only. No fabricated outer CV score or hyperparameter search. Decision predictions are preserved even when they differ from probability argmax. |
| BulkFormer/CellFM `mlp`, `mlp_smote` | Original article head: LayerNorm, [64] GELU/dropout .2, 50 fixed epochs, batch size 8, AdamW lr .001/weight_decay .0001. Pooled reduction fits on outer training; optional SMOTE sees that training set only. No holdout-based epoch selection or CV score. |
| BulkRNABert pooled `mlp`, `mlp_smote` | Original width-adapted 256→128 SELU head, Adam, no scaling/dropout/LayerNorm; max 200 epochs, patience 20, stratified training-only validation fraction .2. Representation and optional SMOTE fit on **inner training only**. Restore best epoch; do not refit on all outer training. |
| All `tokens` + `mlp` | Original fixed-epoch article token head: LayerNorm, shared token projection to 8, GELU, mask, flatten, [64] GELU/dropout .2, logits. Per-view loss and mean view probabilities. Full-token SMOTE/SVM/LR/Omiclass have no matched protocol. |

Head defaults can be overridden explicitly under `foundation.article_head` or `foundation.bert_head`; defaults reproduce the recovered protocol. BERT pooled and article/token MLPs remain distinct implementations. All neural heads in this API train on CPU, matching the recovered repeated-seed head protocol; encoder devices remain independently configurable. GPU head training is not exposed in this integration.

Omiclass's original applicable candidate sets are preserved:

- Balanced: plain RF, LightGBM, XGBoost, stacker.
- Mild imbalance: add weighted RF, weighted LightGBM and weighted XGBoost.
- Severe imbalance: also add SMOTE, ADASYN, SVM-SMOTE and borderline-SMOTE RF.

The existing severity thresholds, candidate sets, search spaces and calibration choices are preserved. Stage 3 makes their application consistent across native candidate fitting and evaluation. `runtime.feature_selection.enabled` must be false for this opt-in protocol, as in the manuscript adapters; representation selection/PCA is the separately fitted backend transformer. Existing marker-based feature selection remains available on the legacy route.

## Historical correction before extraction

Enable explicitly:

```python
foundation = {
    'enabled': True, 'model': 'cellfm_20view',
    'representation': 'full', 'classifier': 'mlp',
    'correction': 'limma',
    'allow_transductive_correction': True,
    'split_path': Path('/inputs/split.json'),
}
# Pass foundation=foundation, batch_column='cohort', design_columns=['sex'],
# metadata and backend assets to api.run(...).
```

The supplied original cohort is corrected before the train/holdout split, as in the historical experiments. Correction is **transductive**; setting the acknowledgment does not make it train-only. Outcome labels cannot be correction variables. Only observed cohort limma with explicit biological design columns is integrated; covariate-matrix/PC correction and frozen-reference alternatives are not mapped onto foundation backends here.

BulkFormer corrects its normalized log-TPM input. CellFM retains original count-derived gene selection, order, padding, unmask channels and ST/library features, substituting corrected continuous scalars only; missing selected genes cause failure. BulkRNABert corrects vocabulary-aligned log10(TPM+1), restores only genes proven all-zero before correction, and applies the historical recorded floor of negative corrected values to zero. No replacement features or new correction algorithm is introduced.

## Saved prediction and output ownership

Every training call creates a new run directory containing the frozen config, split, extracted features/metadata, fit-scope records, leaderboard, holdout metrics/predictions, promoted head artifacts, and portable `models/*.joblib`. `foundation.json` records canonical classes, protocol, selected model, source sample IDs, diagnostic/evidence flags and artifact checksums. Existing outputs are never overwritten.

```python
prediction = api.predict(
    Path('/inputs/external_counts.csv'),
    model_dir=result.out_dir,
    output_dir=Path('/outputs/omiclass-public-predictions'),
)
print(prediction.artifacts['predictions'])
```

For BERT TPM mode, supply an external **unlogged TPM** CSV. When training used a separate `bulkrnabert_tpm_input_path`, prediction does not inherit that cohort file: provide a new `bulkrnabert_tpm_input_path` explicitly, or supply counts with the saved verified gene-length normalization. Prediction reloads the saved config and every fitted model, verifies model/configuration checksums, encoder asset digests and configuration compatibility, extracts frozen features, and reuses fitted transforms/scalers and class order. It never tunes, resamples or fits heads. New native models fit logging decisions and constant-column filters inside each estimator's training partition and persist that state. Historical native artifacts retain their saved outer preprocessing wrapper. New prediction outputs must be outside the saved training run.

New bundles use artifact format 2, including checkpoint contents and upstream Python/vendor implementation identities. Existing format-1 bundles remain readable with their original file checks; their missing source/checkpoint-directory identities cannot be reconstructed retrospectively. `prediction.json` records which verification was available. Saved training bundles are never rewritten.

Default prediction excludes training/holdout IDs and returns external IDs only. To replay inference explicitly on known samples, pass `foundation={'prediction_scope': 'all'}` to `api.predict`; this does not fit anything.

For **corrected** runs, supply the full original cohort plus external samples and corresponding full-input metadata (`meta_path=...`), as required by the original external re-correction protocol. Limma is recomputed on that supplied prediction matrix; no invented reusable correction coefficients are transferred. Only external predictions are returned by default. The output records this distinction. Configurations/feature schemas that differ from the fitted run are rejected.

Original marker calls through `api.run(expression, markers_path=..., ...)` retain their original path and behavior. The new foundation route does not import manuscript comparison tools, benchmark launchers, or ignored experiment packages.

## Unsupported configurations and Step 3 boundaries

Unsupported: token SVM/LR/SMOTE/native suites; CellFM/BERT max/median pooling; non-RNA-seq modalities; outcome-dependent correction; correction methods without an observed adapter; unsupported checkpoint shapes/layers; GPU head training. Missing local vendor/checkpoint/JAX/R dependencies produce errors rather than downloads, silent fallback, or modified scientific settings.

The following remain for subsequent approved stages or explicit discussion: external re-correction; BERT corrected-value clipping; BulkFormer all-pooling sum; positional alignment of CellFM token sequences; and native stacking evaluation/fitting consistency. Explicitly transductive limma correction, existing imbalance thresholds, fixed heads without outer CV and BERT early stopping without all-training refit retain their intended protocols. Historical repeated seeds on fixed splits measure a different source of variability from newly generated splits.

No CPOP, RankTreeEnsemble, Multi-class Pairs, MINT or other external comparison-tool logic is incorporated.
