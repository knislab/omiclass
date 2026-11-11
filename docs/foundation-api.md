# Foundation models through api.py

The opt-in `foundation` block preserves the existing encoder/head protocols. It does not route every classifier through the same CV or training procedure. Existing marker/legacy calls remain on their original route. See [installation](installation.md), [external encoder requirements](encoders.md), [licensing/resource blockers](licensing.md), and the source-adaptation hashes in [SOURCE_PROVENANCE.md](../omiclass/foundation/SOURCE_PROVENANCE.md).

## Supported catalogue

| Model | Pooling | Representations | Classifiers | Count |
| --- | --- | --- | --- | ---: |
| `bulkformer` | `mean`, `max_mean_median` | `full`, `supervised`, `pca`, `supervised_pca` | `omiclass`, `lr`, `svm`, `mlp`, `mlp_smote` | 40 |
| `cellfm_1view`, `cellfm_20view` | `mean` | Same four pooled representations | Same five classifiers | 20 each |
| `bulkrnabert` | `mean` | Same four pooled representations | Same five classifiers | 20 |
| Each of the four model aliases above | `mean` (no pooled aggregation applied) | `tokens` | `mlp` | 4 |

This is 104 entries. `api.foundation_configurations()` enumerates them without loading an encoder. BulkRNABert full tokens retain their experimental-protocol evidence flag: no completed full-scale manuscript fits are claimed. Existing BulkFormer max-only/median-only and non-1/20 CellFM view settings remain implemented extensions, outside the matched catalogue. CLS is unsupported.

CellFM mean-pools valid gene tokens per view, then averages view embeddings. **CellFM full-token positions are sampled positions, not a guaranteed common gene axis across samples or views.** This limits positional interpretation and may affect generalization. Sorting selected IDs cannot remove this limitation without changing membership/dimensions. The existing projection/flatten/dense head and selection rule are retained. Token heads train each view with equal loss weight and average view probabilities at inference.

BulkFormer `max_mean_median` retains the historical elementwise max + mean + median sum, not concatenation. BulkRNABert retains layer 4, width 256 and all 19,062 vocabulary token positions, including zero-expression genes.

## Training

Supply your own permitted expression, metadata, split and encoder assets. Expression CSVs have genes in rows, samples in columns; metadata has samples in rows. The split JSON uses `train_index` and `test_index`; both must be nonempty, unique, disjoint, present and labelled, with all canonical classes in training.

```python
from pathlib import Path
import api

result = api.run(
    Path("inputs/counts.csv"),
    meta_path=Path("inputs/metadata.csv"),
    classification_column="class",
    output_dir=Path("outputs/foundation"),
    foundation={
        "enabled": True, "model": "cellfm_20view",
        "representation": "pca", "classifier": "svm",
        "split_path": Path("inputs/split.json"),
    },
    cellfm_root_dir=Path("assets/CellFM/CellFM-torch"),
    cellfm_checkpoint_path=Path("assets/CellFM/weights/CellFM_80M_weight.ckpt"),
    cellfm_gene_info_path=Path("assets/CellFM/CellFM-torch/csv/expand_gene_info.csv"),
    cellfm_device="cpu",
)
print(result.out_dir)
```

This is a protocol template, not a bundled executable checkpoint example. CellFM Ensembl mapping requires approved annotation resources currently omitted from the candidate. BulkFormer and CellFM use raw counts. BERT uses unlogged TPM with `foundation.input_scale='tpm'`, or counts with verified `gene_lengths_path`/an explicitly supplied `bulkrnabert_tpm_input_path`. Set the BERT worker interpreter explicitly; do not inherit a research-server environment.

Both nested configuration dictionaries and the existing flat CLI aliases are supported:

```bash
python -m api \
  --override expression_file inputs/counts.csv \
  --override meta_path inputs/metadata.csv \
  --override classification_column class \
  --override output_dir outputs/foundation \
  --override foundation_enabled true \
  --override foundation_model cellfm_20view \
  --override foundation_classifier svm \
  --override foundation_representation pca \
  --override foundation_split_path inputs/split.json \
  --override cellfm_root_dir assets/CellFM/CellFM-torch \
  --override cellfm_device cpu
```

The foundation route disables the native marker-feature-selection option; its existing representation selection/PCA is separately fitted. Do not enable a conflicting backend or feature-selection option. New CellFM CSV parsing preserves literal dotted symbols and intentional repeated-gene aggregation. BulkFormer/BERT retain their historical identifier processing. Duplicate sample IDs and ambiguous metadata columns fail explicitly.

## Fitting and evaluation protocols

| Classifier | Existing protocol |
| --- | --- |
| `omiclass` | Native RF/XGBoost/LightGBM/stacker suite with its existing imbalance dispatch, nested tuning/selection, fit-local weighting, RF calibration and nested stacking. Candidate selection uses training CV, not holdout. No in-sample OOF or uncalibrated fallback. |
| `lr` | L2/lbfgs LR with max_iter 5000. C grid .01/.1/1/10/100 and class_weight None/balanced; outer/inner folds default 5/3, constrained by class counts. Representation/scaler fit within each search fit. Final search refits on all training samples. |
| `svm` | Fixed RBF SVC, C=1, gamma=scale, no class weights, probability=True and ovr decision function. Training-only representation/scaler and libsvm calibration. Ordinary SVM predictions remain authoritative when they differ from probability argmax. No invented outer CV score. |
| BulkFormer/CellFM pooled `mlp`, `mlp_smote` | Existing LayerNorm → [64] GELU/dropout .2 head; 50 fixed epochs, batch 8, AdamW lr .001/weight_decay .0001. Representation and optional SMOTE see training only. No holdout epoch selection. |
| BERT pooled `mlp`, `mlp_smote` | Existing width-adapted 256 → 128 SELU head, Adam, no scaler/dropout/LayerNorm; max 200 epochs, patience 20, stratified training-only validation fraction .2. Representation/SMOTE fit on inner training; restore best epoch without all-training refit. |
| `tokens` + `mlp` | Existing LayerNorm → shared projection to 8 → GELU/mask → flatten → [64] GELU/dropout .2 → logits. Fixed article training; equal per-view loss and mean view probabilities. No token SVM/LR/SMOTE/native-suite protocol. |

Neural heads run on CPU, preserving the recovered head protocol. Encoder device settings are independent. Explicit existing head overrides can shorten validation runs; doing so is not a manuscript-performance replication. Standalone heads keep their own metric definitions and absence of outer CV where applicable. Holdouts lacking a class retain the full fitted probability schema; undefined full-task AUROC remains missing.

New CellFM sampling uses the existing SHA256 sample/global-seed/zero-based-view mechanism, including view 0 for single-view. Effective encoder seeds are `cellfm.random_state`, BulkFormer's `foundation.extraction_seed`, and BERT's fixed JAX key 0. Extraction and fitting restore caller RNG/determinism settings, including failures. This does not guarantee bitwise agreement across hardware/dependency versions.

## Explicitly transductive correction

`foundation.correction='limma'` requires `allow_transductive_correction=True`, a batch column and compatible design columns. Outcome labels cannot be correction variables. Correction fits the supplied cohort before extraction, including samples later held out; scores therefore reflect a transductive procedure. This is deliberately preserved and is not the core route's frozen-reference alternative.

BulkFormer corrects normalized log-TPM. CellFM selects original count-derived views and preserves IDs, masks, unmask channels, padding and library auxiliaries while substituting corrected scalar values. Missing selected genes fail. BERT corrects vocabulary-aligned log10(TPM+1), restores only proven-zero removed genes and retains its recorded negative-value floor to zero. The worker does not log twice; official tokenizer normalization/binning occurs once. Out-of-range expression IDs fail instead of upper-clipping.

## Saved prediction and compatibility

```python
prediction = api.predict(
    Path("inputs/prediction_counts.csv"),
    model_dir=result.out_dir,
    output_dir=Path("outputs/external-prediction"),
)
print(prediction.artifacts["predictions"])
```

`api.predict` is foundation-only. It verifies the saved configuration/model checksums, encoder identities and feature/class schema, reloads fitted transforms/models, and creates a separate prediction directory without fitting a head, representation, scaler, sampler or DE selector. Known train/holdout IDs are excluded by default; `foundation={'prediction_scope':'all'}` explicitly replays known samples.

For corrected models, supply the complete original input cohort plus external samples and matching metadata using `meta_path`. Limma is intentionally recomputed on that supplied prediction cohort; only external predictions are returned by default. For BERT TPM input, provide unlogged TPM; a training cohort's separate TPM path is not inherited for external samples.

New bundles keep format 2; formats 1/2 retain their original available checks. Historical bundles without the new input-identifier marker replay the old shared BERT-style reader. Historical CellFM single-view bundles without the stable-sampling marker replay the old global-NumPy/source-order path. New bundles persist both choices. Unknown protocols fail rather than guess. Old bundles are never rewritten by prediction.

New token heads bind dimensions, view counts and cache schemas. BulkFormer/BERT bind fixed gene identity/order; CellFM validates the sampled-position protocol, vocabulary/seed, valid-prefix masks and padding without reselecting tokens. Historical untagged caches/models retain their corresponding limits on identity verification.

**Portability restriction:** saved signatures and identities currently include absolute asset/resource paths. Moving identical encoders or annotations can be rejected. Preserve the saved layout; do not edit signed artifacts to bypass verification. Arbitrary dependency-version compatibility and reconstruction of missing historical provenance are not claimed. Real encoder/GPU/native-R validation and the licensing/resource decisions remain release gates, separately from mocked API coverage.
