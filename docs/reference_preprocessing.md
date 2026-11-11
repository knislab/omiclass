# Core preprocessing scope and frozen references

The core route's existing `batch_correction.algorithm` values are `limma`, `none`, `frozen_quantile`, `fsva_exact` and experimental `reference_ruv`. They are not interchangeable statistical procedures. No reference method has been added as a replacement for the intentionally transductive foundation limma workflow.

## Split and fitting ownership

A single immutable labelled population and train/holdout split is resolved before supervised DE or learned preprocessing. DE reporting uses training samples and does not supply a global gene list to CV. Existing limma-voom selection, importance selection, learned transformations and native classifier recipes fit inside their permitted inner/outer/calibration/stacking/final training partitions. Validation/holdout labels do not enter their corresponding supervised fits.

For frozen-reference core workflows, `batch_correction.training_samples_path` must list exactly the classifier training partition. Without another explicit/reused split, those IDs define training and the other eligible labelled samples define holdout. With an explicit split, a disagreement fails. Reference fitting is deferred to the estimator's training scope; internal holdout samples are not a reference-fitting population. Each CV fit learns its own reference, rather than reusing one globally fitted reference. The final estimator retains its training-fitted transform.

The foundation route supports only its existing optional cohort limma adapter, not these frozen alternatives. Reference workflows require non-embedding RNA-seq with explicit `tpm` or natural-log `log1p_tpm` scale. Direct raw counts are rejected on this route; first use a consistent verified gene-length/measured-universe normalization protocol. Do not label log2/log10 data as natural-log expression.

## Transductive limma

The supplied expression cohort enters the existing limma batch correction. Resolving the classifier split does not turn correction into an inductive method or restrict that correction cohort. Preparatory CPTAC transformations learn their permitted training state, while the intended cohort correction is still transductive. Outcome variables must not be passed as correction covariates.

At external prediction, historical limma re-correction uses the supplied compatible cohort according to its saved protocol. It does not transfer invented frozen limma coefficients. Cohort composition can affect corrected expression and predictions; classifier preprocessing/selection is still replayed from fitted state. Report such evaluation as transductive.

## Existing frozen methods

| Method | Existing fit and application |
| --- | --- |
| `frozen_quantile` | Fit the arithmetic mean of training-column sorted log-expression ranks. Map each incoming sample to that fixed target with the existing tie/rank handling. This is ordinary frozen-target quantile normalization, not fRMA. Requires fixed feature coverage/comparable distributions; ties can leave differences. |
| `reference_ruv` | Existing continuous-expression control-gene factor/regression procedure. Require explicit justified controls and fixed factor count; do not choose them using held-out outcomes/performance. Incoming nuisance effects must be representable by the fitted control relationships. It is experimental, not a count GLM or a new DE method. |
| `fsva_exact` | Native `sva::sva` plus `sva::fsva(method='exact')`, with its training database, weights and biological design fitted in the current training scope. Training-constant genes pass through unchanged. Incoming surrogate scores are recomputed per patient against the fixed database; the saved classifier/database remain unchanged. This is not the same fixed projection as the other two methods. |

Perfect biological/technical confounding, invalid controls, new nuisance mechanisms or mismatched feature coverage cannot be repaired by assuming a reference exists. Native fSVA requires the same compatible sva/R implementation for replay and sufficient samples/variable genes/design rank. Its existing `fast` alternative remains unsupported.

## Configuration and prediction

Use the existing configuration fields, for example:

```python
settings = {
    'batch_correction.algorithm': 'frozen_quantile',
    'batch_correction.input_scale': 'log1p_tpm',
    'batch_correction.training_samples_path': 'inputs/training_ids.txt',
}
```

For RUV, additionally supply the existing `control_genes_path`, `control_justification` and `n_factors` fields. For native fSVA, use explicit compatible `rscript_path`, `r_library_path`, design columns and factor count. No R package is installed automatically. Verify all field spellings against `omiclass.config.schema.BatchCorrectionCfg` and the public aliases when building a run.

New saved core estimators own their fitted preprocessing. `preprocessing/scoped.json` records fitting ownership and train/test IDs. External prediction loads that state, restores required feature order, and rejects missing required features; it does not refit a reference, DE selector, imputer, scaler or head. For old standalone reference artifacts, the existing `preprocessing/reference.npz` compatibility path remains available. Do not describe that historical artifact as the only storage layout for newly fitted models.

Core prediction uses `api.run` with the actual completed saved run directory as `output_dir`; see the [README](../README.md). Foundation `api.predict` instead uses a new output directory and follows its own explicitly transductive correction contract. Core raw-count prediction reuses saved normalization state when present; historical artifacts lacking required state have explicit limits or fail rather than silently reconstructing it from prediction samples.

## Resources and evidence

Gene annotations and example datasets are excluded from the publication candidate pending permission review. Supply permitted normalization resources; do not replace a missing annotation with fabricated lengths or mappings. Native R execution, mocked bridge routing, synthetic label-boundary checks and real estimator fitting are separate validation categories. Tests do not launch manuscript experiments. See [installation](installation.md) and [licensing](licensing.md) for current validation/resource limits.
