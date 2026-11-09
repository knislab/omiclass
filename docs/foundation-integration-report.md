# Step 2 integration report

Implemented only in `/data01/home/ls/katzm/omiclass-public`, on `public-release`, with HEAD still `5a64f41f619b1daf76c580460c8433bcc58673ee`. Changes are uncommitted. No original checkout files or experiment processes were modified by this work; no branches or shared Git refs were moved. No Step 3 fixes or publication cleanup were performed.

## Integrated configurations and invocation

[The complete invocation guide](foundation-api.md) supplies Python/CLI commands, backend resource options, correction/prediction examples and protocol defaults. [The pre-implementation plan](foundation-integration-plan.md) records stages and source evidence.

Use `api.run(expression, foundation={...}, meta_path=..., classification_column=..., output_dir=..., backend_resource_options...)`. Each row below expands only the listed combinations; `api.foundation_configurations()` returns the full 110-entry catalogue, including one separately validated experimental BERT-token entry.

| Model | Pooling | Representation | Classifier |
|---|---|---|---|
| BulkFormer (`bulkformer`) | `mean`, `max_mean_median` | `full`, `supervised`, `pca`, `supervised_pca` | `omiclass`, `lr`, `svm`, `mlp`, `mlp_smote` |
| CellFM (`cellfm_1view`, `cellfm_20view`) | `mean` | Same four pooled representations | Same five heads |
| BulkRNABert (`bulkrnabert`) | `mean` | Same four pooled representations | Same five heads |
| BulkFormer and both CellFM settings | `mean` | `tokens` | `mlp` |
| Both CellFM settings | `mean` | `cls` | `svm`, `mlp`, `mlp_smote`, marked diagnostic |
| BulkRNABert | `mean` | `tokens` | `mlp`, marked experimentally validated without completed full-scale manuscript fits |

`max_mean_median` preserves the original elementwise sum. Existing BulkFormer `max`/`median` pooling and arbitrary positive CellFM view counts are retained as implemented extensions rather than matched manuscript evidence. Both uncorrected and explicitly acknowledged historical limma-before-extraction workflows are available. Use `foundation.correction='limma'`, `allow_transductive_correction=True` and explicit batch/design metadata for the latter.

The Omiclass route reuses the existing Trainer's applicable plain/weighted RF, SMOTE/ADASYN/SVM-SMOTE/borderline-SMOTE RF, plain/weighted XGBoost and LightGBM, and stacker. Its original balanced/mild/severe dispatch is unchanged. Standalone LR has nested CV and fold-local representation/scaling. SVM remains fixed-parameter. Article MLP/token heads retain fixed epochs and view handling. BERT pooled MLP retains inner-validation selection and optional inner-training-only SMOTE, without an all-training refit. No nonexistent CV score is substituted for fixed-head protocols.

Every run saves configuration, split, training-scope records, holdout results, class mapping and model artifacts. `api.predict(external_expression, result.out_dir, separate_output_dir)` verifies the model/configuration and supplied encoder asset identities, reloads fitted transforms and heads, and writes only a new prediction run. Corrected prediction requires the full original cohort plus external samples and performs the historical full-input re-correction; it does not invent transferable correction coefficients.

## Unsupported configurations

- Non-CellFM CLS, or CLS with LR/Omiclass: no observed compatible protocol.
- Full-token SVM, LR, SMOTE or native suites: no matched aligned-token training protocol.
- CellFM/BERT max/median pooling, alternative BERT layers/checkpoint shapes: not recovered supported extraction protocols. Only the observed gtex/encode BERT checkpoint is claimed here.
- Covariate-matrix/PC correction or frozen-reference correction on foundation features: not represented by the observed pre-extraction adapters.
- Non-RNA-seq foundation inputs and outcome-dependent correction: rejected by configuration validation.
- GPU neural-head training: the public adapter exposes the recovered CPU head protocol. Encoder GPU selection remains available through the existing backend settings.
- Missing vendor assets, checkpoint weights, JAX/Haiku or R packages: provision separately; this integration performs no installation/download or scientific fallback.

No external benchmark tools (CPOP, RankTreeEnsemble, Multi-class Pairs, MINT, etc.) are incorporated.

## Tests and evidence

Final combined run: **222 passed, 1 skipped**, with existing pandas/SciPy warnings. The skip is `tests/test_fsva_reference.py:25`, requiring `OMICLASS_TEST_SVA_LIB` for native sva R integration. No checkpoint downloads, real encoder inference or full manuscript experiments ran.

The combined run covered three new modules (167 tests) and eight self-contained existing modules (55 passed, one skipped):

```bash
/tmp/omiclass-public-step2-venv/bin/python -B -m pytest \
  tests/test_foundation_api.py tests/test_foundation_protocols.py tests/test_foundation_extraction.py \
  tests/test_bulkformer_config_runtime.py tests/test_bulkformer_features.py \
  tests/test_bulkformer_reduction.py tests/test_cellfm_backend.py \
  tests/test_pre_348201d_training_behavior.py tests/test_reference_integration.py \
  tests/test_fsva_reference.py tests/test_expression_reference.py \
  -q -ra -p no:cacheprovider --basetemp=/tmp/omiclass-step2-final-combined-tests
```

The process environment disabled bytecode and CUDA, limited BLAS/OpenMP threads to one, and redirected matplotlib/XDG caches to separate `/tmp` paths. The separate venv inherited existing packages read-only; no package in viz_env was installed or upgraded.

New tests exercise all 110 catalogue entries through training, saving, loading and external prediction with mocked encoders and shortened synthetic neural fits; binary/multiclass class ordering; schema and CLI dispatch; invalid combinations; inner/outer CV scope; SMOTE scope; saved model/config/asset checks; corrected-input ordering and external re-correction requirements; token masks/view coverage; instance-local callbacks; and marker-route compatibility. Native full-suite tests use lightweight trainer-method stand-ins to validate integration/dispatch. Six actual RF/resampling pipelines additionally fit within synthetic CV and reproduce predictions after joblib reload. Full-size native boosting/stacking is not claimed as tested.

Nine additional read-only historical parity checks passed: all four representation strategies yielded bitwise-identical numerical matrices; original and promoted BERT MLP/SMOTE loops produced bitwise-identical weights and training metadata; original and promoted article vector/SMOTE/token loops produced bitwise-identical weights and predictions. All parity inputs and outputs were synthetic and temporary.

Two older test modules could not collect because the public worktree intentionally lacks ignored manuscript-only packages: `test_bulkrnabert_backend.py` imports `final_results_runs.test_bulkrnabert.prepare`; `test_bulkformer_umap_separation.py` imports a manuscript plotting script. They were recorded as unavailable rather than copied into the framework or counted as passing. New tests cover the public BERT dispatch/worker contract without those packages. Real R/limma, JAX/checkpoint and GPU numerical replay remain outside this lightweight verification.

Test logs: `/tmp/omiclass-step2-final-combined-tests.log`; parity evidence: `/tmp/omiclass-step2-historical-parity.json` and `.log`; isolation evidence: `/tmp/omiclass-step2-isolation-verification.json`.

## Differences from historical implementations

- New opt-in schema, compatibility catalogue, output ownership and API lifecycle; default marker/legacy paths remain unchanged.
- Manuscript components promoted into importable public modules with source hashes, avoiding runtime imports from ignored manuscript directories.
- Instance-local extraction observers expose existing forward outputs; pooled extraction defaults remain unchanged. The BERT worker adds explicit already-log-transformed input and streamed token-export modes.
- Shared existing representation code replaces the standalone representation copy. Numerical outputs matched bitwise in parity checks; PCA column names use the existing core prefix rather than the standalone prefix.
- Portable neural model serialization, canonical probability-column alignment, encoder-asset identities and immutable saved-prediction inputs are new API contracts. Native adapter log decisions and constant-column lists are persisted for consistent prediction.
- New CellFM foundation calls select the saved batch size 4 by default; explicitly disabled/legacy foundation calls retain their prior defaults. Fixed/head-specific protocols remain separate; no extra training, CV or refit is added.
- No intentional algorithmic change to the source head loops, selector/PCA, original Trainer, correction semantics, clipping, token architecture, CLS values or scientific defaults.

## Remaining Step 3 issues

Review transductive/pre-split correction and external re-correction; BERT corrected-input zero flooring; interpretation of BulkFormer's pooled sum; positional token alignment; nearly constant CellFM CLS diagnostics; native calibration/stacking/imbalance and configuration-selection choices; and distinctions between fixed-head, nested-CV and inner-validation protocols. Fixed-partition repeated head seeds must not be confused with repeated cohort splits. None of these methodological questions was silently resolved here.

## Isolation verification and running experiments

Original `main` remains clean at `41b71d9`; its HEAD/index/config are byte-for-byte unchanged. Existing `omiclass-dev` files and uncommitted changes match the pre-worktree baseline. All code edits and added documentation/tests are inside the public worktree. No original missing/different pinned source files were restored. No experiment or worker was stopped/restarted.

The local original launcher PID 678519 remains alive and workers continue from original paths. Latest inspected shared logs report 10 active / 86 failed on knislab02 and 21 active / 206 failed on knislab04, with completion counters 102. These are per-launcher snapshots, not additive unique-job counts. The prior source-pin failures remain unresolved, as requested; remote process status is inferred from its advancing shared log.

A broader check examined 5,918 original paths. Six ignored files in `experiments/knn_purity_als_ibd_20261109` changed during the session: README.md, figure_validation.json, output_manifest.json, plot.py, report.py and validate.py. This work made no writes to those paths. None occurs in the inspected active repeated-seed plan's pinned inputs. Those concurrent changes were neither reverted nor modified, so an assertion that every ignored manuscript file remained unchanged would be incorrect.

## Files changed

| Status | Path |
|---|---|
| Modified | [README.md](../README.md) |
| Modified | [api.py](../api.py) |
| Modified | [omiclass/config/loaders.py](../omiclass/config/loaders.py) |
| Modified | [omiclass/config/overrides.py](../omiclass/config/overrides.py) |
| Modified | [omiclass/config/schema.py](../omiclass/config/schema.py) |
| Modified | [omiclass/features/bulkformer.py](../omiclass/features/bulkformer.py) |
| Modified | [omiclass/features/bulkrnabert_worker.py](../omiclass/features/bulkrnabert_worker.py) |
| Modified | [omiclass/features/cellfm.py](../omiclass/features/cellfm.py) |
| Modified | [omiclass/pipeline/runner.py](../omiclass/pipeline/runner.py) |
| Added | [docs/foundation-api.md](../docs/foundation-api.md) |
| Added | [docs/foundation-integration-plan.md](../docs/foundation-integration-plan.md) |
| Added | [docs/foundation-integration-report.md](../docs/foundation-integration-report.md) |
| Added | [omiclass/foundation/SOURCE_PROVENANCE.md](../omiclass/foundation/SOURCE_PROVENANCE.md) |
| Added | [omiclass/foundation/__init__.py](../omiclass/foundation/__init__.py) |
| Added | [omiclass/foundation/article_head.py](../omiclass/foundation/article_head.py) |
| Added | [omiclass/foundation/bert_head.py](../omiclass/foundation/bert_head.py) |
| Added | [omiclass/foundation/catalogue.py](../omiclass/foundation/catalogue.py) |
| Added | [omiclass/foundation/cellfm_corrected.py](../omiclass/foundation/cellfm_corrected.py) |
| Added | [omiclass/foundation/correction_compat.py](../omiclass/foundation/correction_compat.py) |
| Added | [omiclass/foundation/extraction.py](../omiclass/foundation/extraction.py) |
| Added | [omiclass/foundation/heads.py](../omiclass/foundation/heads.py) |
| Added | [omiclass/foundation/io.py](../omiclass/foundation/io.py) |
| Added | [omiclass/foundation/runner.py](../omiclass/foundation/runner.py) |
| Added | [tests/test_foundation_api.py](../tests/test_foundation_api.py) |
| Added | [tests/test_foundation_extraction.py](../tests/test_foundation_extraction.py) |
| Added | [tests/test_foundation_protocols.py](../tests/test_foundation_protocols.py) |

9 modified and 17 added files; no commits or index changes were made.
