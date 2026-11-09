# Step 3 Stage 1 implementation report

Stage 1 is complete in `/data01/home/ls/katzm/omiclass-public`, on `public-release`, starting from `0314c7c`. This report accompanies the reviewed Stage 1 commit, `Fix configuration validation and artifact reliability`. Stage 2 implementation has not started.

## Changes

### Configuration and tuning controls

- Validate user overrides recursively, including nested and optional schema blocks. Unknown fields and conflicting nested/dotted values now produce explicit errors instead of disappearing or depending on argument order. Frozen settings receive the same checks in either form.
- Preserve explicit empty optional blocks: `{}` still instantiates the existing schema defaults, which can differ from automatic search-space defaults.
- Honor existing native `runtime.tuning.enabled=False` and `bayesian_iterations=0` in inner CV, stack-base CV and final tuning. The existing zero-iteration recipe also omits its tuning-coupled feature-selection step. Reports record the effective iteration count. Resolve an unspecified inner-fold count to the existing fallback of three.
- Permit the existing tuning enabled/method controls through override normalization. Reject new training requests for unimplemented `random`/`grid` native searches; no search method was added. Historical prediction configurations can still load those formerly ignored fields.
- Reject explicit positive early-stopping overrides, which the existing trainer disables. Existing stored defaults remain readable; the existing estimator-count search is unchanged.
- Validate new training requests for sklearn scorer names, nonnegative search iterations and valid explicit inner-fold counts. Keep prediction-only loading tolerant of historical training controls, which are unused during inference. Declare the existing stack `passthrough` option with its existing default, `True`.

### Selection and failures

- Record the requested sklearn scorer on each native outer fold and add `selection_score_mean`, `selection_score_std` and `selection_scoring` to native leaderboards. Existing macro-F1/accuracy columns remain.
- Use that scorer consistently for base-family selection, the core selected model and foundation-native selection. Sklearn negative-loss scores retain their higher-is-better convention. Default macro-F1 and alphabetical tie behavior remain.
- Remove silent fallback to another selection metric when the requested score is unavailable. Exclude candidates without usable finite scores or completed artifacts.
- Prevent leftover CV/model state from a failed final fit from qualifying a stack base. Reset per-run candidate state when reusing a Trainer.
- Preserve existing completion policies: the foundation-native suite remains strict; the legacy trainer retains candidate failure records. No partial-results mode was added.

These changes make selection consistent; they do not repair the stacker's globally reused OOF features or its CV/final fitting mismatch. Those remain Stage 4 work.

### Saved foundation runs and prediction inputs

- New foundation bundles use artifact format 2. Individual encoder assets are hashed as before, with added identities for upstream Python sources, CellFM vendor Python/native-library files, CellFM gene-symbol mapping, and the selected BERT checkpoint's `config.json` and `params.joblib`.
- Verify that every saved model and frozen configuration has a checksum, the run is complete, the format is supported, and checksum paths remain inside the saved run.
- Keep format-1 prediction support through the original asset-checking implementation. Prediction provenance explicitly identifies its weaker historical verification. Existing bundle files are not migrated or rewritten.
- Clear inherited training-cohort TPM paths when loading a foundation prediction configuration. An explicitly supplied external TPM path remains supported. Otherwise the existing verified count-to-TPM route is required. Prediction provenance records the supplied expression and separate TPM path.

### R dependency handling

Replace automatic installation in `limma_correction.R` and `limma_voom_de.R` with explicit missing-package errors. Existing correction and DE function bodies are unchanged. Dependencies must be provisioned separately in the intended environment.

## Compatibility and scientific scope

- The catalogue remains exactly 104 configurations. `api.py`, classifier architectures, representation/extraction implementations, model builders and hyperparameter search-space definitions are unchanged.
- The explicitly transductive limma procedure, external re-correction, BERT clipping, BulkFormer sum pooling, fixed-head training and BERT inner-validation protocol remain unchanged.
- Necessary behavior changes are stricter invalid/unsupported setting errors, effective tuning disablement, correct requested-metric selection, exclusion of failed candidates, stronger artifact verification and invocation-specific TPM inputs.
- A synthetic format-1 bundle with the original stack configuration shape loads and predicts successfully. A checksum snapshot verifies that prediction leaves every saved training file unchanged. This is compatibility coverage, not numerical replay of real manuscript checkpoints or all historical environments.
- Format 2 cannot add missing identities to old bundles. Vendor-source identity checks are conservative: changing a hashed upstream source can invalidate a bundle even if that source change would not affect a particular inference.
- No saved estimator classes or neural architectures were changed. CellFM token alignment remains unresolved and untouched; any correction requiring an architecture change must return for discussion.

## Validation

Final combined result: **263 passed, 1 skipped**, with 30 existing-style pandas/SciPy warnings. This comprises the Step 2 228-test passing baseline plus 35 new Stage 1 tests. The skip remains native sva integration, which requires `OMICLASS_TEST_SVA_LIB`.

Coverage includes:

- Train/save/load/external prediction for all 104 configurations using mocked encoders and existing shortened synthetic head fits.
- Actual small RF, XGBoost and LightGBM training, requested probability scoring, disabled inner/final search, and serialization replay.
- Invalid/nested/conflicting configuration handling and preservation of empty search-block defaults.
- Failed candidate exclusion, strict native-suite completion and retained diagnostic records.
- Checkpoint/source/vendor changes rejected before extraction, missing model checksums, version-1 prediction and saved-input immutability.
- R dependency guards exercised with an actual Rscript process and a simulated missing package, without installing or loading model dependencies.
- Existing extraction, representation, sampling, calibration-helper and frozen-reference regressions.

The 13 tested modules are `test_step3_stage1`, `test_foundation_api`, `test_foundation_protocols`, `test_foundation_extraction`, `test_foundation_heads_smote`, `test_bulkformer_config_runtime`, `test_bulkformer_features`, `test_bulkformer_reduction`, `test_cellfm_backend`, `test_pre_348201d_training_behavior`, `test_reference_integration`, `test_fsva_reference`, and `test_expression_reference`.

Final log: `.step3-validation/stage1/verified-tests.log`; pytest artifacts: `.step3-validation/stage1/verified-tests/`. `git diff --check` passes. The original catalogue, encoder/head implementations, model builders and search-space files also pass an unchanged-file comparison against `0314c7c`.

Tests used the existing `/tmp/omiclass-public-step2-venv/bin/python` read-only, with Python 3.9.19, scikit-learn 1.5.2, scikit-optimize 0.10.2, XGBoost 2.1.1 and LightGBM 4.5.0. No packages were installed or upgraded. Bytecode and CUDA were disabled, CPU thread counts limited, and temporary files/Matplotlib/XDG caches redirected inside `.step3-validation/stage1/`. That validation directory is ignored by Git.

No real encoder inference, full native benchmark suites, manuscript experiments, checkpoint downloads, or manuscript-only test imports were performed. The optional native-sva skip remains visible rather than being counted as passing.

## Remaining work and isolation

Split/DE boundaries, fitted preprocessing scope, weighting/calibration consistency, honest stacking, threshold/reporting policy, canonical holdout metrics and CellFM extraction/token issues remain for the approved later stages. Stage 1 does not establish release readiness by itself.

All source edits, documentation and generated validation artifacts are confined to the public worktree. No commands modified the manuscript checkout, running experiments, existing results, shared environments or checkpoints. The initial Stage 1 session created no commit; the reviewed Stage 1 changes are committed only at the user's subsequent explicit request. No push is authorized. Stage 2 remains subject to its approved scope and the explicit stop condition for a substantial fold-local DE redesign.
