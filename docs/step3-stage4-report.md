# Step 3, Stage 4: correct the existing stacked ensemble

Stage 3 was reviewed and committed on `public-release` as **0512624**, with the requested message `Fix classifier weighting and calibration consistency`. Only its nine reviewed files were included; 41 focused pre-commit tests passed. Nothing was pushed. This report accompanies the reviewed Stage 4 commit, `Fix nested stacking and meta-feature consistency`, made at the user's subsequent explicit request. Stage 5 proceeds only under its separate approval.

## Confirmed issues and corrections

| Issue | Correction and code paths |
| --- | --- |
| Meta-CV sliced a globally generated base OOF matrix. Base fits contributing to its meta-training rows could see outer-validation labels indirectly. | `train/stacking_engine.py`: `train_oof_stacker` now builds a fresh base OOF matrix from each outer-training partition using `stacking_oof`, fits the logistic-regression meta-learner there, and predicts outer-validation rows using bases fitted on that outer-training partition. The full-training base OOF matrix is generated separately for final meta fitting and never sliced into outer evaluation. |
| RF/XGBoost/LightGBM variant choices used whole-training CV records before stack CV, so they could also depend on outer-validation labels. | `train/trainer.py`: `_fit_stacking_bases` repeats the existing imbalance dispatch and candidate-selection procedure within each supplied training partition, including each base-OOF training fold. It evaluates eligible variants using the existing nested CV/scorer, retunes/refits the best completed variant, and records local failures. Each family keeps a stable probability slot even when its chosen variant changes. |
| Separate setup paths could drift while rebuilding stacking bases. | `_candidate_setup` is shared by normal RF fitting, boosting fitting and stacking base fits. It preserves parameter routing, existing search-space builders, wrapper prefixes, deterministic seeds, fit arguments, weighted XGBoost behavior, RF calibration settings and existing classifiers/resamplers. `_model_keys_for_labels` shares the original imbalance thresholds and candidate sets with `Trainer.run`. |
| Stack CV omitted raw-feature passthrough while final fitting enabled it by default; their preprocessing and feature layouts differed. | Outer CV and final fitting both use `HonestStackingClassifier.fit_honest`, the configured `passthrough` policy and the same meta-input construction. Passthrough preprocessing is cloned/fitted only on the corresponding meta-training partition; final passthrough preprocessing uses all training rows. Selected passthrough dimensions may differ between fitted folds, as expected for fitted preprocessing, but each fit and its prediction path use the same stored schema. |
| Probability columns relied on implicit class/base ordering; new stack prediction could silently zero-fill missing inputs. | `models/stacked.py`: `aligned_base_probabilities` aligns each base's columns to the stack's canonical class order. `probability_columns` retains every class column, including both binary columns. New fits save probability, passthrough and complete meta-feature names. New prediction uses those schemas, orders each bare base's own features, delegates to fitted partition preprocessors where present, and rejects missing required features, incompatible classes and invalid probabilities. |
| RF consensus truncated the shared stack input, potentially replacing other bases' necessary features with zeros. | `Trainer.run_stacked` no longer truncates the common input to RF consensus genes. Each fitted base retains its own selected features/preprocessor; passthrough retains its own fitted schema. A regression test exercises the historical truncation condition with deliberately disjoint base feature sets. |
| Direct stack `.fit` trained a meta-learner on in-sample base predictions; the trainer also retained an in-sample fitting fallback. | Remove the trainer fallback. Direct `.fit` raises an actionable error requiring training-only OOF probabilities through `Trainer` or the existing `fit_honest` interface. Infeasible nested OOF or requested calibration fails explicitly. No alternate classifier or in-sample replacement is fitted. |
| Holdout/external input matching could apply a single stack-wide feature filter or zero-fill before individual fitted schemas ran. | `eval/holdout.py`: `_match_features_to_model` delegates schema validation to new-format stacks. Historical stack matching remains on its original path. This changes schema handling only; threshold selection and holdout metrics are not refactored in Stage 4. |

The stack still has the existing three base-model families and logistic-regression meta-learner with the existing solver, iteration limit and seed. No architecture, classifier family, search dimension, tuning method or public configuration option is added. The internal fixed-base CV helper remains available and applies the existing `PartitionClassifier` when importance selection needs a fitting boundary.

When a training partition has only one eligible variant in a family, there is no variant-selection CV decision to make; its existing tuning/calibration/refit procedure is still fitted on that partition. When several variants are eligible, local CV ranks them using the existing scorer and alphabetical tie break, and failed final fits are excluded before selecting another completed variant. If a family has no usable candidate, the stack fails instead of changing families or disabling requested calibration.

## Scope and configuration impact

All **104 configurations** are preserved. New stack training/evaluation behavior affects the legacy native framework and the **20 foundation configurations with `classifier='omiclass'`**: BulkFormer has 8 entries across its two pooling choices and four pooled representations; CellFM one-view, CellFM twenty-view and BulkRNABert each have 4 pooled-representation entries. The remaining 84 standalone LR/SVM/neural/token configurations retain their training protocols.

Individual native base construction was factored into shared helpers; its intended algorithms, full-training fitting protocol and configured search spaces remain unchanged. Stack base choices, preprocessing, feature selection, tuning, class weights and calibration now depend only on the current fit's training labels. Existing data-dependent search-space functions use that permitted partition's dimensions rather than imposing ranges derived from a larger cohort. The search-space definitions and budgets themselves are unchanged.

The Stage 2 split and DE implementation, limma-voom algorithm, marker-selection rules, explicitly transductive limma correction, representation architectures, checkpoints, catalogue, configuration schema and public `api.py` signatures are untouched. CLS remains removed. The supplied passthrough setting is honored rather than changed.

New stack runs write `stacking_fit_scope.json` under their model-artifact directory. It records outer and base-OOF sample partitions, local selected variants, candidate scores/failures, final base names and probability/meta/input schemas. This is fitting provenance, not a new prediction option. Model prediction does not depend on this report file.

## Numerical findings

Corrected stack CV scores are expected to change because the previous estimates used indirect label leakage and omitted configured passthrough. Global base-variant choices also no longer influence held-out outer evaluation. Newly fitted meta coefficients, stack rankings, chosen base variants and holdout/external predictions can change. There is no claim that scores must improve, decrease by a particular amount, or reproduce historical manuscript results.

A small comparison against the actual Stage 3 implementation from public commit `0512624` used 48 synthetic samples, a fixed three-tree/depth-two RF base, two folds and unchanged logistic regression, with CV passthrough disabled to isolate the nesting correction:

- Previous global-OOF meta-CV mean negative log loss: **-0.6919788184**.
- Correct nested-stack mean negative log loss: **-0.6904067363**.
- Maximum change in outer OOF probability: **0.1373280537**.
- Full-training base OOF was **bitwise identical** for these fixed base recipes.
- With identical fixed final recipes and OOF, hard predictions were identical for both passthrough settings. Maximum final probability difference was **0** without passthrough and **1.11e-16** with passthrough, reflecting floating-point layout differences in the shared schema path.

These are targeted synthetic correctness checks, not manuscript experiments or performance estimates. Results are in `.step3-validation/stage4/numerical-comparison.json` and `numerical-comparison-final.log`. The initial comparison's bitwise assertion for newly refitted passthrough probabilities detected the `1.11e-16` difference; the final check explicitly permits absolute error up to `1e-14`. Historical artifact reload checks below remain bitwise checks.

An independent manual replay verifies each new outer-fold probability/prediction and requested log-loss score for both passthrough settings. A fixed-split perturbation test changes only an outer-validation fold's labels and verifies bitwise-identical predictions for that same fold. Final full-training OOF is verified separately from outer stack predictions.

## Saved-model and API compatibility

`HonestStackingClassifier` keeps its class/import path. New fits carry an internal `_stack_schema_version_ = 1` marker. Historical objects without that marker use the preserved `_legacy_transform`, retaining original base order, probability order, preprocessing and missing-feature zero-fill behavior. Loading does not fit or migrate them. Existing estimator and label-encoder state is retained.

Compatibility checks include synthetic artifacts created using the **actual Stage 3 class source** from public commit `0512624`, serialized under its original import path, then loaded with the new implementation. Both hard predictions and probabilities are **bitwise identical**, with saved artifact bytes unchanged, for passthrough enabled and disabled. These generated fixtures are entirely inside the public worktree; no saved manuscript model/result was accessed. Log: `.step3-validation/stage4/historical-source-compatibility.log`.

The regression tests also cover historical persisted layouts and zero-fill behavior, new binary/multiclass serialization, reversed base class ordering, per-base disjoint schemas, noncontiguous labels, extra/reordered inputs, fitted preprocessing with irrelevant missing constants, strict required-feature failures, and direct versus saved/external prediction consistency. New stack models retain identical predictions after joblib reload in the available dependency environment. Compatibility across arbitrary sklearn/XGBoost versions is not asserted.

New stack artifacts should be used with Stage 4 code; older code does not understand the new schema contract. Historical loading/prediction compatibility is preserved. Direct `.fit` on a prefit stack is deliberately no longer supported as a training shortcut, because it cannot supply honest base OOF. The public `api.run`/`api.predict` path remains supported and unchanged in signature.

## Validation

The 16-module public-framework regression suite passed **347 tests, with 1 optional native-sva test skipped**, including **21 new Stage 4 tests** and the 326-test Stage 3 baseline. Log: `.step3-validation/stage4/regression.log`. The existing Stage 3 stacking test was updated to expect the extra nested base fits while preserving its weighting and calibration assertions. `git diff --check` passed.

Tests cover sample/label separation, independent nested-CV replay, local variant dispatch and final-fit failure handling, inner tuning/importance selection, training-only DE calls with a synthetic R bridge, sigmoid/isotonic calibration with both ensemble settings, binary/multiclass fit-local weighting, passthrough policy, RF-consensus isolation, probability/feature alignment, explicit infeasibility, serialization and external prediction. Small real RF/XGBoost/LightGBM native suites run in both binary and multiclass settings. The broader suite retains mocked-encoder API train/save/load/external-prediction coverage across all 104 configurations.

No real encoder/checkpoint inference or manuscript experiments ran. Real R/limma numerical execution was not repeated; the Stage 2 R/rpy2 binary limitation remains. The optional native-sva test still requires `OMICLASS_TEST_SVA_LIB`. Existing pandas, skopt/numpy and SciPy warnings remain visible.

## Remaining limits and isolation

- Honest nesting costs more: with K outer folds and K base-OOF folds, the stack calls its base-family fitting procedure K² + 2K times, before counting local candidate CV/tuning/calibration. Full final base fits are reused from completed native candidates. Automatic class-count caps can reduce fold counts. No new cache, approximation or tuning-budget reduction was introduced.
- Small training partitions can be unable to support nested OOF, local variant-selection CV or requested calibration. Such candidates fail explicitly, preserving the existing legacy partial-suite and foundation strict-completion policies. No configuration has been removed to hide this limit.
- The optional saved-estimator feature-reduction diagnostic calls direct `.fit` without a full stacking refit recipe. It cannot produce an honest stack curve and now encounters the explicit OOF-required error; its existing pipeline error-suppression behavior is unchanged. Implementing a new diagnostic/refit interface is outside this stage.
- Threshold selection/holdout reporting remain for Stage 5. Foundation external preprocessing, corrected-value clipping, pooling and token positional concerns remain for the subsequent approved stage. CellFM architecture is unchanged.

All implementation, logs, generated fixtures, temporary directories and caches are confined to `/data01/home/ls/katzm/omiclass-public`, under `.step3-validation/stage4/` for validation artifacts. The existing interpreter is reused read-only; bytecode/CUDA are disabled and numerical threads are limited. No manuscript source, worker, environment/dependency, checkpoint or saved result was modified. The initial Stage 4 session created no commit; the reviewed changes are committed only at the user's subsequent explicit request. No push is authorized.
