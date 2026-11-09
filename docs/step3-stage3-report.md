# Step 3, Stage 3: consistent weighting, calibration and candidate fitting

Stage 2 was reviewed and committed on `public-release` as **1350af7**, with the requested message `Fix split consistency and fold-local feature selection`. The commit contains only the 28 reviewed Stage 2 files. Its focused pre-commit check passed 80 tests. Nothing was pushed.

This report accompanies the reviewed Stage 3 commit, `Fix classifier weighting and calibration consistency`, created at the user's subsequent explicit request. Stage 4 proceeds only under the user's separate approval.

## Changes and affected paths

| Confirmed problem | Correction | Files and functions |
| --- | --- | --- |
| Binary weighted XGBoost used majority/minority counts, which upweighted class 1 even when it was the majority. Counts also came from the entire training cohort rather than the current fit. | Calculate `scale_pos_weight = n(label 0) / n(label 1)` from the actual labels received by each fit. Binary labels retain the framework's existing 0/1 encoding. | `models/xgboost.py`: `FitLocalWeightedXGB.fit`, `build_xgboost_weighted`; `models/registry.py`: existing `weighted_xgboost` entry; `train/trainer.py`: `run_boosting`. |
| Multiclass weighted XGBoost used cohort-wide balanced weights in final/outer fits, omitted weights in some inner searches and importance fits, and lost weighting when rebuilding stacking bases. | Calculate the existing balanced sample-weight formula at the innermost classifier's fit boundary. Every search clone, importance-ranking fit, calibration fit, stacking base and final fit therefore uses its own training labels. Remove cohort-wide weight construction from the trainer. | Same builder/fit paths, plus existing `inner_select_recipe`, `_feature_rank_by_importance`, `PartitionClassifier.fit` and stacking base CV. |
| Fit arguments were dropped in several tuning, ranking and calibrated-CV paths. | Forward existing fit arguments consistently. Slice supplied sample-weight arrays by training-row position at outer CV and importance-ranking boundaries; sklearn search/calibration slices them within its own folds. Remove wrapper prefixes when passing them to the underlying estimator. | `train/cv_engine.py`: `_partition_fit_kwargs`, `inner_select_recipe`, `_feature_rank_by_importance`, `nested_oof_cv`, `calibrate_within_training`; `train/scoped.py`: `PartitionClassifier.fit`. |
| RF outer CV was generally uncalibrated while final fitting used configured calibration. CV also hardcoded sigmoid. Stacking RF bases were rebuilt without calibration. | Use the configured `calibrate`, `method` and `ensemble` settings for RF outer CV and stacking base probabilities. Share the existing calibration fitting helper with final fitting. Preserve both sigmoid/isotonic and the historical ensemble option. | `train/trainer.py`: `run_rf_variant`, `_fit_final`, `run_stacked`; `train/stacking_engine.py`: `train_oof_stacker`; `train/cv_engine.py`: `calibrate_within_training`. |
| Infeasible outer CV could fit/predict on the same rows and return those predictions as OOF. Infeasible calibration silently fitted an uncalibrated model instead. | Raise `CVError` with phase, requested folds and training-class counts. Reject infeasible base/meta OOF before constructing candidates. Reject infeasible requested calibration instead of changing its protocol. | `train/cv_engine.py`: `_safe_stratified_kfold`, `require_stratified_cv`, `nested_oof_cv`, `calibrate_within_training`; `train/stacking_engine.py`: `train_oof_stacker`; `train/trainer.py`: `_fit_final`. |

`FitLocalWeightedXGB` is a small internal subclass of the existing XGBoost estimator that sets fit arguments. It uses the same booster, objectives, trees and prediction implementation. It does not introduce a classifier algorithm, architecture, tuning method, search dimension or public configuration option. Explicit observation weights, if supplied through existing internal fit arguments, remain effective; multiclass balanced class weights multiply them. Ordinary public native training supplies no observation-weight array.

RF and LightGBM balanced-class weighting already operates within their own estimator fits and is retained. Boosting remains uncalibrated under the existing public schema. RF inner hyperparameter searches retain their existing estimator/search objective; the outer evaluation now evaluates the configured calibrated candidate that will be fitted finally.

Existing automatic fold-count reduction remains: requested folds are capped by the minimum training-class count. Outer CV/calibration explicitly requires at least two classes and two folds. Inner tuning can still be skipped when its training partition cannot support a split; it produces no invented inner score. A calibrated candidate can fail on a small outer-training partition even when uncalibrated outer CV would be possible. The framework does not disable requested calibration to make that candidate succeed.

## Configuration and protocol impact

All **104 catalogue entries** are preserved. Changes affect the **20 entries with `classifier='omiclass'`** and the legacy native framework:

- BulkFormer: both existing pooling choices, each with `full`, `supervised`, `pca` and `supervised_pca` (8 entries).
- CellFM one-view and twenty-view: each of the four pooled representations (4 entries per model).
- BulkRNABert: each of the four pooled representations (4 entries).

RF calibration consistency affects native candidates when RF calibration is enabled. Weighted XGBoost changes apply when the unchanged imbalance dispatch selects that variant, or when its existing trainer path is called directly. Stacks and the ultimately selected candidate can change as a consequence. The other **84 entries**, including token heads and standalone LR/SVM/neural heads, retain their protocols.

The train/holdout split, limma-voom method, DE marker rules, explicitly transductive limma batch correction, representations, neural architectures, resamplers, severity thresholds, search-space definitions and search budgets are unchanged by Stage 3. No CLS support is introduced.

## Numerical behavior

- For binary training counts `n0=8, n1=24`, the previous ratio was `3`; the correct positive-class ratio is `1/3`. Both class-majority directions and balanced binary labels are tested. Ratios are recomputed on each training fold, so small fold-count differences can also change results when class 0 is the majority.
- Multiclass sample weights use the same formula, `n_training / (n_classes * n_training_in_class)`. Inner fitting and importance ranking now actually use it, and outer/inner fits derive counts locally. Tuned parameters, selected features, predictions and CV scores can therefore change. With identical full-training labels and fixed parameters, the balanced-weight formula at the final fit is unchanged.
- RF final calibration uses the same procedure as before on feasible data with fixed parameters. Outer and stacking-base probabilities now reflect that procedure. Synthetic tests compare every outer-fold probability/prediction against a manual invocation of the final-fitting helper for both calibration methods and both ensemble settings.
- The corrected binary weighted estimator matches a manually configured reference XGBoost model. A targeted test confirms that its probabilities differ from the historical incorrect-ratio model on a small synthetic dataset.
- Candidate rankings, chosen stack bases, stack predictions and holdout/external predictions from newly trained models may change through corrected fitting and selection. No claim is made about improving a particular dataset's score. Holdout labels do not determine class weights.
- Infeasible CV yields an explicit failure rather than a resubstitution result; infeasible requested calibration yields an explicit failure rather than an uncalibrated artifact.

## Saved-model and API compatibility

The existing `XGBNoEarlyStop` class, its import path and its fit/prediction behavior are retained. Historical weighted XGBoost artifacts saved using that class keep their historical fitted booster and weighting settings when loaded. Loading/prediction does not run the new fit-local policy. Existing RF, calibrated, preprocessing, native and stacked model class layouts are retained; configuration and foundation manifest formats are unchanged.

Checks cover historical-class XGBoost loading with an intentionally historical weight, bitwise-identical predictions after reload, new weighted-estimator cloning/search and joblib round trips, calibrated RF round trips, existing native wrapper compatibility, and all 104 mocked-encoder API train/save/load/external-prediction cases. Existing saved manuscript results were not opened or rewritten. These checks establish compatibility of the supported class paths in the available dependency environment; they do not promise pickle compatibility across arbitrary library versions.

New weighted XGBoost artifacts reference `FitLocalWeightedXGB` and require this Stage 3 code when loading. Backward loading of historical models is preserved; running new artifacts on older framework code is not supported. Existing artifacts are not automatically corrected: the fitting changes apply when retraining.

The public `api.py` signatures and configuration catalogue are unchanged. Additional arguments are confined to existing internal CV/calibration helpers. The previous candidate failure policy is retained: legacy native training records failed candidates and can retain completed candidates; the foundation native adapter still rejects an incomplete applicable suite. Failed candidates cannot supply stack prerequisites or completed-model selection.

## Tests and validation

The 15-module public-framework regression run passed **326 tests, with 1 optional native-sva test skipped**, including **34 new Stage 3 checks** and the 292-test Stage 2 baseline. Log: `.step3-validation/stage3/regression.log`. Existing pandas, SciPy, sklearn feature-name and skopt/numpy warnings remain visible.

The final focused Stage 3 run also passed **34 tests** after the final error-message review. Log: `.step3-validation/stage3/final-targeted.log`. `git diff --check` passed; the configuration schema, catalogue, batch-correction/DE implementations and public API have no Stage 3 changes.

Stage 3 tests use small real XGBoost/RF models, Bayesian-search clones and synthetic labels. They verify binary ratios, multiclass fit arguments at the actual XGBoost fit boundary, inner/outer/importance/final weighting, supplied-weight row alignment through nested search/ranking/calibration, representation/pipeline wrapping, RF CV versus final procedure, all six RF variants, weighted and calibrated stacking bases, infeasible CV/calibration and candidate failure diagnostics, historical-class loading and prediction round trips. Held-out rows are excluded from the observed weighted fitting calls.

The broader suite includes foundation API coverage for all 104 configurations with mocked encoders, split/DE scope checks, preprocessing, saved-model prediction, representations, extraction contracts and existing calibration compatibility. No manuscript experiments, real encoder/checkpoint inference or environment repairs were run. The optional native-sva check requires `OMICLASS_TEST_SVA_LIB`; the Stage 2 native R/rpy2 binary mismatch remains outside this stage's validation.

## Remaining limitations and isolation

Stage 3 does **not** resolve or certify nested stacking evaluation: meta-CV still reuses globally generated base OOF features, and CV/final passthrough handling still needs the approved Stage 4 review. This stage corrects the existing bases' weighting/calibration and removes infeasible-CV resubstitution only. Holdout threshold selection and foundation preprocessing/token positional issues remain for subsequent approved stages. CellFM token architecture is unchanged.

Implementation, test outputs, temporary directories and caches stay in `/data01/home/ls/katzm/omiclass-public`, with validation artifacts under `.step3-validation/stage3/`. Tests reuse the existing interpreter read-only, disable bytecode/CUDA and limit numerical-library threads. No manuscript source, running worker, shared dependency/environment, checkpoint or saved result was modified. The initial Stage 3 session created no commit; the reviewed changes are committed only at the user's subsequent explicit request. No push is authorized.
