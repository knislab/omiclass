# Step 3 Stage 2: split consistency and preprocessing scope

Stage 2 is implemented in `/data01/home/ls/katzm/omiclass-public`, starting from Stage 1 commit `11791e0` on `public-release`. This report accompanies the reviewed Stage 2 commit, `Fix split consistency and fold-local feature selection`. Stage 3 implementation follows only after the user's subsequent explicit approval.

## Split and data flow

- A new internal split step resolves the labelled population, dense class mapping and one train/holdout split before DE or learned preprocessing. DE reporting, classifier fitting and evaluation consume the same recorded IDs. Samples excluded for missing labels remain available for the existing external/survival path.
- Existing `split_json` and `reuse_split` execution options now reach their intended runtime consumer. Explicit split/holdout files are checked for missing samples, empty/overlapping/duplicate partitions and training-class coverage. Invalid requests fail rather than silently choosing a replacement split. Post-correction population changes also fail validation.
- Classes present only in unrelated metadata rows do not create gaps in classifier labels; fractional numeric labels are mapped without truncation. CV fold counts are resolved from training labels.
- Existing frozen-reference training IDs define or must match the classifier training partition. References fit independently within estimator fitting scopes; labelled holdout samples are transformed with fitted state rather than included in reference fitting.
- Explicit transductive limma remains cohort-wide. Split resolution does not restrict its supplied expression matrix to training samples or replace correction with another method. Its resulting classifier scores remain transductive, not inductive performance estimates.

## DE fitting and supervised selection

- The existing limma-voom R script and marker-selection implementation are unchanged. One-vs-rest/close-pair contrasts, ranking/budget rules, thresholds and classifier search-space definitions remain the same.
- DE receives raw counts by sample ID for RNA and the existing training-fitted CPTAC preprocessing for protein inputs. Metadata is narrowed to fitting IDs; outcome values are constructed from that fit call's labels. Existing numeric/category covariate encoding is fitted locally, so validation-only levels cannot change fitting codes. Covariate-driven sample removal is rejected instead of silently changing the partition.
- DE reporting uses only the overall training partition. It no longer passes its selected gene list into CV. Report rendering is a separate method; fit-only calls skip rendering and use disposable output directories.
- A cloneable internal estimator wrapper fits existing preprocessing/DE within each outer fit, inner Bayesian fit, calibration fit and final fit. It delegates prediction to the existing classifier; no classifier or neural architecture was added.
- Existing importance-based feature selection now fits its ranking within each scoring fit's training data. The preliminary Bayesian ranking search, mean-fold importance rule, configured subset fractions and feature-count bounds are retained. Its preliminary search also fits DE/preprocessing within its own training folds. This increases fitting work.
- Final feature selection uses the same training-only recipe as CV. Cross-fold consensus remains a reporting artifact instead of a globally imposed final input restriction.
- Corrected the DE count-filter argument being overwritten by `runtime.min_features`; explicit `de.min_count`/runner arguments now reach R.

## Fitted preprocessing and prediction

- Native foundation logging decisions no longer inspect holdout embeddings. Logging and constant-column filtering fit inside each estimator clone; frozen encoder extraction itself remains unchanged.
- CPTAC missingness filters, log2 offsets and medians fit on training samples and are reused for prediction. For transductive correction, its preparatory CPTAC transform fits on the declared training cohort and is persisted for external re-correction.
- Core reference transforms fit inside CV/final estimators while retaining the existing reference algorithm and numeric/categorical biological-design semantics.
- Saved new estimators contain their fitted transforms and selected columns. Raw expression and fitting metadata are omitted from serialized preprocessing objects. Prediction does not call DE, tune or refit these components.
- New core runs record preprocessing ownership, train/test IDs and input mode in `preprocessing/scoped.json`. Prediction recovers that input mode and avoids double transformation. New corrected runs use recorded split IDs when excluding known samples during re-correction, including runs with a separate split-output directory.
- The existing stack's passthrough values reuse a fitted training preprocessor so raw NaNs/untransformed values are not appended to probabilities. This is preprocessing plumbing only; the stack's nested-CV correction remains Stage 4 work.
- Out-of-domain prediction values are rejected when incompatible with the saved log1p transform or CPTAC log2 offset.

## Compatibility and numerical behavior

| Area | Behavior and expected numerical differences |
|---|---|
| Catalogue and standalone foundation heads | All 104 configurations remain. Encoder extraction, pooling, token handling, standalone LR/SVM/MLP training protocols and neural architectures are unchanged. |
| Native foundation suite | The 20 configurations using the native suite now learn preprocessing within fitting partitions. Results can change when a holdout previously changed logging or a validation partition influenced constant-column decisions. |
| DE/classifier splits | Identical valid supplied partitions retain their IDs/order. Previously ignored or incompatible split requests now fail or use the requested partition. |
| Supervised selection | Fold-specific markers/rankings, CV scores, chosen hyperparameters and final models can change. No claim is made that corrected CV scores must improve or decrease on every dataset. |
| DE covariates | The same category-coding rule now learns codes from the fitting partition. Validation-only covariate levels cannot shift training codes; affected fits may change numerically. |
| DE count filtering | The default DE cutoff is 10 rather than the incorrectly substituted runtime feature minimum, normally 20. This can change filtered genes and selected markers. Explicit caller cutoffs remain respected. |
| CPTAC | On the same fitting population, synthetic tests reproduce the previous transform mathematics. Held-out/external values now reuse training medians/offsets instead of changing them. |
| Limma batch correction | Cohort-wide correction is preserved. Predictions remain dependent on the supplied cohort under the intended transductive protocol. No transferable foundation correction was added. |
| Frozen references | Holdout samples and CV validation samples no longer enter the corresponding reference fit. Corrected features and scores may change. |
| Historical artifacts | Existing foundation format-1/2 and native outer wrappers remain readable. Old core models without the new preprocessing marker retain their legacy prediction path. No existing saved artifact is rewritten or migrated. |

Configured search ranges and search-space definitions are unchanged. Automatic ranges still use the existing data-dependent functions; their effective inputs can differ when fitting scope/available dimensions change. Feature-importance tuning is now repeated within permitted partitions, so runtime increases and selected recipes can differ even with the same seed. Historical manuscript scores are not reproduced or updated.

Synthetic numerical evidence includes bitwise-identical fitted-model probabilities after changing only holdout labels and after joblib reload, identical CPTAC training-transform values, identical marker selection with reporting enabled/disabled on fixed bridge results, and preserved historical native logging semantics.

## Validation and limits

The combined 14-module regression run passed **292 tests, with 1 optional native-sva test skipped**. This includes 29 new Stage 2 checks, the prior 263-test passing baseline, and the updated existing native-preprocessing assertion. Log: `.step3-validation/stage2/final-validation.log`; artifacts: `.step3-validation/stage2/final-validation/`.

New checks cover explicit split failures/reuse, unrelated-metadata classes and fractional labels, nested DE label/raw-count lineage, nested importance selection including preliminary tuning, calibration scopes, real small RF fitting with a DE adapter, full core API split/evaluation integration, frozen-reference scopes and numeric outcome design, CPTAC parity, input-mode replay, stack passthrough preprocessing, old native wrappers, serialization without fitting data, and full-cohort transductive limma dispatch.

The DE test bridge substitutes R results while executing the public fitting/marker-selection code; it verifies min-count forwarding and identical selection with reporting on/off. Direct native R/rpy2 execution could not be validated because the supplied environment reports a `methods.so`/`R_typeToChar` binary mismatch. No environment repair or installation was attempted. The R import is now lazy and produces an actionable error when DE is actually requested. R script and close-class-selection source comparisons confirm unchanged statistical code, but do not constitute real-R numerical replay.

All 104 foundation entries pass mocked-encoder train/save/load/external prediction checks. Small real classifiers and shortened synthetic neural fits are covered; no real encoder/checkpoint inference or manuscript experiments ran. Existing pandas/SciPy, sklearn feature-name and skopt/numpy search-result warnings remain visible.

## Remaining stages and isolation

Stage 2 does not certify the existing stacked ensemble's performance estimates: reuse of globally generated base OOF features can still create indirect leakage into meta-CV. Correct nested stack evaluation remains Stage 4. Weighting/calibration consistency, threshold/holdout reporting and CellFM/token issues remain for Stages 3–6. No new architecture, correction method or optional classifier feature was introduced.

Implementation and validation writes are confined to the public worktree, including temporary files/caches under `.step3-validation/stage2/`. Tests reused the existing interpreter read-only, disabled bytecode/CUDA, and limited numerical-library threads. No manuscript checkout, worker, shared environment, checkpoint or existing result was modified. The initial Stage 2 session created no commit; the reviewed changes are committed only at the user's subsequent explicit request. No push is authorized.
