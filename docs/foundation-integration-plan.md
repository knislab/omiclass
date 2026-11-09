# Foundation-model API integration plan (Step 2)

## Scope and evidence

All implementation changes stay in the public-release worktree. The manuscript checkout is read-only. No experiment, checkpoint, environment, result, or shared Git reference is changed. Tests use a separate virtual environment, CPU-only mocked encoders and temporary outputs; no checkpoint downloads or manuscript runs.

Evidence: the saved `experiments/non_cancer_als/foundation_models/config/cancer_protocols.json`, `_shared/catalogue.py`, `_shared/classification.py`, `_shared/extraction.py`, `_shared/preprocessing.py`; the saved foundation statistics configuration audit; `article_input_correction` and `article_foundation_heads`; the standalone BulkRNABert models; and the separately validated `bulkrnabert_full_tokens` implementation. These are inspected in the manuscript checkout, never imported from it in the public implementation. The latter has validation artifacts, not completed full-scale benchmark fits; expose it as an experimental protocol, not a recovered paper score.

## Configuration matrix

- BulkFormer: gene mean and the historical max + mean + median **sum**, each with full/supervised/PCA/supervised-PCA pooled vectors and Omiclass, nested-CV LR, fixed SVM, fixed-epoch MLP, or MLP+SMOTE. Full tokens use the fixed-epoch token MLP only. Existing max-only/median-only extraction remains a framework extension, not a claim of matched manuscript evidence.
- CellFM: original single-view and deterministic multiview (including the saved 20-view setting); valid-gene means averaged over views, the four pooled representations and five head protocols above. Full tokens use per-view token MLP with probability averaging.
- BulkRNABert: original frozen layer-4 gene mean and four pooled representations; Omiclass, nested-CV LR, fixed SVM, and the original SELU/inner-validation MLP with optional inner-training-only SMOTE. Full tokens use the separately validated article token head, not the pooled BERT MLP. No alternative layers/checkpoints or token SVM/SMOTE.
- Correction: opt-in historical limma on model-specific continuous inputs before extraction, including pre-split cohort fitting and full-input external re-correction. Preserve CellFM original count-selected views and substitute corrected scalars; preserve BERT proven-zero gene restoration and recorded zero floor. No train-only correction claim. Other correction methods are outside this integration.
- Omiclass candidates and imbalance dispatch reuse Trainer unchanged: plain/weighted RF, SMOTE/ADASYN/SVM-SMOTE/borderline-SMOTE RF, plain/weighted XGBoost and LightGBM, existing stacker, as applicable to the original severity policy.

## Reusable components and API gaps

Reuse existing backend resource checks, count/TPM preparation, frozen encoder calls, CellFM view driver, embedding selector/PCA, Trainer, holdout metrics, split utilities and result type. Promote the existing article head and BERT MLP loops into importable public modules with source attribution, without importing manuscript paths or duplicating those loops. Add instance-local extraction observers to expose tokens while leaving the default pooled extractor behavior intact. Keep JAX in its configured subprocess environment.

Add an opt-in `foundation` configuration block and CLI aliases. Default disabled preserves marker and legacy embedding routes. Route the new mode before constructing the legacy orchestrator; this avoids forcing heads with incompatible evaluation protocols through Trainer. Separate training/prediction actions, saved configuration, immutable input/extraction identity, explicit split, fitted transforms, canonical class order, head artifacts and prediction outputs. Predict reloads fitted components and never refits representations, scaling, SMOTE, or heads. Reject incompatible combinations before loading encoders or creating outputs.

## Logical implementation stages

1. Add schema, compatibility catalogue and API dispatch. Validate backend exclusivity, modality, representation/pooling/head/view combinations, correction variables and prediction requirements. Add parsing/dispatch/legacy-route tests.
2. Add extraction adapters and model-specific continuous preprocessing. Reuse backend forwards and pooling, capture token records, use isolated BERT worker and per-run cache/runtime paths. Add mocked-forward, view/mask, normalization-once, correction ordering and subprocess tests.
3. Add protocol adapters: nested LR; fixed SVM; promoted fixed-epoch article vector/token head; promoted inner-validation BERT head; existing Trainer suite. Keep selection/scaling inside each CV training fold, BERT reduction inside inner training, and fixed heads train-only. Save scope/protocol evidence; fixed heads have no invented CV scores. Preserve default algorithms and original imbalance dispatch.
4. Complete saved prediction for all heads, encoder/schema checks, class/probability alignment, strict train/holdout/external separation and output collision protection. Add round-trip, rejection and synthetic API integration tests.
5. Add invocation documentation, supported/unsupported matrix, source/protocol provenance and Step 3 issue list. Run targeted new and applicable existing tests; verify manuscript and existing worktrees unchanged.

## Compatibility, dependencies and tests

Use existing NumPy/pandas/SciPy/sklearn/pydantic/PyYAML/joblib. Optional Torch for BulkFormer/CellFM and MLPs; JAX/Haiku/tokenizer in the configured BERT Python; existing imbalanced-learn/skopt/XGBoost/LightGBM for the original suite; R/limma for correction. No dependency is installed into viz_env. A separate virtual environment can inherit its installed packages read-only; all runtime caches, bytecode, pytest cache and model outputs go to separate temporary locations.

Tests cover supported and rejected combinations; parsing through api.py; lazy backend selection; synthetic per-sample/view token observations; train-fold transform fit IDs; MLP inner validation exclusion; resampling train-only; CV/holdout separation; artifact reload and external prediction equivalence; existing marker route and backend tests. Full encoder/checkpoint and R integration remain unexecuted unless separately provisioned.

## Historical behavior deliberately retained for Step 3

Pre-split/transductive limma and external re-correction; original BERT corrected-input zero floor; BulkFormer all-pooling sum; position-flattening token head and CellFM view positions; legacy Trainer calibration/stacking/selection/imbalance choices; fixed SVM and fixed-epoch MLP without nested CV; BERT early stopping without all-training refit. Do not silently rewrite any of these algorithms. Configuration guards, serialization, API wiring, and new output isolation are integration work. Historical numerical replay is limited to lightweight parity tests; real checkpoint replay is not claimed.
