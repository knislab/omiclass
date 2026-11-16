# Pairwise marker selection

Pairwise selection is opt-in: `marker_selection.method: pairs`. The default
`de` path and its frozen configuration format remain unchanged.

## Algorithm

`omiclass.features.pairs.OvRTSPSelector` implements the development version's
deterministic DE-guided one-versus-rest top-scoring-pairs algorithm:

1. Within each fit, require at least two samples per class. Keep nonconstant
   genes expressed above `min_expression` in at least `min_fraction` of samples
   in any training class (so rare-class markers can survive).
2. For each class versus pooled rest, rank genes by two-sided Welch p-value,
   descending absolute mean difference, then gene identifier. Take up to `g`
   positive-effect and `g` negative-effect candidates. Binary tasks use only
   the first sorted class comparison.
3. Score each up/down pair `(A, B)` by `P(A > B | target)` minus the mean of
   `P(A > B | other class)` over other classes, weighting those classes equally.
   Rank by descending score then A/B identifiers. Greedily keep up to `k`
   positive-score pairs per class, respecting `max_gene_reuse` per class.
4. Union the selections, merging duplicate/reversed pairs with the first
   orientation retained. Transform each sample into signed int8 features:
   `+1` for A > B, `-1` for A < B, and `0` for an exact tie. No expression
   features pass through to the classifier.

The selector is the unchanged development implementation (archived revision
`2ef206af43fe0673f65d0708d8075568eb23827c`). Welch DE, rather than limma-voom,
is already part of that reference. Selection is refitted in inner/outer CV,
calibration, and final training. Candidate ranking uses pooled-rest samples;
pair scores balance the other classes. There is no FDR cutoff or randomness.

## Enable through the standard API or CLI

```yaml
marker_selection:
  method: pairs
  g: 50
  k: 25
  max_gene_reuse: 2
  min_expression: 0.0
  min_fraction: 0.1
  search_space: null
batch_correction:
  algorithm: none
  input_scale: tpm  # use log1p_tpm for already logged expression
runtime:
  feature_selection:
    enabled: false
  feature_reduction:
    enabled: false
```

```bash
python api.py \
  --override expression_file /path/to/normalized_expression.csv \
  --override meta_path /path/to/metadata.csv \
  --override classification_column target \
  --override output_dir /path/to/new_run \
  --override marker_method pairs \
  --override batch_correction.algorithm none \
  --override batch_correction.input_scale tpm \
  --override feature_selection_enabled false \
  --override feature_reduction_enabled false
```

Input is genes by samples, with unique gene/sample identifiers, finite values,
and normalized RNA expression. For TPM, apply log1p only when the existing
`runtime.log_transform` is true; `log1p_tpm` skips it. Scale handling is explicit
and sample-local, including saved-run prediction. Normalize counts independently
per sample first. Raw-count routing, batch correction, embeddings, upstream
preselected gene sets/surface filters, and legacy importance reduction are
rejected. The stacker is excluded because its expression passthrough changes the
feature representation. Existing RF, boosting and imbalance variants are reused;
SMOTE interpolates signed pair features and needs separate biological assessment.

Aliases are `marker_method`, `pair_g`, `pair_k`, `pair_max_gene_reuse`. Other
settings use dotted keys. Optional positive integer lists under
`marker_selection.search_space` tune `g`, `k`, or `max_gene_reuse` within the
existing Bayesian budget. Classifier parameters route through `classifier__`
(and sampled RFs retain `rf__`); pair parameters route through `pairs__`.

## Artifacts and inference

Saved joblib models retain the complete selector. `MODEL.pairs_MEMBER.csv`
exports class scores/orientations for each fitted calibration member. Fold
reports contain `cv.per_fold[].pair_selection` and selected pair counts.
External prediction uses fixed pairs, requires every selected gene, tolerates
missing unused training genes and extra genes, and never fills missing genes
or refits selection. Prediction needs no external outcome labels.

```bash
python api.py --override expression_file /path/to/external_expression.csv \
  --override output_dir /path/to/completed_run
```

Direct saved-model use expects samples by genes on the fitted expression scale:

```python
import joblib
from omiclass.features.pairs import prepare_pair_prediction
model = joblib.load('/path/to/model.joblib')
X = prepare_pair_prediction(model, external_samples_by_genes)
predictions = model.predict(X)
```

## Adaptation and validation

The port starts at corrected public commit
`07752936aa02ebbfffa889520667c702edcb02c2`. Public split resolution, model
selection, probability alignment, threshold persistence and saved-run provenance
remain in place. The pair pipeline itself owns fold-fitted preprocessing;
the public generic preprocessor is bypassed for pairs so its automatic logging,
constant filtering, finite-value filling and full-input-schema requirements do
not change the reference behavior. Pair inference integrates at the public
shared `eval.prediction.match_features` boundary. The old development comparison
launchers and saved-report finalization changes are outside this port.

Focused tests include exact candidates, selected pairs, scores, orientations,
signed features and RF predictions versus the retained development Git snapshot
on seeded binary, multiclass and imbalanced examples; inner/outer/calibration
fit scopes; serialization; RF/SMOTE/weighted boosting; the real training and
prediction-only pipeline; and exact default-path regression against the public
starting commit. Historical Git comparisons skip in source exports that lack
the archived revisions. No large cohort performance claim is made.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -p no:cacheprovider tests/test_pairs*.py -q
```
