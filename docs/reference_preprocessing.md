# Frozen expression preprocessing

`batch_correction.algorithm` selects `limma` (default), `none`,
`frozen_quantile`, `fsva_exact`, or `reference_ruv`. The existing `batch_correction.method`
list still means limma's `batch` / `design` / `covariates` / `pcs` flags.
Old configurations retain their execution behavior. The limma runner and R
correction code are unchanged.

The new paths require RNA-seq **TPM or natural-log `log1p(TPM)`** matrices,
genes in rows, sample IDs in columns. Choose `input_scale` explicitly; there is
no inference from the incoming cohort. They output corrected log-scale values,
which may be negative for factor correction. No clipping, exponentiation, or
additional log transform is applied before classification.

Omiclass's raw-count normalizer computes length-adjusted TPM followed by natural
`log1p`. Convert counts with that protocol first, using the same measured gene
universe and gene lengths for training and application. Direct raw-count and
embedding-backend execution are rejected for these new paths. Do not supply
log2 or log10 expression as `log1p_tpm`. TPM's compositional denominator makes
changes in measured gene coverage consequential even when model genes overlap.
The comparison preparer derives its retained gene set from positive CLL-Map
training counts only, then requires external coverage of that fixed set. It
does not adopt the historical merged file's external intersection.

## Existing architecture and integration

`LoadInputStep` loads expression; `DEStep` can select genes using training counts;
`RawCountNormalizationStep` produces log1p(TPM); `BatchCorrectionStep` precedes
`TrainAndEvalStep`. The latter performs the existing split, feature-selection,
optimization, CV, final fitting and evaluation. Historically limma estimates
correction on the supplied full matrix before that split. `RebatchCorrectionStep`
reruns limma on prediction input and retains incoming samples. There is no saved
limma fit for unseen-cohort application.

For the frozen alternatives, supply a one-ID-per-line `training_samples_path`.
Only those columns enter reference fitting **and** the downstream classification
input. Additional columns in a combined input are excluded from training.
The fitted state and corrected training expression are saved in
`preprocessing/reference.npz` and `preprocessing/training_expression.csv`;
`reference.json` records the sample IDs, schema, scale, and diagnostics. The
numeric artifact is a versioned NPZ with JSON metadata, loaded without pickle.
Quantile/RUV artifacts save parameters; exact fSVA additionally retains its fixed
training database and corrected training values. Existing reference files are
not replaced by fitting.

The explicit reference population is the preprocessing training pool, before
Omiclass's internal train/holdout split. Thus internal holdout samples influence
the reference when listed, and `reference_ruv` also uses their training-pool
biological design. This follows the requested preprocessing-before-classification
architecture; internal validation is **not** a nested estimate of preprocessing.
DKFZ must never appear in this list. Use external performance to assess transfer.

The existing `clean_and_predict` path automatically loads
`preprocessing/reference.npz` from the new model's run directory, or an explicit
`reference_path`. It transforms external-only input or removes reference IDs
from combined input. Correction does not open metadata, inspect labels, estimate
new parameters, or replace the saved training expression/reference. Output goes
to `preprocessing_external/`. Existing survival/prediction reporting then runs
as before. Native prediction reports may read labels for reporting after
correction; the correction itself has no label argument.

## Frozen quantile algorithm

For log-scale training matrix `Y` with G genes and N samples, sort each column,
then take the arithmetic mean at each rank:

`target[r] = mean_j sort(Y[:, j])[r]`.

For each incoming sample, sort its G values and map their ranks onto this fixed
target. For ties, use the average rank and linearly interpolate adjacent target
entries for half-integer ranks. This is the complete-data, equal-length target
algorithm used by `preprocessCore::normalize.quantiles.determine.target` and
`normalize.quantiles.use.target`. The Python implementation avoids introducing
an R runtime dependency for this transformation. It is ordinary frozen-target
quantile normalization, **not** the more elaborate fRMA method.

It assumes comparable marginal expression distributions and fixed coverage.
Ties (especially zeros) can leave distributions differing after normalization.
Large real shifts in overall biology can be distorted. This is distribution
normalization and does not estimate gene-specific batch effects. Fitting is
label-free. No cohort identity is needed at application.

Reference: [Bolstad et al. (2003), Bioinformatics](https://doi.org/10.1093/bioinformatics/19.2.185);
[preprocessCore target API and missingness assumptions](https://bioc.r-universe.dev/preprocessCore/doc/manual.html),
[implementation and tie handling](https://github.com/bmbolstad/preprocessCore/blob/master/src/qnorm.c).

## RUV-inspired frozen control-gene factor correction

This is a **custom reference factor model**, not RUVSeq, RUVg, or fSVA.
The reviewed [RUVg implementation](https://github.com/drisso/RUVSeq/blob/master/R/RUVg-methods.R)
centers and decomposes the supplied matrix and returns factors/corrected counts;
it does not expose a saved external-sample projection. Established
[fSVA](https://doi.org/10.7717/peerj.561) supports prediction, but the reviewed
[sva implementation](https://github.com/bioc/sva/blob/master/R/fsva.R) recomputes
an augmented projection in exact mode; its fast mode scales across new samples.
Neither reviewed API directly provides the particular frozen control-gene,
sample-independent map implemented here. This is a scoped assessment of these
APIs, not a claim that no other transferable method exists.

With samples in rows, let `Y` be log1p(TPM), `C` its user-specified control genes,
and `B` the biological design (intercept, target, other protected covariates).

1. Store training control means `mu`, and take the first k right singular
   vectors `V` of `C - mu`. Controls are centered, not variance standardized.
2. Store `P = V / sd((C - mu)V)` and compute `W = (C - mu)P`.
3. Jointly fit `Y = B beta + W alpha + error` by ordinary least squares.
   Store `alpha`, `P`, `mu`, controls, gene order, design encoding and diagnostics.
4. For any sample `y`, compute `w = (y_controls - mu)P`, then
   `y_corrected = y - w alpha`. Training uses precisely this same transform.

No external centering, factor scaling, regression, covariates or labels occur.
The training mean technical reference remains; no per-batch centering is applied.
Numerically rank-deficient controls, underdetermined joint designs and joint
design condition numbers above 1e8 cause informative errors. The scale-normalized
condition number and factor variance explained by the biological design are
saved for confounding review. High factor/biology R2 is a warning sign even
before numerical nonidentifiability is reached.

The design always includes `project.cluster_column`; `design_columns` adds
protected covariates. Numeric columns are standardized training covariates,
categorical columns use dummy variables. Encode multi-level biological categories
as strings rather than numeric codes. Do **not** protect cohort if it is the
nuisance you intend to remove. Constant/redundant design variables are rejected.
Joint regression avoids attributing all training target-associated expression to
factors merely because they correlate. It does not guarantee preservation of
marginal class means under cohort/class confounding, nor protect biology absent
from the specified design.

Controls must be measured, non-differential for the target/relevant biology,
and responsive to relevant technical effects. Uniform spike-ins with valid
experimental handling or independently validated disease-specific controls can
be appropriate. Housekeeping status, low variance, or a nonsignificant test
alone do not establish control validity. Supply a control list and a substantive
`control_justification` (source, assay, target relevance); no automatic control
selection is supplied. The software cannot verify the biological truth of that
justification. No CLL/IGHV control list is asserted or bundled.

The SVD/control rationale follows [Risso et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4404308/).
The [RUVSeq vignette](https://bioconductor.org/packages/release/bioc/vignettes/RUVSeq/inst/doc/RUVSeq.html)
also cautions that subtracting unwanted variation can remove biology. This
implementation is a continuous-expression regression for prediction, not an
RNA-seq count GLM or a differential-expression procedure.

Transfer requires that external nuisance effects lie in the learned control
subspace and retain the training relationships between controls and other genes.
A completely new cohort name is supported; completely new technical mechanisms
or changed loadings are not identifiable from a frozen reference. Perfect
biology/technical confounding and invalid controls cannot be repaired by this
implementation. Choose controls and k without DKFZ performance selection.

## Native exact fSVA

`fsva_exact` is the established `sva::sva` + `sva::fsva(method="exact")`
implementation, using the R package through an isolated Rscript subprocess.
It requires no predefined control genes. SVA uses only the declared training
pool and its protected biological design, with a fixed `n_factors` (default 1)
and `reference_random_state` (default 11). Training-constant genes are excluded
from SVA estimation and passed through unchanged. The package version, R version,
joint-design conditioning, serialized R fit, training database, and corrected
training values are saved in the NPZ reference. Application requires the same
sva version; configure the compatible R executable and library paths.

For each incoming patient, native exact fSVA appends that patient's expression
to the fixed database to estimate its surrogate scores by weighted SVD. It uses
the training-estimated weights and gene coefficients, and leaves the saved
database correction and classifier unchanged. It supports unseen cohort names
and does not use incoming labels. **The projection is recomputed per patient**;
this is an established database-based transfer procedure, not a fixed linear
projection like quantile/RUV. Different incoming patients do not share an SVD.

Training IDs are checked against stored input hashes and return the original
native database correction. Reusing a training ID with changed expression fails.
This distinction preserves native fSVA's database/new-patient definitions.
For sva 3.50, the wrapper supplies one training column as `newdat` during fitting
and retains only `$db`, avoiding that version's `newdat=NULL` error.

Native `method="fast"` is deliberately excluded: sva 3.50's incoming-score
scaling depends on the incoming cohort and is nonfinite for a singleton. An
executed synthetic audit gave fast-mode cohort-dependence difference 5.892819;
exact-mode cohort-dependence and singleton differences were both zero. This
choice was explicitly agreed for the comparison. Reference:
[Parker, Corrada Bravo and Leek (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4179553/),
[native source](https://github.com/bioc/sva/blob/RELEASE_3_18/R/fsva.R).

fSVA assumes transferable nuisance gene effects and enough training separation
between biology and technical factors. It cannot resolve complete confounding
or guarantee that every biological dimension is preserved. Exact application is
computationally expensive. Evaluate factor counts without selecting on DKFZ.

## Configuration and API

```yaml
batch_correction:
  algorithm: frozen_quantile
  input_scale: tpm
  training_samples_path: /path/to/cllmap_reference_ids.txt
```

For factor correction:

```yaml
batch_correction:
  algorithm: reference_ruv
  input_scale: tpm
  training_samples_path: /path/to/cllmap_reference_ids.txt
  control_genes_path: /path/to/validated_controls.txt
  control_justification: "Describe independent validation for CLL/IGHV and assay here"
  n_factors: 1
  design_columns: [age, sex]  # target is automatically included
```

For native exact fSVA:

```yaml
batch_correction:
  algorithm: fsva_exact
  input_scale: tpm
  training_samples_path: /path/to/cllmap_reference_ids.txt
  n_factors: 1
  reference_random_state: 11
  design_columns: [age, sex]
  rscript_path: /path/to/compatible/environment/bin/Rscript
  r_library_path: /path/to/isolated/R/library
```

All identifiers must be unique and nonmissing. Reordering genes is allowed;
missing **or extra** genes produce an error. No intersection chosen with DKFZ,
implicit zero fill, missing-value imputation or label-based external adaptation
is performed. Supply the frozen complete gene schema explicitly.

```python
from omiclass.batch.reference import ExpressionReference, FrozenQuantile, ReferenceRUV
from omiclass.batch.fsva import FrozenSVA

ref = FrozenQuantile(input_scale="tpm")
corrected_training = ref.fit_transform(training_tpm)
ref.save("reference.npz")
corrected_external = ExpressionReference.load("reference.npz").transform(external_tpm)

ref = ReferenceRUV(control_genes=controls, control_justification=rationale,
                   n_factors=1, design_columns=["ighv_mut", "age", "sex"],
                   input_scale="tpm")
corrected_training = ref.fit_transform(training_tpm, training_metadata)

ref = FrozenSVA(["ighv_mut", "age", "sex"], n_factors=1, input_scale="tpm")
corrected_training = ref.fit_transform(training_tpm, training_metadata)
```

Using the existing CLI (new output directory for training):

```bash
python api.py --override expression_file /path/to/cllmap_tpm.csv \
  --override output_dir /path/to/new_projection_runs \
  --override meta_path /path/to/cllmap_metadata.csv --override class_col ighv_mut \
  --override batch_correction.algorithm frozen_quantile \
  --override batch_correction.input_scale tpm \
  --override batch_correction.training_samples_path /path/to/cllmap_reference_ids.txt

# Use the resulting run directory (contains leaderboard.csv and frozen config).
python api.py --override expression_file /path/to/dkfz_tpm.csv \
  --override output_dir /path/to/new_projection_runs/Omiclass-TIMESTAMP
```

The frozen convenience route accepts normalized expression and disables count-DE
execution; it retains the existing downstream classifier feature selection.
Supply `project.geneset_path` to reuse a training-only DE gene list. The comparison
runner does this explicitly for each saved manuscript split.

See [comparison commands](../experiments/reference_preprocessing/README.md).
