# Frozen-reference preprocessing comparisons

## BRCA: TCGA -> APOLLO (launch manually)

`run_brca.sh` prepares a new TCGA/APOLLO snapshot, then runs **both frozen
quantile and native exact-fSVA**, each with seeds **11, 61, 73, 80, 89**.
Nothing starts until you invoke it. Preview the commands without preparing data
or fitting anything:

```bash
bash experiments/reference_preprocessing/run_brca.sh --dry-run
```

Start both methods, allowing **two classifier seeds per method** concurrently,
with **five physical CPU cores per seed** (**20 cores total**):

```bash
bash experiments/reference_preprocessing/run_brca.sh \
  --workers 2 --cores-per-seed 5 \
  --out experiments/reference_preprocessing/outputs/brca_tcga_apollo_5seeds
```

The output directory must be new. Omit `--out` to generate a unique timestamped
directory. Preparation finishes first; both methods then preprocess concurrently,
and each queues its five classifier runs as soon as its own preprocessing finishes.
Each method fits one shared TCGA reference and applies it to APOLLO once, reusing
those transformed matrices across its five classifier seeds.

For **all five seeds per method at once**, use `--workers 5 --cores-per-seed 5`
(**50 physical cores total**). For a smaller allocation, use
`--workers 1 --cores-per-seed 5` (10 cores total). CPU sets are chosen from the
current scheduler/process affinity, skipping SMT siblings. Override them with
`--frozen-quantile-cpus 0-9 --fsva-exact-cpus 10-19`; overlapping physical cores
are rejected. `--workers` always means workers **per method**.

Run either method alone using `--method frozen_quantile` or `--method fsva_exact`.
For example, in separate terminals or servers:

```bash
bash experiments/reference_preprocessing/run_brca.sh --method frozen_quantile \
  --out experiments/reference_preprocessing/outputs/brca_quantile_5seeds
bash experiments/reference_preprocessing/run_brca.sh --method fsva_exact \
  --out experiments/reference_preprocessing/outputs/brca_fsva_5seeds
```

When launching separate processes on the same server, explicitly assign disjoint
CPU sets with the corresponding CPU options; automatic allocation is coordinated
within a single launcher only. To keep both methods running after disconnecting:

```bash
screen -dmS brca-reference bash experiments/reference_preprocessing/run_brca.sh \
  --workers 2 --cores-per-seed 5 \
  --out experiments/reference_preprocessing/outputs/brca_tcga_apollo_5seeds
screen -r brca-reference
```

Detach with Ctrl-A then D. Ctrl-C while attached stops the launchers and their
owned experiment processes. Logs and progress are saved under the output directory:

```text
prepare.log, launch_plan.json, exit_codes.json
prepared/                         # hashed immutable inputs
frozen_quantile/, fsva_exact/
  progress.json, cpu_allocation.json
  preprocessing.log, shared/      # one fixed reference per method
  seed11.log, seed61.log, ...
  seed11/, seed61/, seed73/, seed80/, seed89/
```

These runs preserve the saved expression-based BRCA configurations, DE gene lists,
and **625/169 train/internal-holdout splits (794 TCGA patients)** from
`final_results_runs/repets_new/brca/no_correction/Omiclass_BRCA_no_correction_seed*/`.
They evaluate the selected model separately on **68 APOLLO patients**. No historical
results are overwritten. Following the CLL experiment, each preprocessing reference
uses the entire saved TCGA pool before the internal classifier split; fSVA therefore
also uses the TCGA internal-holdout subtype labels in its biological design. This is
the same reference-population protocol as CLL, not a preprocessing-held-out internal
validation. APOLLO expression and outcomes never enter reference fitting, classifier
training, tuning, or model selection.

The default fSVA biological design protects **`classification` (subtype) only**,
with **one surrogate variable** and a fixed preprocessing seed of 11. Several saved
TCGA patients lack age/sex/race/stage, so those covariates are not included by default.
Use `--design-columns age sex ...` to request additional complete covariates; missing
values fail rather than altering the saved population or imputing. This option
changes fSVA's design, not the saved classifier/DE settings.

TPM is computed independently for each sample using TCGA count genes and fixed
**hg38 GENCODE v34 gene lengths** (`omiclass/data/gencode_v34_GRCh38_hg38.csv`).
These lengths cover all 300 saved BRCA DE genes; the CLL hg19 lengths would omit
18 of them. This explicitly supplies log1p(TPM) to the new methods; historical
BRCA input was passed through the expression-file route, so the saved old results
should not be described as a comparison that changes only batch correction.
The TPM denominator schema is fixed from TCGA; APOLLO-only genes
do not alter it. Genes with no positive TCGA counts are then removed. APOLLO must
provide every normalization gene, including observed zeros. APOLLO subtype labels
are copied into the snapshot as an opaque file and opened for scoring only after
predictions are saved. Snapshot hashes cover that file as well.

The script uses the workspace's `viz_env` Python and isolated `outputs/runtime_r_library`
for sva. Override Python with `OMICLASS_PYTHON=/path/to/python`; use `--rscript` and
`--r-library` for another compatible R installation. It never installs packages.
Input paths are configurable through `--source-root`, `--training-counts`,
`--training-metadata`, `--external-counts`, `--external-metadata`, and `--gene-lengths`.
Reuse a prepared BRCA snapshot with `--prepared PATH` and a new `--out` directory.

Collect results after completion:

```bash
python -m experiments.reference_preprocessing.run summarize \
  --runs-root experiments/reference_preprocessing/outputs/brca_tcga_apollo_5seeds \
  --out experiments/reference_preprocessing/outputs/brca_tcga_apollo_5seeds_summary
```

## CLL-Map -> DKFZ

This additive runner uses the expression-based manuscript Omiclass configuration,
its saved train/holdout split and training-only DE preselection for each of seeds
11, 61, 73, 80, 89. Downstream classifier training, feature selection, optimization,
CV, calibration, evaluation, and selection are delegated to `TrainAndEvalStep`.
No historical model/results/index files are updated. All output directories must
be new. `outputs/` is ignored by Git.

The actual saved reference population is **625 CLL-Map patients (500/125 split)**,
not the initially proposed 615. DKFZ contains 122 patients. The user confirmed
preserving the saved 625-patient populations; the runner does not invent exclusions.

Prepare an immutable input snapshot (reads original inputs, copies split/config/DE
genes, converts TPM independently per sample, records source and snapshot hashes):

```bash
python -m experiments.reference_preprocessing.run prepare \
  --input-scale tpm --out experiments/reference_preprocessing/outputs/comparison/prepared
```

Defaults: `final_results_runs/ighv/tpm_merged_with_dkfz.csv`,
`final_results_runs/ighv/meta_merged_with_dkfz.csv`, and
`final_results_runs/repets_new/ighv/corrected/Omiclass_IGHV_corrected_seed*/`.
Training TPM is independently recomputed from `filtered_count.csv` and fixed
`omiclass/data/gencode_GRCh37_hg19.csv` lengths, preserving Omiclass's TPM
denominator. A gene is retained if it has positive counts in at least one of
the 625 reference patients. This produces **55,366 genes**, excluding 1,876
training-zero genes. DKFZ must supply every retained gene. Its extra genes are
discarded by this predefined training schema; missing expressed genes fail.

The legacy merged TPM had already been intersected with DKFZ. Its 47 omitted
CLL genes are all zero in training. Recomputing from counts, with training-only
zero filtering, prevents that historical intersection from selecting reference
features. Training TPM is checked against the retained saved training values.
Five nonidentical duplicate patient columns in source counts are resolved by the
existing loader's first-column rule, with positions and identities recorded in
`source_duplicate_samples.json`. No source file is changed.

Override `--expression`, `--metadata`, `--source-root`, `--training-counts`, and
`--gene-lengths` for an explicitly revised analysis with matching source splits.
External outcome values are excluded from preparation and correction.

Lightweight real-data reference validation, without classifier fitting:

```bash
python -m experiments.reference_preprocessing.run validate \
  --prepared experiments/reference_preprocessing/outputs/comparison/prepared \
  --method frozen_quantile \
  --out experiments/reference_preprocessing/outputs/comparison/validate_quantile

python -m experiments.reference_preprocessing.run validate \
  --prepared experiments/reference_preprocessing/outputs/comparison/prepared \
  --method reference_ruv --controls /path/to/validated_cll_controls.txt \
  --control-justification 'Source and justification for CLL/IGHV controls' --n-factors 1 \
  --out experiments/reference_preprocessing/outputs/comparison/validate_ruv
```

Full comparison commands (use the original Omiclass environment with R/limma,
Pydantic 2, and classifier dependencies). These are substantial training runs:

```bash
prepared=experiments/reference_preprocessing/outputs/comparison/prepared
for seed in 11 61 73 80 89; do
  for method in none limma frozen_quantile; do
    python -m experiments.reference_preprocessing.run run \
      --prepared "$prepared" --method "$method" --seed "$seed" \
      --out "experiments/reference_preprocessing/outputs/comparison/${method}_seed${seed}"
  done
  python -m experiments.reference_preprocessing.run run \
    --prepared "$prepared" --method reference_ruv --seed "$seed" \
    --controls /path/to/validated_cll_controls.txt \
    --control-justification 'Source and justification for CLL/IGHV controls' --n-factors 1 \
    --out "experiments/reference_preprocessing/outputs/comparison/reference_ruv_seed${seed}"
done
```

Each method/seed writes its selected model, `external_predictions.csv`,
`external_metrics.json` (accuracy and canonical macro-F1), `diagnostics.json`,
and a completion record with correction protocol and provenance. Predictions are
saved before external labels are loaded for scoring. Model selection uses the
existing training CV rule, never DKFZ. Source DE preselection is reused so that
preprocessing is the experimental difference; no DE is fitted to transformed data.
Standard downstream Omiclass feature-selection is still fitted normally.

`none` and both frozen methods use no DKFZ information for fitting. Limma fits
the training correction on CLL-Map, then reproduces the existing prediction
procedure by rerunning correction on CLL-Map + DKFZ, retaining DKFZ values. That
external correction is **transductive** and its resulting training values are
discarded. It never replaces the saved training matrix/model. This baseline
requires external cohort/covariate metadata but no external class labels.
If limma's filtering removes required classifier genes, evaluation fails rather
than silently inserting zeros.

Reference fitting uses the declared CLL-Map pool before the internal classification
split. Internal holdout labels participate in the RUV biological design under
that preprocessing convention. Internal CV/holdout therefore do not independently
validate preprocessing; DKFZ is the independent evaluation. See the
[method and limitations](../../docs/reference_preprocessing.md).

Diagnostics include training cohort/target cross-tabulation and mean gene-wise
partial R2 for cohort given target and target given cohort, before/after correction.
`validate` also reports external mean scores in a PCA basis learned only from
corrected training expression. Batch predictability can reflect biology; these
diagnostics cannot establish that biology was preserved. Review them alongside
external accuracy/macro-F1 and control-gene validity.

Focused tests:

```bash
python -m pytest tests/test_expression_reference.py tests/test_reference_integration.py
```

No biologically justified CLL/IGHV negative-control list has been supplied, so
real-data `reference_ruv` validation remains pending. It does not block the
quantile or exact-fSVA comparisons. Synthetic tests cover valid controls,
partial and complete confounding, and known biological/nuisance effects.

## Prioritized quantile / exact-fSVA runs

The agreed established method is `sva::fsva(method="exact")`, because the tested
native fast implementation violates sample independence. Use a compatible R
installation with the sva package. The current workspace has isolated prebuilt
sva 3.50.0 and genefilter 1.84.0 libraries in
`outputs/runtime_r_library/` (package download URLs and hashes are retained there).
They supplement the existing `viz_env` R 4.3.1 environment without modifying it.
These packages were built under R 4.3.3; native integration tests check actual use.

Fit and apply one preprocessing reference for the shared CLL pool. The five
classifier seeds use the same reference because their source population, design,
gene universe, and fixed preprocessing seed are identical. Each classifier
retains its own saved train/holdout split and random state.

```bash
prepared=experiments/reference_preprocessing/outputs/comparison/prepared
r_library=experiments/reference_preprocessing/outputs/runtime_r_library
for method in frozen_quantile fsva_exact; do
  python -u -m experiments.reference_preprocessing.run preprocess \
    --prepared "$prepared" --method "$method" --n-factors 1 \
    --r-library "$r_library" \
    --out "experiments/reference_preprocessing/outputs/comparison/${method}_shared"
  for seed in 11 61 73 80 89; do
    python -u -m experiments.reference_preprocessing.run run \
      --prepared "$prepared" --method "$method" --seed "$seed" --n-factors 1 \
      --r-library "$r_library" \
      --preprocessing-cache "experiments/reference_preprocessing/outputs/comparison/${method}_shared" \
      --out "experiments/reference_preprocessing/outputs/comparison/${method}_seed${seed}"
  done
done
python -m experiments.reference_preprocessing.run summarize \
  --runs-root experiments/reference_preprocessing/outputs/comparison \
  --out experiments/reference_preprocessing/outputs/comparison/summary_v1
```

The cache is hash-checked against its reference and prepared population. This
avoids repeating exact-fSVA's expensive independent external SVDs for each
classifier seed; it does not share classifier fitting or feature selection.
All source files and existing manuscript results remain inputs only.

For native integration tests, additionally run:

```bash
OMICLASS_TEST_SVA_LIB="$r_library" python -m pytest tests/test_fsva_reference.py
```

For a monitored launch of all five seeds (one shared preprocessing fit per
method), run each command with a **new** output directory:

```bash
python -u -m experiments.reference_preprocessing.launch_comparison \
  --prepared "$prepared" --method frozen_quantile --workers 2 --cores-per-seed 5 \
  --out experiments/reference_preprocessing/outputs/comparison/frozen_quantile
python -u -m experiments.reference_preprocessing.launch_comparison \
  --prepared "$prepared" --method fsva_exact --r-library "$r_library" --workers 2 --cores-per-seed 5 \
  --out experiments/reference_preprocessing/outputs/comparison/fsva_exact
```

`progress.json` records preprocessing/classification/completion state and each
seed's exit status; per-seed logs are retained. `--workers` controls concurrent
seed processes and does not alter any per-seed classifier configuration. The
actual strict run uses `outputs/strict_20261102_v2/`. Initial runs using the legacy
merged schema were stopped and marked superseded; they are not comparison results.

## Launch later in screen

**All experiments started during implementation were stopped at the user's
request.** Partial artifacts and logs are preserved; the strict launchers'
`progress.json` files record `stopped_by_user`. There are no completed five-seed
performance comparisons. Nothing resumes automatically.

The prepared 625-patient CLL-Map / 122-patient DKFZ snapshot and isolated sva R
library remain available. From the repository root, these commands run all five
seeds in separate screen sessions **when you choose to execute them later**:

```bash
bash experiments/reference_preprocessing/run_later.sh frozen_quantile --screen omiclass-quantile
bash experiments/reference_preprocessing/run_later.sh fsva_exact --screen omiclass-fsva
```

Each command prints a new timestamped output directory and preserves every
existing run. By default each method runs two seeds concurrently, with five CPU
cores per seed. The remaining three seeds queue automatically and start as a
slot finishes, including when a seed fails. Omit `--screen NAME` for a foreground
run. Model/CV settings are retained; `runtime.n_jobs` is explicitly set to the
requested per-seed core count.

```bash
screen -ls
screen -r omiclass-quantile
screen -r omiclass-fsva
# Detach without stopping: Ctrl-A, then D.
# Stop a run: attach, then Ctrl-C.
# Alternatively stop its named screen session:
screen -S omiclass-quantile -X quit
screen -S omiclass-fsva -X quit
```

The Python launcher handles SIGINT, SIGTERM, and SIGHUP, cancels queued seeds,
and terminates only its owned subprocess groups (including R and fitting
workers). It records a stopped status and leaves artifacts intact. Its scoped
cancellation is tested with dummy sleepers, without running an experiment.

Overrides and help:

```bash
bash experiments/reference_preprocessing/run_later.sh --help
bash experiments/reference_preprocessing/run_later.sh fsva_exact \
  --prepared /path/to/prepared --r-library /path/to/R/library \
  --python /path/to/omiclass/environment/bin/python --out /path/to/new/run
```

The default interpreter is the verified `viz_env` Python in this workspace;
`--python` or `OMICLASS_PYTHON` overrides it. GNU screen is installed here.
To rebuild inputs on another machine, use the earlier `prepare` command and a
compatible sva installation. Interrupted output directories are kept as evidence;
fresh launches use new directories rather than continuing partially fitted runs.

## Concurrent methods with a 20-core budget

The two commands below may be executed one after the other; both screen sessions
then run simultaneously. They explicitly select the defaults:

```bash
cd /data01/private/projects/omiclass
bash experiments/reference_preprocessing/run_later.sh frozen_quantile \
  --screen omiclass-quantile --workers 2 --cores-per-seed 5 --cpus 0-9
bash experiments/reference_preprocessing/run_later.sh fsva_exact \
  --screen omiclass-fsva --workers 2 --cores-per-seed 5 --cpus 10-19
```

| Method | Concurrent seeds | Cores per seed | Maximum method cores |
|---|---:|---:|---:|
| Frozen quantile | 2 | 5 | 10 |
| Exact-fSVA | 2 | 5 | 10 |
| Combined | 4 | 5 | **20** |

CPU allocation uses explicit sets, with no shared allocation lock or service:

| Method | Slot 1 | Slot 2 |
|---|---|---|
| Frozen quantile | CPUs 0–4 | CPUs 5–9 |
| Exact-fSVA | CPUs 10–14 | CPUs 15–19 |

These are also the method defaults when `--cpus` is omitted. CPU IDs 0–19 were
checked to be available and on distinct physical cores on this server. Linux
affinity (the same mechanism as `taskset`) is applied before numerical imports
in every seed/preprocessing process and inherited by its threads, native
libraries and R children. A seed is confined to its five CPUs; the launcher
itself is confined to its method's ten CPUs. The runner rejects CPU IDs outside
the inherited server/scheduler allocation and duplicate physical cores/SMT
siblings. When overriding `--cpus`, choose disjoint sets for the two methods;
the runner validates its own set, not other screen sessions or unrelated jobs.

Within a seed, at most one joblib level runs in parallel, capped at five threads.
Nested searches/estimators use a sequential backend; shared-memory prediction
calls are also capped. BLAS, OpenMP, MKL, BLIS, NumExpr and other native thread
pools are limited to one thread per worker, including an OpenMP hard thread
limit. The exact-fSVA wrapper registers a serial BiocParallel backend. Shared
preprocessing occupies one five-core slot before classification starts and is
currently conservative about native threading; five cores is a ceiling, not a
promise of full utilization at every stage. This small joblib execution context
exists only in the experiment runner. No classifier implementations, CV folds,
search iterations, feature-selection steps, or statistical model parameters
are changed.

Each seed has a distinct output directory, log, plotting cache and joblib temp
directory. A failed seed receives its own failure record and does not stop or
overwrite other seeds. `progress.json` shows `queued`, `running`, completed/failed
states and assigned CPU IDs; `cpu_allocation.json` records the allocation and
thread settings. Cancelling a launcher terminates its owned process groups and
keeps all artifacts. Each newly queued seed reuses the CPU slot released by a
finished or failed seed.

Resource-control tests use tiny joblib probes, fake experiment calls and dummy
sleepers only. They do not train classifiers or run CLL-Map/DKFZ experiments.

## Retry external evaluation after a completed training run

`failed:1` is a subprocess exit code; inspect that seed's log and `failure.json`.
An early runner version checked only `feature_names_in_`, while the existing
`HonestStackingClassifier` can save its feature order as `_feature_names_in`.
The runner now accepts both fitted schemas, retaining strict gene matching.
This failure happens after training/internal evaluation and requires no retraining.

After the current jobs finish, recover that seed's external evaluation into a
**new directory**:

```bash
/data01/home/ls/katzm/.conda/envs/viz_env/bin/python -m experiments.reference_preprocessing.run evaluate \
  --run-dir experiments/reference_preprocessing/outputs/METHOD_RUN/seed61 \
  --out experiments/reference_preprocessing/outputs/METHOD_RUN/seed61_external_retry \
  --cores-per-seed 5 --cpu-list 15-19
```

The command reads the source's saved selection/configuration to recover the
method, seed, prepared snapshot and shared frozen cache. It loads the selected
model and cached external matrix, predicts, then reads outcomes for scoring.
It performs no reference fitting/transformation, tuning or model selection;
prediction uses the classifier's already fitted feature transforms normally.
Source model/config/selection/holdout records are hash-checked and preserved.
It leaves the source failure/progress record intact and writes predictions,
metrics and completion evidence in the new directory. The existing `summarize`
command includes that recovered completion. Running seeds have already loaded
their code; if they hit the same old schema check, recover them the same way
after they finish rather than restarting their training.
