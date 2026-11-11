# Publication candidate review

The approved foundation input correction, public-source separation and packaging/documentation work are prepared for review. **Publication remains blocked** on the software license, permission-dependent resources, exact encoder acquisition/real execution checks. No publication, push, GitHub release, commit, remote change or inherited-history rewrite was performed.

The separately requested raw-count routing correction was explicitly approved after the unmocked validation finding and is included below. The source baseline is `c58f2338be5d8aa7dadeedb16baec9c994b29749`, plus the reviewed working-copy changes and per-file snapshot hashes. All 104 configurations remain: BulkFormer 41, CellFM one-view 21, CellFM twenty-view 21, BulkRNABert 21. Architectures, search spaces, classifiers, representations, calibration choices and intended correction/head protocols are unchanged.

## Phase 1: input correctness

`omiclass.utils.identifiers` validates nonempty, trimmed, unique axes and raw CSV/TSV headers before pandas can rename duplicates. Core matrix/metadata loaders reject duplicates instead of keeping the first row/column. Parquet validation is covered with a reader stub. Validation preserves valid identifier types and does not apply new biological transformations.

New CellFM foundation input preserves literal symbols, including dotted suffixes and numeric-looking strings. Its existing preparation still sums repeated input genes and Ensembl-mapping collisions before count capping/selection. The public input reader does not discard those rows. BulkFormer and BulkRNABert retain the exact historical BERT-style version-removal path; this stage does not change their normalization/tokenizer protocols.

New foundation manifests add `extraction_protocol.input_identifiers`; prediction records the protocol used. Historical bundles lacking that marker use the original shared reader, including its historical suffix removal. Unknown/incompatible markers fail. Existing model/config import paths, format versions and historical CellFM sampling replay are retained. Tests verify old/new dispatch, real SVM head serialization with mocked encoders, and training-bundle immutability.

Correcting CellFM names can change new-run mappings and therefore extracted values when the old reader damaged symbols. That is the approved correctness correction. Duplicate/empty/whitespace identifiers that were previously ambiguous now fail. Historical invalid core inputs are not promised silent-duplicate compatibility. No real checkpoint was used to estimate the frequency/performance effect.

## Phase 2: publication boundary

A private source-only publication candidate is assembled by an explicit allowlist, with no `.git` directory, inherited history, gitlinks or remotes. Its per-file size/SHA256 inventory is retained separately for owner review. No Git mutations were needed in the linked public worktree, whose metadata belongs to the manuscript repository.

Research-only tests, launchers, experiments, CellFM tutorial/cluster/notebook fragments, old Docker environment export, historical development reports, screenshots and generated HTML were moved to an ignored quarantine inside the public worktree. They were not deleted from or changed in the manuscript checkout. The public reference test no longer imports a benchmark helper; a benchmark-only case was archived. CellFM mapping tests now use an explicit test mapping instead of depending on redistribution of bundled annotation data.

The proposed inventory contains 159 allowlisted source/configuration/documentation/test files, without Git metadata. The candidate retains the API, existing framework modules and import paths, three required R scripts, BERT worker, source-adaptation hashes, public tests, current guides, argument-based examples, packaging files, core environment/Docker configuration and a CPU test workflow. It excludes unverified GENCODE/surface-marker resources and historical example data under the owner's explicit instruction. The existing private copies remain available and unchanged.

A bounded scan checks credential patterns, personal/server absolute paths, symlinks and prohibited data/report artifacts. No candidates were found in the publication snapshot. This is not a general security/permission certification. The inherited repository history is not being proposed for direct publication.

## Phase 3: packaging and documentation

- `pyproject.toml`, `setup.cfg`, `setup.py` and `MANIFEST.in` provide namespace-package discovery while retaining `api` and `omiclass.*`. R scripts and worker code are included. Unverified annotations/data and the empty legacy LICENSE are explicitly excluded.
- Core direct dependencies are declared; optional groups describe Torch heads, BulkFormer/common CellFM requirements, R bridge, I/O and tests. JAX remains in an explicitly configured external BERT environment. No guessed CellFM/JAX version is claimed as compatible.
- The core Conda manifest replaces a research-environment export. Docker uses that manifest, installs the source package, excludes quarantine/data/checkpoints, and retains provenance in the mounted output by default. Neither genome annotation is silently omitted while the other is bundled: both are deliberately excluded until permissions are resolved.
- README/examples use correct matrix orientation, actual saved-run directories and prepared-expression marker inputs. HTML-only folders fail the prediction example's prerequisite check. Foundation and core prediction contracts are separate.
- Frozen-reference documentation reflects training-partition/CV-local fitting; transductive limma remains cohort-based, including intended external re-correction. CellFM sampled-position limitations and current absolute-path/dependency-version portability restrictions are explicit.
- A proposed GitHub workflow collects/runs CPU synthetic tests and builds a wheel. It has read-only repository permissions and no publication job. No remote CI was executed.

The owner has not selected a software license. Metadata intentionally does not declare one; older setuptools renders `License: UNKNOWN`. Development version `0.0.0.dev0` is an internal build placeholder, not a chosen public release/tag. No third-party encoder source/checkpoint is included.

## Validation evidence

| Check | Result and scope |
| --- | --- |
| Phase 1 focused | 80 passed: input/identifier contracts, replay seams, restoration and extraction checks. No real pretrained encoder/native R calculation. |
| Phase 1 broader integration | 255 passed: includes all 104 mocked-encoder API cases and split/evaluation regressions. Real small classifier/head computations occur where the tests specify them. |
| Phase 2 focused | 52 passed: retained reference/input/CellFM checks. Public collection succeeded without manuscript-module imports. |
| Initial Phase 3 focused | 40 passed: packaging metadata, exclusions, example argument handling and strict input checks. |
| Installed-wheel full suite | 471 passed, 1 optional native-fSVA test skipped before the final example-contract assertion was added. CPU fits include real sklearn/RF, XGBoost, LightGBM and Torch components; pretrained extraction is mocked. |
| Separate raw-count focused checks | 77 passed before the final saved-mode example assertion was added; includes routing, normalization, historical-scale, evaluation and backend configuration checks. |
| Final public suite | **479 passed, 1 optional native-fSVA check skipped**, after the separately approved raw-count fix and saved-mode example test. Complete collection contains 480 cases across 23 modules, without manuscript-module imports. |
| Offline distributions | Wheel and source archive built without dependency downloads. Required R/worker files present; unapproved annotations, datasets, model arrays, generated reports, research modules and LICENSE absent. |
| Isolated installation | Wheel installed with `--no-index --no-deps` into a new worktree-local target. Imports from outside the checkout resolve to that target. All 104 configurations validate there. Existing dependencies were read only; this is not a fresh dependency solve. |
| Real marker workflows | Unmocked RF, LightGBM, XGBoost and stacked candidates completed on small generated fixtures. After the approved correction, raw-count training stored `raw_counts` plus fitted lengths and produced four external predictions. An earlier ready-expression model replayed under the corrected code with model files and prediction CSV byte-identical to the pre-fix version. No encoder or native-R calculation was used. |

Private evidence is retained under `.step3-validation/release/phase1`, `phase2` and `phase3`; it is not part of the candidate. Ordinary pandas/skopt/SciPy warnings remain visible. No manuscript benchmark, pretrained checkpoint execution or GPU kernel validation was performed. Native R dependency-guard checks are distinct from scientific native-R execution; the optional fSVA integration is skipped without an explicit compatible library.

## Separately approved raw-count routing correction

The unmocked core marker check found a pre-existing route inconsistency: training with `raw_count=True`, DE disabled and batch correction disabled records `ready_for_training`, so raw-count normalization is not invoked. The saved model then rejects a later explicit raw-count request because it must preserve its recorded scale.

The owner explicitly authorized a separate fix for new training while preserving historical replay. New raw-count/no-DE/no-batch training now selects `raw_counts` instead of ready-expression mode, invoking the existing gene-length/TPM/log1p normalization and persisting its lengths. This is routing to the existing method, not a new algorithm. New normalized values and resulting fitted models can differ from the formerly ignored request, as explicitly approved. Historical saved input modes remain authoritative; no old ready-expression model is silently normalized. Targeted tests cover new routing, prepared input, old/new saved-scale mismatches and exactly-once normalization with fitted-length replay. Prediction examples now defer to the saved mode and reject foundation runs. The minimal marker example intentionally remains on a prepared scale.

## Remaining release gates and options

1. **Software license:** owner selects/approves it before publication. No license was inferred.
2. **Resources:** GENCODE, surface-marker and old example-data permissions/provenance remain unverified and excluded. Options are verified authorized redistribution, or a separately reviewed external-resource arrangement. No synthetic replacement annotation is shipped. Annotation fallback, Ensembl mapping and resource-dependent reports need explicit validation with approved resources.
3. **Encoder source/assets:** historical BulkFormer source/checkpoint availability and the exact compatible CellFM PyTorch port require confirmation. A cited CellFM-torch URL was not retrievable. Do not substitute a newer model/architecture. BERT source/dependency/checkpoint/vocabulary matching also needs real execution verification. No large assets were downloaded.
4. **Routing compatibility:** review the separately approved raw-count correction and its recorded numerical difference. Historical saved scales remain authoritative; explicit mismatches still fail.
5. **Environment execution:** fresh dependency resolution, core Docker build/native-R smoke tests, actual GPU/encoder extraction and all four real token protocols remain unverified. The proposed workflow is not a completed CI result.
6. **Compatibility:** signed foundation paths currently restrict relocation even with identical assets; historical bundles requiring omitted annotations cannot gain verified replay until their original authorized resources are supplied. No signatures were weakened or artifacts migrated.
7. **Final publication:** review the source inventory/hashes, privacy/resource decisions and validation results, then approve the destination/version and publication action. Do not push the linked manuscript history or alter shared remotes.

All preparation, builds, installations, generated fixtures, caches and quarantine writes stay inside the public worktree. The original manuscript checkout/history, running workers, saved results, shared environments/dependencies and checkpoints remain untouched. The candidate is unpublished and implementation changes remain uncommitted.
