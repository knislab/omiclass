"""Prepare, validate, or run frozen-reference comparisons without touching manuscript outputs.

Invoke from the repository root: python -m experiments.reference_preprocessing.run --help
Imports of the full classification/R stack are deferred until `run`.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

from experiments.reference_preprocessing.cpu_budget import bootstrap, execution_limits

if __name__ == "__main__":
    bootstrap(sys.argv[1:])

import numpy as np
import pandas as pd
import yaml

from omiclass.batch.reference import ExpressionReference, FrozenQuantile, ReferenceRUV
from omiclass.batch.reference_pipeline import read_expression, read_ids, fit_reference

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (11, 61, 73, 80, 89)
METHODS = ("none", "limma", "frozen_quantile", "fsva_exact", "reference_ruv")
SOURCE = ROOT / "final_results_runs/repets_new/ighv/corrected"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False))


def training_tpm(counts, lengths, ids):
    """Existing length-adjusted TPM; schema selection uses training counts only."""
    train = counts.loc[:, ids]
    train = train.loc[train.index.intersection(lengths.index)]
    lens = lengths.loc[train.index].astype(float)
    if not np.isfinite(train.values).all() or (train.values < 0).any() or not np.isfinite(lens.values).all() or (lens <= 0).any():
        raise ValueError("Invalid training counts or fixed gene lengths")
    rpk = train.divide(lens / 1000, axis=0)
    denominator = rpk.sum(axis=0)
    if (denominator <= 0).any():
        raise ValueError("Zero training library size")
    tpm = rpk.divide(denominator, axis=1) * 1e6
    selected = (train > 0).any(axis=1)
    return tpm.loc[selected], train.index[~selected].tolist()


def read_training_counts(path):
    """Reproduce original load_matrix's first-duplicate policy, with an audit."""
    with Path(path).open(newline="") as stream:
        columns = next(csv.reader(stream))[1:]
    counts = pd.read_csv(path, index_col=0)
    counts.columns = columns  # restore identifiers before pandas' .1 mangling
    audit, first = [], {}
    for j, sample in enumerate(columns):
        if sample in first:
            audit.append(dict(sample=sample, kept_column_position=first[sample],
                excluded_column_position=j,
                identical_values=bool(np.array_equal(counts.iloc[:, first[sample]], counts.iloc[:, j]))))
        else:
            first[sample] = j
    counts = counts.loc[:, ~counts.columns.duplicated(keep="first")]
    ExpressionReference("log1p_tpm")._matrix(counts, fitting=True)
    return counts, audit


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records, universe = {}, None
    files = [args.expression, args.metadata]
    for seed in SEEDS:
        source = args.source_root / f"Omiclass_IGHV_corrected_seed{seed}"
        split_path = source / "split.json"
        cfg_path = source / "config.freeze.yaml"
        genes_path = source / "de_results/limma_voom_selected_genes.txt"
        split = json.loads(split_path.read_text())
        ids = split["train_index"] + split["test_index"]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate/overlapping source split IDs")
        if universe is not None and set(ids) != universe:
            raise ValueError("Manuscript seed reference populations differ")
        universe = set(ids)
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg["project"]["cluster_column"] != "ighv_mut":
            raise ValueError("Expected IGHV manuscript configuration")
        if any(cfg.get(k, {}).get("enabled", False) for k in ["bulkformer", "cellfm", "bulkrnabert"]):
            raise ValueError("Expected the expression-based Omiclass manuscript runs")
        target = out / f"seed{seed}"
        target.mkdir()
        (target / "split.json").write_text(split_path.read_text())
        (target / "config.yaml").write_text(cfg_path.read_text())
        (target / "genes.txt").write_text(genes_path.read_text())
        records[str(seed)] = dict(source=str(source.resolve()), split=str(split_path.resolve()))
        files += [split_path, cfg_path, genes_path]
    # The legacy merged file was intersected with DKFZ before saving. Never
    # use that intersection to define reference features or normalization.
    X = read_expression(args.expression)
    if universe - set(X.columns):
        raise ValueError("Source split samples missing from expression")
    counts, duplicate_audit = read_training_counts(args.training_counts)
    write_json(out / "source_duplicate_samples.json", dict(policy="existing_load_matrix_keep_first", exclusions=duplicate_audit))
    train_ids = [s for s in counts.columns if s in universe]
    if set(train_ids) != universe:
        raise ValueError("Training-only counts lack saved split samples")
    lengths_frame = pd.read_csv(args.gene_lengths, index_col=0)
    if not lengths_frame.index.is_unique:
        raise ValueError("Duplicate fixed gene-length identifiers")
    reference_tpm, zero_genes = training_tpm(counts, lengths_frame.iloc[:, 0], train_ids)
    genes = reference_tpm.index
    if set(genes) - set(X.index):
        raise ValueError("External expression lacks training-expressed reference genes; no intersection or zero fill allowed")
    # A provenance check, not parameter estimation: the saved training TPM
    # should use the same established normalization convention.
    source_training = X.loc[genes, train_ids]
    source_training = np.expm1(source_training) if args.input_scale == "log1p_tpm" else source_training
    if not np.allclose(source_training.values, reference_tpm.values, rtol=1e-5, atol=1e-6):
        raise ValueError("Legacy training TPM differs from independent count/length normalization")
    files += [args.training_counts, args.gene_lengths]
    # External outcome column is not loaded during preparation/correction.
    meta = pd.read_csv(args.metadata, index_col=0, usecols=[0, 1, 2, 3])
    if list(meta.columns) != ["cohort", "age", "sex"] or not meta.index.is_unique:
        raise ValueError("Expected unique metadata with leading columns: ID,cohort,age,sex")
    external_ids = meta.index[meta.cohort == "DKFZ"].tolist()
    if not external_ids or set(external_ids) & universe or set(external_ids) - set(X.columns):
        raise ValueError("Invalid external cohort or sample overlap")
    first_cfg = yaml.safe_load((out / "seed80/config.yaml").read_text())
    train_meta_path = Path(first_cfg["project"]["meta_path"])
    train_meta = pd.read_csv(train_meta_path, index_col=0).loc[train_ids]
    files += [train_meta_path]
    train_meta.to_csv(out / "training_metadata.csv")
    meta.loc[train_ids + external_ids].to_csv(out / "correction_covariates.csv")
    # Explicit scale, independently computed for each sample; never re-estimated.
    np.log1p(reference_tpm).to_csv(out / "training.csv")
    external = X.loc[genes, external_ids]
    if args.input_scale == "tpm":
        external = np.log1p(external)
    external.to_csv(out / "external.csv")
    write_json(out / "gene_schema.json", dict(policy="training_count_positive_in_at_least_one_reference_sample",
        selected_genes=genes.tolist(), excluded_training_zero_genes=zero_genes,
        selected_using_external_data=False, external_missing_value_policy="reject"))
    (out / "training_ids.txt").write_text("\n".join(train_ids) + "\n")
    pd.crosstab(train_meta.cohort, train_meta.ighv_mut).to_csv(out / "cohort_target_counts.csv")
    write_json(out / "manifest.json", dict(seeds=list(SEEDS), sources=records,
        training_count=len(train_ids), external_count=len(external_ids), gene_count=len(genes),
        source_input_scale=args.input_scale, prepared_input_scale="log1p_tpm",
        original_source_hashes={str(Path(p).resolve()): digest(p) for p in files},
        prepared_hashes={str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()},
        external_outcomes_loaded=False, expected_description_training_count=615,
        population_note="Saved manuscript splits define inclusion; inspect the actual count before full runs."))
    print(f"Prepared {len(train_ids)} CLL-Map and {len(external_ids)} DKFZ samples in {out}")


def check_inputs(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    # Check the prepared snapshot, including each seed's copied split/config/genes.
    for path, sha in manifest["prepared_hashes"].items():
        if digest(prepared / path) != sha:
            raise ValueError(f"Prepared input changed: {path}")
    return manifest


def build_context(args, out):
    from omiclass.config.schema import Config
    from omiclass.cli.runtime import RuntimeOptions
    raw = yaml.safe_load((args.prepared / f"seed{args.seed}/config.yaml").read_text())
    raw["project"].update(out_dir=str(out), split_data_dir=str(out),
        meta_path=str(args.prepared / "training_metadata.csv"), hold_out_path=None)
    raw["runtime"]["random_state"] = args.seed
    raw["runtime"]["n_jobs"] = getattr(args, "cores_per_seed", 5)
    raw["runtime"]["log_transform"] = False  # prepared data already log1p(TPM)
    raw["runtime"]["survival_prediction"]["enabled"] = False
    raw["batch_correction"].update(algorithm=args.method, input_scale="log1p_tpm",
        training_samples_path=str(args.prepared / "training_ids.txt"), reference_path=None,
        control_genes_path=str(args.controls.resolve()) if args.controls else None,
        control_justification=args.control_justification, n_factors=args.n_factors,
        rscript_path=str(args.rscript) if args.rscript else None,
        r_library_path=str(args.r_library) if args.r_library else None)
    cfg = Config.model_validate(raw)
    X = read_expression(args.prepared / "training.csv")
    genes = read_ids(args.prepared / f"seed{args.seed}/genes.txt")
    if set(genes) - set(X.index):
        raise ValueError("Saved training-only DE preselection genes absent from expression")
    (out / "split.json").write_text((args.prepared / f"seed{args.seed}/split.json").read_text())
    opts = RuntimeOptions(mode="batch-train", input_mode="batch_correction", run_batch=True,
                          run_de=False, reuse_split=True)
    return SimpleNamespace(cfg=cfg, opts=opts, out_dir=out, split_out=out,
        cache={"full_expr": X, "gene_list": genes, "_did_log_transform": True}, artifacts={})


def diagnostics(before, after, metadata, target_column="ighv_mut", batch_column="cohort"):
    """Training-only partial R2: cohort given target, target given cohort.

    These descriptive linear diagnostics do not establish absence of confounding.
    """
    cohort = pd.get_dummies(metadata[batch_column], drop_first=True).to_numpy(dtype=float)
    target = pd.get_dummies(metadata[target_column], drop_first=True).to_numpy(dtype=float)
    one = np.ones((len(metadata), 1))
    full = np.column_stack([one, cohort, target])
    result = {}
    for label, frame in [("before", before), ("after", after)]:
        Y = frame.loc[:, metadata.index].T.to_numpy()
        sse_full = np.sum((Y - full @ np.linalg.lstsq(full, Y, rcond=None)[0]) ** 2, axis=0)
        values = {}
        for name, reduced in [("cohort_given_target", np.column_stack([one, target])),
                              ("target_given_cohort", np.column_stack([one, cohort]))]:
            sse = np.sum((Y - reduced @ np.linalg.lstsq(reduced, Y, rcond=None)[0]) ** 2, axis=0)
            valid = sse > 1e-12
            values[name + "_mean_partial_r2"] = float(np.mean(1 - sse_full[valid] / sse[valid])) if valid.any() else None
        values["mean_gene_variance"] = float(Y.var(axis=0).mean())
        result[label] = values
    return result


def save_frame(path, frame):
    np.savez_compressed(path, values=frame.to_numpy(), genes=frame.index.to_numpy(dtype=str),
                        samples=frame.columns.to_numpy(dtype=str))


def load_frame(path):
    with np.load(path, allow_pickle=False) as data:
        return pd.DataFrame(data["values"], index=data["genes"], columns=data["samples"])


def preprocess(args):
    """One shared training-pool fit; classifier seeds never change this reference."""
    check_inputs(args.prepared)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ctx = build_context(args, out)
    if args.method not in {"frozen_quantile", "fsva_exact", "reference_ruv"}:
        raise ValueError("Shared preprocessing cache requires a fitted reference method")
    print(f"Fitting {args.method} on the declared training reference pool", flush=True)
    fit_reference(ctx)
    reference = out / "preprocessing/reference.npz"
    save_frame(out / "training.npz", ctx.cache["X_corrected"])
    external = read_expression(args.prepared / "external.csv")
    print(f"Applying {args.method} to {external.shape[1]} independent external samples", flush=True)
    external = ExpressionReference.load(reference).transform(external)
    save_frame(out / "external.npz", external)
    write_json(out / "complete.json", dict(method=args.method,
        prepared_manifest_sha256=digest(args.prepared / "manifest.json"), n_factors=args.n_factors,
        reference=str(reference.resolve()), external_labels_loaded=False,
        hashes={str(p.relative_to(out)): digest(p) for p in [reference, out / "training.npz", out / "external.npz"]}))
    print(f"Shared frozen preprocessing complete: {out}", flush=True)


def cached_preprocessing(args):
    root = args.preprocessing_cache.resolve()
    state = json.loads((root / "complete.json").read_text())
    if state["method"] != args.method or state["prepared_manifest_sha256"] != digest(args.prepared / "manifest.json"):
        raise ValueError("Shared preprocessing cache method/population mismatch")
    if state["n_factors"] != args.n_factors:
        raise ValueError("Shared preprocessing cache factor count mismatch")
    for name, sha in state["hashes"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Shared preprocessing cache changed: {name}")
    return load_frame(root / "training.npz"), load_frame(root / "external.npz"), state


def validate(args):
    check_inputs(args.prepared)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    X = read_expression(args.prepared / "training.csv")
    external = read_expression(args.prepared / "external.csv")
    meta = pd.read_csv(args.prepared / "training_metadata.csv", index_col=0)
    config = yaml.safe_load((args.prepared / f"seed{args.seed}/config.yaml").read_text())
    target = config["project"]["cluster_column"]
    design = list(dict.fromkeys([target] + (config["batch_correction"].get("design_columns") or [])))
    if args.method not in {"frozen_quantile", "reference_ruv", "fsva_exact"}:
        raise ValueError("Reference validation supports frozen_quantile, fsva_exact or reference_ruv")
    if args.method == "frozen_quantile":
        ref = FrozenQuantile("log1p_tpm")
    elif args.method == "fsva_exact":
        from omiclass.batch.fsva import FrozenSVA
        ref = FrozenSVA(design, args.n_factors, "log1p_tpm",
                         args.rscript, args.r_library)
    else:
        if not args.controls:
            raise ValueError("Supply --controls and --control-justification for reference_ruv")
        ref = ReferenceRUV(read_ids(args.controls), args.control_justification, args.n_factors,
                           design, "log1p_tpm")
    ref.fit(X, meta)
    ref.save(out / "reference.npz")
    before_hash = digest(out / "reference.npz")
    corrected = ref.transform(X)
    ext = ref.transform(external)
    for s in external.columns[:3]:
        pd.testing.assert_frame_equal(ext[[s]], ref.transform(external[[s]]))
    pd.testing.assert_frame_equal(ext, ExpressionReference.load(out / "reference.npz").transform(external))
    pd.testing.assert_frame_equal(corrected, ref.transform(X))
    assert before_hash == digest(out / "reference.npz")
    result = diagnostics(X, corrected, meta, target_column=target)
    result.update(training_count=X.shape[1], external_count=external.shape[1], method=args.method,
                  single_sample_equal=True, serialization_equal=True, external_labels_loaded=False,
                  factor_diagnostics=getattr(ref, "diagnostics_", {}))
    # Evaluate shifts in a PCA basis fitted only on corrected training expression.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(5, X.shape[1] - 1), svd_solver="randomized", random_state=11).fit(corrected.T)
    result["external_mean_training_pca_score"] = pca.transform(ext.T).mean(axis=0).tolist()
    write_json(out / "validation.json", result)
    print(json.dumps(result, indent=2))


def classifier_features(model):
    """Read the fitted public or Omiclass stacking schema without inferring it."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "_feature_names_in", None)
    if names is None:
        raise ValueError("Selected classifier lacks a saved input feature schema")
    names = np.asarray(names, dtype=object)
    if names.ndim != 1 or not len(names) or not all(isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError("Selected classifier has an invalid saved input feature schema")
    return names.tolist()


def external_metadata_path(args):
    explicit = getattr(args, "external_metadata", None)
    if explicit is not None:
        return explicit
    snapshot = args.prepared / "external_metadata.csv"
    return snapshot if snapshot.is_file() else ROOT / "final_results_runs/ighv/meta_merged_with_dkfz.csv"


def predict_external(model_path, external, classes, metadata_path, out, target_column="ighv_mut"):
    """Only saved-model prediction and scoring; no fitting or model selection."""
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    model = joblib.load(model_path)
    names = classifier_features(model)
    if set(names) - set(external.index):
        raise ValueError("External correction removed classifier features; no zero fill allowed")
    predictions = model.predict(external.loc[names].T)
    pd.DataFrame({"prediction": predictions}, index=external.columns).to_csv(out / "external_predictions.csv")
    # Outcome labels are opened only after predictions have been saved.
    id_column = pd.read_csv(metadata_path, nrows=0).columns[0]
    labels = pd.read_csv(metadata_path, index_col=0, usecols=[id_column, target_column])
    truth = labels.loc[external.columns, target_column]
    if not classes or set(truth.dropna()) - set(classes):
        raise ValueError("External outcomes incompatible with saved class names")
    mask = truth.notna().to_numpy()
    if not mask.any():
        raise ValueError("No labeled external samples to score")
    encoded = truth.map({c: i for i, c in enumerate(classes)})
    return dict(accuracy=float(accuracy_score(encoded[mask], predictions[mask])),
        f1_macro=float(f1_score(encoded[mask], predictions[mask], labels=list(range(len(classes))),
                                average="macro", zero_division=0)), n_scored=int(mask.sum()))


def evaluate(args):
    """Retry only external evaluation in a new directory using saved artifacts."""
    source = args.run_dir.resolve()
    if (source / "complete.json").exists():
        raise ValueError("Source run already completed external evaluation")
    selection = json.loads((source / "selection.json").read_text())
    config = yaml.safe_load((source / "config.freeze.yaml").read_text())
    args.method, args.seed = selection["method"], selection["seed"]
    bc = config["batch_correction"]
    if args.method != bc["algorithm"] or args.seed != config["runtime"]["random_state"]:
        raise ValueError("Saved selection/configuration method or seed mismatch")
    if args.method not in {"frozen_quantile", "fsva_exact", "reference_ruv"}:
        raise ValueError("Evaluation-only recovery requires a saved frozen preprocessing cache")
    args.n_factors = bc["n_factors"]
    args.prepared = (args.prepared or Path(bc["training_samples_path"]).parent).resolve()
    reference = bc.get("reference_path")
    if args.preprocessing_cache is None and not reference:
        raise ValueError("Supply --preprocessing-cache for the original frozen cache")
    args.preprocessing_cache = (args.preprocessing_cache or Path(reference).parent.parent).resolve()
    manifest = check_inputs(args.prepared)
    model_path = Path(selection["model"]).resolve()
    if not model_path.is_relative_to(source / "train_artifacts"):
        raise ValueError("Selected model is outside the source run's train_artifacts")
    files = [model_path, source / "selection.json", source / "config.freeze.yaml", source / "test_metrics.csv"]
    hashes = {str(p): digest(p) for p in files}
    _, external, _ = cached_preprocessing(args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metrics = predict_external(model_path, external, config["project"]["classes_names"],
        external_metadata_path(args), out, config["project"].get("cluster_column", "ighv_mut"))
    if any(digest(p) != sha for p, sha in hashes.items()):
        raise ValueError("Source artifacts changed during evaluation")
    # Copy only the small records needed for the existing summary collector.
    (out / "selection.json").write_text((source / "selection.json").read_text())
    (out / "test_metrics.csv").write_text((source / "test_metrics.csv").read_text())
    write_json(out / "external_metrics.json", metrics)
    write_json(out / "complete.json", dict(method=args.method, seed=args.seed, metrics=metrics,
        correction_fit_uses_external_expression=False, external_outcomes_used_for_fit=False,
        classifier_refitted=False, preprocessing_refitted=False, source_run_dir=str(source),
        source_hashes=hashes, source_population=manifest["training_count"],
        source_manifest_sha256=digest(args.prepared / "manifest.json")))
    print(metrics)


def run(args):
    manifest = check_inputs(args.prepared)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ctx = build_context(args, out)
    from omiclass.pipeline.steps.batch_step import BatchCorrectionStep
    from omiclass.pipeline.steps.train_eval_step import TrainAndEvalStep
    from omiclass.utils.pdf_export import configure_pdf_export
    configure_pdf_export(out, enabled=bool(ctx.cfg.PDF))
    step = BatchCorrectionStep()
    cached_external = None
    if args.preprocessing_cache:
        training, cached_external, cache_state = cached_preprocessing(args)
        if not training.columns.equals(ctx.cache["full_expr"].columns) or not training.index.equals(ctx.cache["full_expr"].index):
            raise ValueError("Cached training preprocessing schema differs from prepared inputs")
        ctx.cache["X_corrected"] = training
        ctx.cfg.batch_correction.reference_path = Path(cache_state["reference"])
    elif step.should_run(ctx):
        step.run(ctx)
    corrected = ctx.cache.get("X_corrected", ctx.cache["full_expr"])
    meta = pd.read_csv(args.prepared / "training_metadata.csv", index_col=0)
    common = corrected.columns.intersection(meta.index)
    write_json(out / "diagnostics.json", diagnostics(ctx.cache["full_expr"].loc[corrected.index, common],
                                                     corrected.loc[:, common], meta.loc[common],
                                                     target_column=ctx.cfg.project.cluster_column))
    write_json(out / "status.json", dict(method=args.method, seed=args.seed, stage="classification"))
    # Unmodified existing feature selection, tuning, CV, fitting and evaluation.
    TrainAndEvalStep().run(ctx)
    write_json(out / "status.json", dict(method=args.method, seed=args.seed, stage="external_evaluation"))
    write_json(out / "selection.json", dict(model=str(ctx.cache["best_model_path"]),
        seed=args.seed, method=args.method, selected_by="existing training CV"))
    external = read_expression(args.prepared / "external.csv")
    if cached_external is not None:
        external = cached_external
    elif args.method in {"frozen_quantile", "reference_ruv", "fsva_exact"}:
        external = ExpressionReference.load(out / "preprocessing/reference.npz").transform(external)
    elif args.method == "limma":
        # Reproduce the existing transductive external correction, retaining its
        # training-only corrected matrix/model unchanged on disk and in memory.
        ext_dir = out / "external_correction"
        ext_dir.mkdir()
        cfg = copy.deepcopy(ctx.cfg)
        cfg.project.meta_path = args.prepared / "correction_covariates.csv"
        extctx = SimpleNamespace(cfg=cfg, opts=ctx.opts, out_dir=ext_dir,
            cache={"full_expr": pd.concat([ctx.cache["full_expr"], external], axis=1),
                   "_did_log_transform": True}, artifacts={})
        step.run(extctx)
        external = extctx.cache["X_corrected"].loc[:, external.columns]
    metrics = predict_external(ctx.cache["best_model_path"], external, ctx.cfg.project.classes_names,
                               external_metadata_path(args), out, ctx.cfg.project.cluster_column)
    write_json(out / "external_metrics.json", metrics)
    write_json(out / "complete.json", dict(method=args.method, seed=args.seed, metrics=metrics,
        correction_fit_uses_external_expression=args.method == "limma", external_outcomes_used_for_fit=False,
        source_manifest_sha256=digest(args.prepared / "manifest.json"),
        source_population=manifest["training_count"],
        note="Shared saved training-only DE preselection; unchanged downstream classification framework."))
    write_json(out / "status.json", dict(method=args.method, seed=args.seed, stage="complete"))
    print(metrics)


def summarize(args):
    """Collect default selected-model holdout and external metrics, without fitting."""
    rows = []
    for marker in sorted(args.runs_root.rglob("complete.json")):
        state = json.loads(marker.read_text())
        if "metrics" not in state or "seed" not in state:
            continue  # preprocessing cache completion is not a classifier result
        root = marker.parent
        selection = json.loads((root / "selection.json").read_text())
        model = Path(selection["model"]).stem
        test = pd.read_csv(root / "test_metrics.csv")
        chosen = test.loc[test.model == model]
        if len(chosen) != 1:
            raise ValueError(f"Cannot identify selected internal holdout metrics: {root}")
        row = dict(method=state["method"], seed=state["seed"], model=model,
            external_accuracy=state["metrics"]["accuracy"], external_f1_macro=state["metrics"]["f1_macro"],
            internal_accuracy=float(chosen.iloc[0]["accuracy"]),
            internal_f1_macro=float(chosen.iloc[0]["f1_macro"]),
            external_expression_used_for_fit=state["correction_fit_uses_external_expression"], run=str(root))
        rows.append(row)
    if not rows:
        raise ValueError("No completed classifier runs found")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_seed.csv", index=False)
    values = ["internal_accuracy", "internal_f1_macro", "external_accuracy", "external_f1_macro"]
    frame.groupby("method")[values].agg(["mean", "std", "count"]).to_csv(out / "summary.csv")
    print(frame.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", type=Path, default=SOURCE)
    prep.add_argument("--expression", type=Path, default=ROOT / "final_results_runs/ighv/tpm_merged_with_dkfz.csv")
    prep.add_argument("--metadata", type=Path, default=ROOT / "final_results_runs/ighv/meta_merged_with_dkfz.csv")
    prep.add_argument("--training-counts", type=Path, default=ROOT / "final_results_runs/ighv/filtered_count.csv")
    prep.add_argument("--gene-lengths", type=Path, default=ROOT / "omiclass/data/gencode_GRCh37_hg19.csv")
    prep.add_argument("--input-scale", choices=["tpm", "log1p_tpm"], required=True)
    prep.add_argument("--out", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--runs-root", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    recovery = sub.add_parser("evaluate", help="External evaluation only; reuse the saved model and frozen cache")
    recovery.add_argument("--run-dir", type=Path, required=True)
    recovery.add_argument("--out", type=Path, required=True)
    recovery.add_argument("--prepared", type=Path)
    recovery.add_argument("--preprocessing-cache", type=Path)
    recovery.add_argument("--cores-per-seed", type=int, default=5)
    recovery.add_argument("--cpu-list")
    recovery.add_argument("--external-metadata", type=Path)
    for command in ("validate", "preprocess", "run"):
        p = sub.add_parser(command)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--seed", type=int, choices=SEEDS, default=80)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--controls", type=Path)
        p.add_argument("--control-justification")
        p.add_argument("--n-factors", type=int, default=1)
        p.add_argument("--rscript", type=Path)
        p.add_argument("--r-library", type=Path)
        p.add_argument("--preprocessing-cache", type=Path)
        p.add_argument("--cores-per-seed", type=int, default=5)
        p.add_argument("--cpu-list", help="Internal launcher-assigned CPU affinity IDs")
        p.add_argument("--external-metadata", type=Path)
    args = parser.parse_args()
    if getattr(args, "prepared", None) is not None:
        args.prepared = args.prepared.resolve()
    out_existed = args.out.exists()
    try:
        if hasattr(args, "cores_per_seed"):
            write_allocation = dict(cores_per_seed=args.cores_per_seed,
                cpu_ids=sorted(os.sched_getaffinity(0)), native_threads=1,
                nested_joblib="sequential")
            with execution_limits(args.cores_per_seed):
                globals()[args.command](args)
            write_json(args.out / "cpu_allocation.json", write_allocation)
        else:
            globals()[args.command](args)
    except Exception as exc:
        if not out_existed and args.out.is_dir() and not (args.out / "complete.json").exists():
            write_json(args.out / "failure.json", dict(command=args.command, error=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
