"""Integration of frozen references with the existing preprocessing caches."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from omiclass.batch.reference import ExpressionReference, FrozenQuantile, ReferenceRUV


FROZEN_METHODS = {"frozen_quantile", "reference_ruv", "fsva_exact"}


def read_expression(path):
    """Strict loader: legacy load_matrix silently drops duplicate genes."""
    path = Path(path)
    if path.suffix == ".parquet":
        X = pd.read_parquet(path)
    elif path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        with path.open(newline="") as stream:
            header = next(csv.reader(stream, delimiter=sep))[1:]
        if any(not s.strip() for s in header) or len(header) != len(set(header)):
            raise ValueError("Missing or duplicate sample identifiers in expression file")
        X = pd.read_csv(path, sep=sep, index_col=0)
    else:
        raise ValueError("Expression must be CSV, TSV or parquet")
    if X.index.hasnans or X.columns.hasnans:
        raise ValueError("Missing gene or sample identifiers in expression file")
    X.index, X.columns = X.index.map(str), X.columns.map(str)
    # Validate before any legacy zero fill, duplicate removal or float32 cast.
    ExpressionReference("log1p_tpm")._matrix(X, fitting=True)
    return X


def read_ids(path):
    ids = Path(path).read_text().splitlines()
    ids = [s.strip() for s in ids if s.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Expected nonempty unique identifiers, one per line: {path}")
    return ids


def fit_reference(ctx):
    bc = ctx.cfg.batch_correction
    if ctx.opts.input_mode == "raw_counts":
        raise ValueError("Frozen references require TPM or log1p(TPM); normalize counts with a fixed gene-length/schema protocol first")
    if bc.training_samples_path is None:
        raise ValueError("batch_correction.training_samples_path must explicitly list the training reference samples")
    ids = read_ids(bc.training_samples_path)
    X = ctx.cache["full_expr"]
    absent = set(ids) - set(X.columns)
    if absent:
        raise ValueError(f"Training reference samples absent from input: {sorted(absent)[:10]}")
    train = X.loc[:, ids]
    metadata = None
    if bc.algorithm == "frozen_quantile":
        reference = FrozenQuantile(input_scale=bc.input_scale)
    else:
        if bc.algorithm == "reference_ruv" and bc.control_genes_path is None:
            raise ValueError("reference_ruv requires control_genes_path; controls are never selected by low variance")
        columns = list(dict.fromkeys([ctx.cfg.project.cluster_column] + (bc.design_columns or [])))
        meta_path = Path(ctx.cfg.project.meta_path)
        # Read only training design columns, never external outcomes at application.
        sep = "\t" if meta_path.suffix == ".tsv" else ","
        metadata = pd.read_csv(meta_path, sep=sep, index_col=0)
        metadata.index = metadata.index.map(str)
        if not metadata.index.is_unique:
            raise ValueError("Training metadata has duplicate sample identifiers")
        metadata = metadata.loc[ids, columns]
        if bc.algorithm == "fsva_exact":
            from omiclass.batch.fsva import FrozenSVA
            reference = FrozenSVA(columns, bc.n_factors, bc.input_scale,
                                  bc.rscript_path, bc.r_library_path, bc.reference_random_state)
        else:
            reference = ReferenceRUV(read_ids(bc.control_genes_path), bc.control_justification,
                                     bc.n_factors, columns, bc.input_scale)
    reference.fit(train, metadata)
    output = ctx.out_dir / "preprocessing"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "reference.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to replace fitted reference: {path}")
    reference.save(path)
    # Only declared training-reference samples enter classification. External
    # expression supplied in the same file never enters its input matrix.
    corrected = reference.transform(train)
    corrected.to_csv(output / "training_expression.csv")
    (output / "reference.json").write_text(json.dumps(dict(
        method=bc.algorithm, input_scale=bc.input_scale, output_scale="log1p_tpm_corrected",
        training_sample_ids=ids, genes=reference.genes_, external_fit=False,
        control_justification=bc.control_justification if bc.algorithm == "reference_ruv" else None,
        diagnostics=getattr(reference, "diagnostics_", {})), indent=2))
    ctx.cache["X_corrected"] = corrected
    ctx.cache["_did_log_transform"] = True
    ctx.artifacts["preprocessing_reference"] = path


def transform_reference(ctx):
    bc = ctx.cfg.batch_correction
    if ctx.opts.input_mode == "raw_counts":
        raise ValueError("External frozen transformation requires TPM or log1p(TPM), matching the saved reference")
    path = bc.reference_path or (ctx.out_dir / "preprocessing" / "reference.npz")
    reference = ExpressionReference.load(path)
    if reference.method != bc.algorithm or reference.input_scale != bc.input_scale:
        raise ValueError("Configured method/input_scale differs from the saved reference")
    X = ctx.cache["full_expr"]
    # Support both external-only input and the legacy combined-input invocation.
    external = X.loc[:, ~X.columns.isin(reference.training_samples_)]
    if external.empty:
        raise ValueError("No unseen samples remain after excluding training reference IDs")
    corrected = reference.transform(external)
    ctx.cache["X_recorrected"] = corrected
    ctx.cache["_did_log_transform"] = True
    output = ctx.out_dir / "preprocessing_external"
    output.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output / "expression.csv")
    (output / "provenance.json").write_text(json.dumps(dict(
        reference_path=str(Path(path).resolve()), method=reference.method,
        sample_ids=corrected.columns.tolist(), refitted=False, external_labels_accessed=False), indent=2))
