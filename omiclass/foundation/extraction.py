"""Model-specific manuscript preprocessing and frozen extraction adapters."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from omiclass.foundation.io import write_json, token_schema, validate_tokens, gene_order_digest


@dataclass
class Extraction:
    features: pd.DataFrame
    metadata: dict
    token_cache: Path | None = None


LEGACY_INPUT_IDENTIFIERS = 'legacy_bulkrnabert_csv_v1'
CELLFM_INPUT_IDENTIFIERS = 'cellfm_gene_symbols_v1'


def input_identifier_protocol(cfg, manifest=None):
    if manifest is None:
        return CELLFM_INPUT_IDENTIFIERS if backend_name(cfg) == 'cellfm' else LEGACY_INPUT_IDENTIFIERS
    protocol = manifest.get('extraction_protocol', {}).get('input_identifiers', LEGACY_INPUT_IDENTIFIERS)
    if protocol not in {LEGACY_INPUT_IDENTIFIERS, CELLFM_INPUT_IDENTIFIERS}:
        raise ValueError('Unsupported saved input identifier protocol')
    if protocol == CELLFM_INPUT_IDENTIFIERS and backend_name(cfg) != 'cellfm':
        raise ValueError('Saved input identifier protocol does not match the encoder')
    return protocol


def read_expression(path, *, protocol=LEGACY_INPUT_IDENTIFIERS):
    # Retain the exact historical reader for old bundles and the other encoders.
    from omiclass.features.bulkrnabert import read_expression_csv
    if protocol == LEGACY_INPUT_IDENTIFIERS:
        return read_expression_csv(Path(path))
    if protocol != CELLFM_INPUT_IDENTIFIERS:
        raise ValueError('Unsupported input identifier protocol')
    from omiclass.errors.errors import DataError
    from omiclass.utils.identifiers import validate_delimited_header, validate_frame_identifiers
    validate_delimited_header(path, error_type=DataError)
    frame = pd.read_csv(path, index_col=0, converters={0: str})
    # CellFM deliberately sums repeated genes, including mapping collisions.
    # Keep them for its existing preparation routine; never discard a row here.
    validate_frame_identifiers(frame, allow_duplicate_features=True, error_type=DataError)
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError('CellFM expression must be numeric') from exc
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(axis=0) <= 0).any():
        raise DataError('CellFM requires finite nonnegative counts and nonempty samples')
    return pd.DataFrame(values, index=frame.index.map(str), columns=frame.columns.map(str))


def backend_name(cfg):
    return "cellfm" if cfg.foundation.model.startswith("cellfm") else cfg.foundation.model


def continuous_input(counts, cfg, *, legacy_cellfm_single_view=False):
    """Prepare exactly once, before optional cohort correction; no learned reduction."""
    backend = backend_name(cfg)
    if backend == "bulkformer":
        from omiclass.features.bulkformer import normalize_counts_to_log_tpm, load_bulkformer_gene_lengths
        return normalize_counts_to_log_tpm(counts, load_bulkformer_gene_lengths(cfg.bulkformer.gene_lengths_path))
    if backend == "cellfm":
        if cfg.foundation.correction == "none":
            return counts
        from omiclass.foundation.cellfm_corrected import continuous_expression
        return continuous_expression(counts, cfg.cellfm, legacy_single_view=legacy_cellfm_single_view)
    from omiclass.features.bulkrnabert import prepare_tpm
    if cfg.foundation.input_scale == "tpm":
        tpm = counts
    elif cfg.bulkrnabert.tpm_input_path is not None:
        tpm = read_expression(cfg.bulkrnabert.tpm_input_path)
    else:
        from omiclass.data.processor import DataProcessor
        from omiclass.features.bulkrnabert import normalize_gene_ids, validate_expression
        path = cfg.normalization.gene_lengths_path
        if path is None:
            raise ValueError("BulkRNABert counts require verified gene_lengths_path, or use foundation.input_scale='tpm'")
        lengths = DataProcessor(cfg).load_precomputed_gene_lengths(str(path))
        frame = validate_expression(counts)
        frame.index = normalize_gene_ids(frame.index)
        lengths.index = normalize_gene_ids(lengths.index)
        retained = frame.index[frame.index.isin(lengths.index)]
        if not len(retained):
            raise ValueError("No count genes have verified lengths")
        values = pd.to_numeric(lengths.reindex(retained), errors="raise")
        if values.isna().any() or (values <= 0).any() or not np.isfinite(values).all():
            raise ValueError("Gene lengths must be finite and positive; no length imputation")
        rpk = frame.loc[retained].div(values / 1000.0, axis=0)
        tpm = validate_expression(rpk.div(rpk.sum(axis=0), axis=1) * 1e6)
    genes = Path(cfg.bulkrnabert.vocabulary_path).read_text().splitlines()
    values, _ = prepare_tpm(tpm, genes, counts.columns.tolist())
    # This is the official preprocess_omic's exact float32 log10(TPM+1).
    return pd.DataFrame(np.log10(values + 1).T, index=genes, columns=counts.columns)


def correct_input(frame, cfg, out, metadata):
    """Historical transductive cohort correction, intentionally retained for Step 2."""
    from omiclass.batch.limma_runner import LimmaRunner
    bc = cfg.batch_correction
    cols = [bc.batch_column, *(bc.design_columns or [])]
    if not metadata.index.is_unique or not set(cols).issubset(metadata.columns):
        raise ValueError("Unique sample metadata and every correction variable are required")
    meta = metadata.reindex(frame.columns).loc[:, cols]
    if meta.isna().any().any() or meta[bc.batch_column].nunique() < 2:
        raise ValueError("Correction requires complete metadata and at least two batches")
    directory = out / "limma"
    directory.mkdir(exist_ok=False)
    LimmaRunner().run_limma(expr_df=frame, meta_df=meta, output_dir=directory,
        batch_col=bc.batch_column, design_cols=bc.design_columns, n_pc=0,
        na_fill=bc.fill_na or None, out_csv="corrected.csv", plot_n_pcs=10)
    result = pd.read_csv(directory / "corrected.csv", index_col=0)
    result.index, result.columns = result.index.map(str), result.columns.map(str)
    if set(result.columns) != set(frame.columns):
        raise ValueError("Limma removed requested samples; refusing a population change")
    result = result.loc[:, frame.columns]
    if backend_name(cfg) == "bulkrnabert":
        from omiclass.foundation.correction_compat import restore_bulkrnabert_zero_genes, clip_bulkrnabert_corrected
        result, restored = restore_bulkrnabert_zero_genes(frame, result)
        write_json(out / "restored_zero_genes.json", restored)
        result = clip_bulkrnabert_corrected(result, out)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite corrected expression")
    write_json(out / "correction.json", dict(protocol="historical_full_input_limma",
        transductive=True, outcome_labels_used=False, fitted_coefficients_transferred=False,
        supplied_samples=frame.columns.tolist(), retained_samples=result.columns.tolist(),
        batch_column=bc.batch_column, design_columns=bc.design_columns))
    return result


class TokenObserver:
    """Stream existing forward results; never recompute or alter a model forward."""
    def __init__(self, directory, backend, num_views):
        self.directory = directory
        self.backend = backend
        self.num_views = num_views
        self.records = []
        self.coordinates = set()
        self.schema = None
        self.cellfm_protocol = {}
        directory.mkdir(exist_ok=False)

    def set_cellfm_vocabulary(self, vocabulary, *, random_state, legacy_single_view=False):
        if self.schema is not None:
            raise ValueError('Cannot change the token protocol after recording tokens')
        self.cellfm_protocol = dict(vocabulary_size=len(vocabulary),
            vocabulary_sha256=gene_order_digest(vocabulary), selection_random_state=int(random_state),
            sampling_strategy=('legacy_global_numpy_random_state' if legacy_single_view else
                               'sha256_sample_id_global_seed_zero_based_view_index'))

    def __call__(self, tokens, valid, genes, ids, view):
        def array(value, floating=False):
            if hasattr(value, "detach"):
                value = value.detach().float() if floating else value.detach()
                return value.cpu().numpy()
            return np.asarray(value, dtype=np.float32) if floating else np.asarray(value)
        values, masks, gene_ids = array(tokens, True), array(valid), array(genes)
        if (values.ndim != 3 or masks.shape != values.shape[:2] or gene_ids.shape != values.shape[:2]
                or len(ids) != len(values) or not isinstance(view, (int, np.integer))
                or not 0 <= int(view) < self.num_views):
            raise ValueError("Token/mask/sample shape mismatch")
        for i, sid in enumerate(ids):
            sid = str(sid)
            coordinate = (sid, int(view))
            if coordinate in self.coordinates or not masks[i].any() or not np.isfinite(values[i]).all():
                raise ValueError("Duplicated or invalid sample/view token record")
            if self.schema is None:
                self.schema = token_schema(self.backend, values[i].shape, gene_ids[i])
                self.schema.update(self.cellfm_protocol)
            validate_tokens(values[i], masks[i], gene_ids[i], schema=self.schema)
            self.coordinates.add(coordinate)
            filename = f"record_{len(self.records):08d}.npz"
            np.savez(self.directory / filename, tokens=values[i], mask=masks[i], gene_ids=gene_ids[i])
            self.records.append(dict(sample_id=sid, view=int(view), file=filename))

    def finish(self, sample_ids):
        expected = {(str(s), v) for s in sample_ids for v in range(self.num_views)}
        if self.coordinates != expected:
            raise ValueError("Incomplete token view traversal")
        write_json(self.directory / "manifest.json", dict(format_version=1, backend=self.backend,
            metadata=dict(num_views=self.num_views, token_schema=self.schema), records=self.records))


def extract_bert(frame, cfg, out, tokens):
    from omiclass.features import bulkrnabert_worker
    b = cfg.bulkrnabert
    np.savez(out / "worker-input.npz", tpm=frame.T.to_numpy(), sample_ids=np.asarray(frame.columns, dtype=str),
             gene_ids=np.asarray(frame.index, dtype=str))
    cmd = [str(Path(b.python_executable).expanduser().resolve()), "-B", str(Path(bulkrnabert_worker.__file__).resolve()),
        "--root-dir", str(Path(b.root_dir).resolve()), "--checkpoint-directory", str(Path(b.checkpoint_directory).resolve()),
        "--model-name", b.model_name, "--input", str(out / "worker-input.npz"), "--output", str(out / "worker-output.npz"),
        "--batch-size", str(b.batch_size), "--device", b.device, "--input-scale", "log10_tpm"]
    token_cache = out / "tokens" if tokens else None
    if token_cache is not None:
        cmd += ["--token-dir", str(token_cache)]
    runtime = out / "runtime"
    runtime.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAX_COMPILATION_CACHE_DIR=str(runtime / "jax"), XDG_CACHE_HOME=str(runtime / "cache"))
    env.pop("PYTHONPATH", None)
    subprocess.run(cmd, cwd=out, env=env, check=True)
    with np.load(out / "worker-output.npz", allow_pickle=False) as z:
        if z["sample_ids"].astype(str).tolist() != frame.columns.tolist():
            raise ValueError("BERT worker sample order changed")
        features = pd.DataFrame(z["embeddings"], index=frame.columns,
            columns=[f"bulkrnabert_feature_{i:04d}" for i in range(z["embeddings"].shape[1])])
        metadata = json.loads(str(z["metadata"]))
    return Extraction(features, metadata, token_cache)


def encoder_seed(cfg):
    backend = backend_name(cfg)
    if backend == 'cellfm':
        return int(cfg.cellfm.random_state), 'cellfm.random_state'
    if backend == 'bulkrnabert':
        return 0, 'fixed JAX PRNGKey(0)'
    return int(cfg.foundation.extraction_seed), 'foundation.extraction_seed'


def extract(counts, cfg, out, metadata=None, *, legacy_cellfm_single_view=False):
    from omiclass.utils.reproducibility import random_state_scope
    seed, _ = encoder_seed(cfg)
    pytorch = backend_name(cfg) != 'bulkrnabert'
    with random_state_scope(seed if pytorch else None, torch_enabled=pytorch,
                            cuda=pytorch and getattr(cfg, backend_name(cfg)).device != 'cpu'):
        return _extract(counts, cfg, out, metadata, legacy_cellfm_single_view=legacy_cellfm_single_view)


def _extract(counts, cfg, out, metadata=None, *, legacy_cellfm_single_view=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    backend = backend_name(cfg)
    legacy = {'legacy_cellfm_single_view': True} if legacy_cellfm_single_view else {}
    frame = continuous_input(counts, cfg, **legacy)
    if cfg.foundation.correction == "limma":
        frame = correct_input(frame, cfg, out, metadata)
    tokens = cfg.foundation.representation == "tokens"
    if backend == "bulkrnabert":
        result = extract_bert(frame, cfg, out, tokens)
    else:
        observer = TokenObserver(out / "tokens", backend, cfg.cellfm.num_views if backend == "cellfm" else 1) if tokens else None
        if backend == "bulkformer":
            from omiclass.features.bulkformer import BulkFormerFeatureExtractor
            if (frame.to_numpy() == -10).any():
                raise ValueError("Input collides with BulkFormer missing-gene sentinel")
            encoded = BulkFormerFeatureExtractor(cfg.bulkformer, observer=observer, continuous=True).extract(frame)
            result = Extraction(encoded.features.T, encoded.metadata, out / "tokens" if tokens else None)
        elif cfg.foundation.correction == "limma":
            from omiclass.foundation.cellfm_corrected import extract_corrected
            kwargs = {'legacy_single_view': True} if legacy_cellfm_single_view else {}
            features, info = extract_corrected(counts, frame, cfg.cellfm, out, observer=observer, **kwargs)
            result = Extraction(features, info, out / "tokens" if tokens else None)
        else:
            from omiclass.features.cellfm import CellFMFeatureExtractor
            kwargs = {'legacy_single_view': True} if legacy_cellfm_single_view else {}
            encoded = CellFMFeatureExtractor(cfg.cellfm, observer=observer, **kwargs).extract(counts)
            result = Extraction(encoded.features.T, encoded.metadata, out / "tokens" if tokens else None)
        if observer is not None:
            observer.finish(counts.columns)
    if result.features.index.has_duplicates or not np.isfinite(result.features.to_numpy()).all():
        raise ValueError("Invalid extracted sample embeddings")
    seed, source = encoder_seed(cfg)
    write_json(out / "extraction.json", dict(backend=backend, representation=cfg.foundation.representation,
        encoder_seed=seed, seed_source=source, configured_extraction_seed=cfg.foundation.extraction_seed,
        metadata=result.metadata))
    return result
