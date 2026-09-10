from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from omiclass.config.schema import CellFMCfg
from omiclass.errors.errors import DataError


CELLFM_EMBEDDING_DIMENSION = 1536
CELLFM_GENES_PER_VIEW = 2048


@dataclass
class CellFMPreparedInput:
    counts: np.ndarray
    gene_ids: np.ndarray
    sample_ids: list[str]
    mapped_genes: list[str]
    unmapped_genes: list[str]
    raw_library_sizes: np.ndarray
    ensembl_mapped_genes: int = 0


@dataclass
class CellFMExtractionResult:
    features: pd.DataFrame
    metadata: dict[str, Any]


class _BulkSampleDataset:
    def __init__(self, prepared: CellFMPreparedInput):
        self.counts = prepared.counts.astype(np.float32, copy=False)
        self.gene_ids = prepared.gene_ids.astype(np.int32, copy=False)
        self.sample_ids = prepared.sample_ids
        self.library_sizes = self.counts.sum(axis=1).astype(np.float32)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        # The last three entries are labels expected (but not used) by CellFM's collator.
        return self.counts[index], self.gene_ids, self.library_sizes[index], 0, 0, 0.0


def load_cellfm_vocabulary(path: Path) -> list[str]:
    frame = pd.read_csv(path, index_col=0)
    genes = frame.index.astype(str).str.strip()
    if len(genes) == 0:
        raise DataError(f"CellFM vocabulary contains no genes: {path}")
    if genes.duplicated().any():
        raise DataError(f"CellFM vocabulary contains duplicate gene identifiers: {path}")
    return genes.tolist()


def _load_ensembl_symbol_map() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "data" / "gencode_v34_GRCh38_hg38.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, usecols=["gene_id", "gene_name"], dtype=str)
    frame = frame.dropna(subset=["gene_id", "gene_name"])
    frame["gene_id"] = frame["gene_id"].str.strip().str.split(".", n=1).str[0]
    frame["gene_name"] = frame["gene_name"].str.strip()
    frame = frame[(frame["gene_id"] != "") & (frame["gene_name"] != "")]
    return dict(zip(frame["gene_id"], frame["gene_name"]))


def _map_ensembl_ids_to_symbols(index: pd.Index, vocabulary: set[str]) -> tuple[pd.Index, int]:
    ensembl_to_symbol = _load_ensembl_symbol_map()
    if not ensembl_to_symbol:
        return index, 0

    mapped = []
    mapped_count = 0
    for gene in index.map(str):
        gene_key = gene.strip().split(".", 1)[0]
        symbol = ensembl_to_symbol.get(gene_key)
        if symbol in vocabulary:
            mapped.append(symbol)
            mapped_count += 1
        else:
            mapped.append(gene)
    return pd.Index(mapped), mapped_count


def prepare_cellfm_input(counts: pd.DataFrame, vocabulary: Iterable[str]) -> CellFMPreparedInput:
    if counts.empty:
        raise DataError("CellFM input count matrix is empty.")
    values = counts.copy()
    values.index = values.index.map(str)
    values.columns = values.columns.map(str)
    values = values.apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=np.float64)
    if not np.isfinite(array).all():
        raise DataError("CellFM input count matrix contains non-numeric or non-finite values.")
    if (array < 0).any():
        raise DataError("CellFM input count matrix contains negative counts.")

    # A duplicated input gene is one biological feature, so combine its counts before mapping.
    if values.index.duplicated().any():
        values = values.groupby(level=0, sort=False).sum()
        array = values.to_numpy(dtype=np.float64)

    genes = list(map(str, vocabulary))
    vocab_to_id = {gene: i + 1 for i, gene in enumerate(genes)}
    values.index, ensembl_mapped_genes = _map_ensembl_ids_to_symbols(
        values.index, set(vocab_to_id)
    )
    if values.index.duplicated().any():
        values = values.groupby(level=0, sort=False).sum()
        array = values.to_numpy(dtype=np.float64)

    mapped_mask = values.index.isin(vocab_to_id)
    mapped_genes = values.index[mapped_mask].tolist()
    unmapped_genes = values.index[~mapped_mask].tolist()
    if not mapped_genes:
        raise DataError("None of the input genes map exactly to the CellFM vocabulary.")

    # Match CellFM read_h5ad: cap each raw library at approximately 1e5 first.
    sample_by_gene = array.T
    raw_library_sizes = sample_by_gene.sum(axis=1)
    divisor = np.maximum(1.0, raw_library_sizes / 1e5)
    capped = np.rint(sample_by_gene / divisor[:, None]).astype(np.int32)
    mapped_counts = capped[:, np.asarray(mapped_mask)]
    if np.any(mapped_counts.sum(axis=1) <= 0):
        raise DataError("At least one sample has no non-zero counts mapped to CellFM.")

    return CellFMPreparedInput(
        counts=mapped_counts,
        gene_ids=np.asarray([vocab_to_id[g] for g in mapped_genes], dtype=np.int32),
        sample_ids=values.columns.tolist(),
        mapped_genes=mapped_genes,
        unmapped_genes=unmapped_genes,
        raw_library_sizes=raw_library_sizes,
        ensembl_mapped_genes=ensembl_mapped_genes,
    )


def cellfm_feature_matrix(embeddings: np.ndarray, sample_ids: Iterable[str]) -> pd.DataFrame:
    array = np.asarray(embeddings, dtype=np.float32)
    ids = list(map(str, sample_ids))
    if array.ndim != 2:
        raise DataError(f"CellFM sample embeddings must be 2D, got shape {array.shape}.")
    if array.shape[0] != len(ids):
        raise DataError(
            f"CellFM embedding/sample mismatch: embeddings={array.shape[0]}, samples={len(ids)}."
        )
    if not np.isfinite(array).all():
        raise DataError("CellFM extraction produced non-finite embeddings.")
    names = [f"cellfm_feature_{i:04d}" for i in range(array.shape[1])]
    return pd.DataFrame(array.T, index=names, columns=ids)


@contextmanager
def _cellfm_import_paths(root_dir: Path, vendor_dir: Path | None):
    root = str(root_dir.resolve())
    vendor = str(vendor_dir.resolve()) if vendor_dir is not None and vendor_dir.is_dir() else None
    inserted = []
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted.append(root)
    # Prefer installed, environment-compatible dependencies; bundled packages
    # remain a fallback for a dedicated CellFM environment.
    if vendor is not None and vendor not in sys.path:
        sys.path.append(vendor)
        inserted.append(vendor)
    try:
        yield
    finally:
        for path in inserted:
            if path in sys.path:
                sys.path.remove(path)


def _validate_resources(cfg: CellFMCfg) -> None:
    required = {
        "model": Path(cfg.root_dir) / "model.py",
        "utilities": Path(cfg.root_dir) / "layers" / "utils.py",
        "gene_info_path": Path(cfg.gene_info_path),
        "checkpoint_path": Path(cfg.checkpoint_path),
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise DataError("CellFM is enabled but required resources are missing: " + "; ".join(missing))


def _import_cellfm(cfg: CellFMCfg):
    # Scanpy/Numba and Matplotlib are imported by upstream CellFM even though the
    # extractor does not use them; give their caches a location available in jobs.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/omiclass_numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/omiclass_matplotlib")
    conflicting = {
        name: module for name, module in list(sys.modules.items())
        if name == "model" or name == "layers" or name.startswith("layers.")
    }
    for name in conflicting:
        sys.modules.pop(name, None)
    with _cellfm_import_paths(Path(cfg.root_dir), Path(cfg.vendor_dir) if cfg.vendor_dir else None):
        try:
            utils = importlib.import_module("layers.utils")
            model_module = importlib.import_module("model")
        except Exception as exc:
            raise DataError(
                "CellFM is enabled but its PyTorch/MindSpore dependencies are not importable."
            ) from exc
        finally:
            imported_names = {
                name for name in sys.modules
                if name == "model" or name == "layers" or name.startswith("layers.")
            }
            for name in imported_names:
                sys.modules.pop(name, None)
            sys.modules.update(conflicting)
    return utils, model_module


def _resolve_device(torch, configured: str):
    requested = str(configured or "auto").strip()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested.startswith("cuda") and not torch.cuda.is_available():
        raise DataError("CellFM device was set to CUDA, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _stable_view_seed(sample_id: str, global_seed: int, view_index: int) -> int:
    """Return the reference implementation's process-stable per-view seed."""
    payload = f"{sample_id}\0{int(global_seed)}\0{int(view_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _tokenize_sample_view(prep, capped_counts: np.ndarray, gene_ids: np.ndarray,
                          *, seed: int):
    """Run CellFM's existing collator operations for one deterministic view."""
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        data = np.asarray(capped_counts, dtype=np.float32)
        genes = np.asarray(gene_ids, dtype=np.int32)
        total = np.asarray(data.sum(), dtype=np.float32)
        raw_data, nonzero, zero = prep.seperate(data)
        data, nonzero, _cut, _zero_sample, seq_len = prep.sample(
            raw_data, nonzero, zero
        )
        raw_data, raw_nonzero, nonzero = prep.compress(raw_data, nonzero)
        _genes, selected_gene_ids, _ = prep.compress(genes, nonzero)
        raw_nonzero, normalized, selected_total, total = prep.bayes(raw_nonzero, total)
        normalized, selected_total = prep.normalize(normalized, selected_total)
        _raw_normalized, total = prep.normalize(raw_nonzero, total)
        st_features = prep.cat_st(selected_total, total)
        valid = prep.attn_mask(seq_len)
        normalized, _mask_gene = prep.mask(normalized)
        normalized = prep.pad_zero(normalized)
        selected_gene_ids = prep.pad_zero(selected_gene_ids)
    finally:
        np.random.set_state(state)

    return (
        np.asarray(normalized, dtype=np.float32),
        np.asarray(selected_gene_ids, dtype=np.int32),
        np.asarray(st_features, dtype=np.float32),
        np.asarray(valid, dtype=np.float32),
    )


def _pool_cellfm_batch(model, expression, genes, st_features, valid):
    encoded, _ = model.net.encode(expression, genes, st_features, valid)
    gene_tokens = encoded[:, 3:, :]
    mask = valid.unsqueeze(-1).to(gene_tokens.dtype)
    return (gene_tokens * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def _extract_legacy_single_view(*, torch, model, loader, device):
    """Preserve the original single pseudo-cell extraction implementation."""
    batches = []
    retained = []
    with torch.inference_mode():
        for batch in loader:
            expression = batch["dw_nzdata"].to(device)
            genes = batch["nonz_gene"].to(device)
            st_features = batch["ST_feat"].to(device)
            valid = batch["zero_idx"].to(device)
            pooled = _pool_cellfm_batch(model, expression, genes, st_features, valid)
            batches.append(pooled.float().cpu().numpy())
            retained.extend(valid.sum(1).to(dtype=torch.int64).cpu().tolist())
    if not batches:
        raise DataError("CellFM extraction produced no embeddings.")
    return np.concatenate(batches, axis=0), retained


def _extract_multiview(*, torch, model, prep, prepared: CellFMPreparedInput,
                       device, batch_size: int, random_state: int, num_views: int):
    """Generate deterministic CellFM views and return their float32 mean."""
    sample_count = len(prepared.sample_ids)
    embedding_sum = np.zeros(
        (sample_count, CELLFM_EMBEDDING_DIMENSION), dtype=np.float64
    )
    retained = np.zeros((sample_count, num_views), dtype=np.int32)

    for view_index in range(num_views):
        for batch_start in range(0, sample_count, batch_size):
            batch_stop = min(batch_start + batch_size, sample_count)
            expression_batch = []
            genes_batch = []
            st_batch = []
            valid_batch = []
            for sample_position in range(batch_start, batch_stop):
                seed = _stable_view_seed(
                    prepared.sample_ids[sample_position], random_state, view_index
                )
                expression, genes, st_features, valid = _tokenize_sample_view(
                    prep,
                    prepared.counts[sample_position],
                    prepared.gene_ids,
                    seed=seed,
                )
                expression_batch.append(expression)
                genes_batch.append(genes)
                st_batch.append(st_features)
                valid_batch.append(valid)
                retained[sample_position, view_index] = int(np.asarray(valid).sum())

            expression_tensor = torch.from_numpy(np.stack(expression_batch)).to(device)
            genes_tensor = torch.from_numpy(np.stack(genes_batch)).to(device)
            st_tensor = torch.from_numpy(np.stack(st_batch)).to(device)
            valid_tensor = torch.from_numpy(np.stack(valid_batch)).to(device)
            with torch.inference_mode():
                pooled = _pool_cellfm_batch(
                    model, expression_tensor, genes_tensor, st_tensor, valid_tensor
                )
            batch_embeddings = pooled.float().cpu().numpy()
            if batch_embeddings.shape != (
                batch_stop - batch_start, CELLFM_EMBEDDING_DIMENSION
            ):
                raise DataError(
                    "CellFM returned an unexpected embedding shape: "
                    f"{batch_embeddings.shape}."
                )
            embedding_sum[batch_start:batch_stop] += batch_embeddings

    return (embedding_sum / num_views).astype(np.float32), retained.tolist()


def _extract_configured_views(*, cfg: CellFMCfg, utils, torch, model, prep,
                              prepared: CellFMPreparedInput, device):
    """Route to the preserved legacy implementation or deterministic multiview."""
    num_views = int(cfg.num_views)
    if num_views == 1:
        loader = utils.build_dataset(
            _BulkSampleDataset(prepared), prep=prep, batch_size=int(cfg.batch_size),
            pad_zero=True, drop=False, shuffle=False,
        )
        state = np.random.get_state()
        np.random.seed(int(cfg.random_state))
        try:
            embeddings, retained = _extract_legacy_single_view(
                torch=torch, model=model, loader=loader, device=device
            )
        finally:
            np.random.set_state(state)
        return (
            embeddings,
            retained,
            "legacy_global_numpy_random_state",
            "single_view",
        )

    # Match the reference implementation's canonical token ordering.
    order = np.argsort(prepared.gene_ids, kind="stable")
    prepared.counts = prepared.counts[:, order]
    prepared.gene_ids = prepared.gene_ids[order]
    prepared.mapped_genes = [prepared.mapped_genes[int(i)] for i in order]
    embeddings, retained = _extract_multiview(
        torch=torch,
        model=model,
        prep=prep,
        prepared=prepared,
        device=device,
        batch_size=int(cfg.batch_size),
        random_state=int(cfg.random_state),
        num_views=num_views,
    )
    return (
        embeddings,
        retained,
        "sha256_sample_id_global_seed_zero_based_view_index",
        "mean",
    )


class CellFMFeatureExtractor:
    """Generate frozen, valid-gene mean-pooled embeddings for bulk RNA-seq samples."""

    def __init__(self, cfg: CellFMCfg):
        self.cfg = cfg

    def extract(self, counts: pd.DataFrame) -> CellFMExtractionResult:
        cfg = self.cfg
        _validate_resources(cfg)
        vocabulary = load_cellfm_vocabulary(Path(cfg.gene_info_path))
        prepared = prepare_cellfm_input(counts, vocabulary)
        utils, model_module = _import_cellfm(cfg)
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise DataError("Install torch in the CellFM environment to enable extraction.") from exc
        device = _resolve_device(torch, cfg.device)

        model_cfg = utils.Config_80M()
        model_cfg.ecs = True
        model_cfg.ecs_threshold = 0.8
        model_cfg.add_zero = True
        model_cfg.pad_zero = True
        model_cfg.mask_ratio = 0.0
        model_cfg.recompute = False
        model_cfg.device = str(device)
        model_cfg.ckpt_path = str(cfg.checkpoint_path)

        prep = utils.Prepare(
            pad_len=model_cfg.nonz_len, pad=0, mask_ratio=0.0, dw=False, random=False
        )
        if int(model_cfg.nonz_len) != CELLFM_GENES_PER_VIEW:
            raise DataError(
                f"Expected CellFM sequence length {CELLFM_GENES_PER_VIEW}, got "
                f"{model_cfg.nonz_len}."
            )
        if int(model_cfg.enc_dims) != CELLFM_EMBEDDING_DIMENSION:
            raise DataError(
                f"Expected CellFM embedding width {CELLFM_EMBEDDING_DIMENSION}, got "
                f"{model_cfg.enc_dims}."
            )
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(int(cfg.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.random_state))
        try:
            model = model_module.Cell_FM(
                n_gene=len(vocabulary), cfg=model_cfg,
                ckpt_path=str(cfg.checkpoint_path), device=str(device),
            ).to(device)
            model.load_model(weight=True, moment=False)
        finally:
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        model.eval()
        model.requires_grad_(False)

        num_views = int(cfg.num_views)
        embeddings, retained, seed_strategy, aggregation = _extract_configured_views(
            cfg=cfg,
            utils=utils,
            torch=torch,
            model=model,
            prep=prep,
            prepared=prepared,
            device=device,
        )
        retained_by_view = None
        if num_views == 1:
            retained_per_sample = list(map(int, retained))
        else:
            retained_by_view = [list(map(int, sample_views)) for sample_views in retained]
            # Preserve the original flat metadata field; selection length is the
            # same across views, while the full per-view values are recorded below.
            retained_per_sample = [sample_views[0] for sample_views in retained_by_view]
        features = cellfm_feature_matrix(embeddings, prepared.sample_ids)
        metadata = {
            "backend": "cellfm",
            "device": str(device),
            "batch_size": int(cfg.batch_size),
            "checkpoint_path": str(cfg.checkpoint_path),
            "gene_info_path": str(cfg.gene_info_path),
            "pooling": "valid_gene_mean",
            "selection_random_state": int(cfg.random_state),
            "num_views": num_views,
            "view_seed_strategy": seed_strategy,
            "view_aggregation": aggregation,
            "input_shape": list(counts.shape),
            "mapped_genes": len(prepared.mapped_genes),
            "unmapped_genes": len(prepared.unmapped_genes),
            "ensembl_mapped_genes": int(prepared.ensembl_mapped_genes),
            "vocabulary_size": len(vocabulary),
            "sequence_length": int(model_cfg.nonz_len),
            "retained_genes_per_sample": retained_per_sample,
            "embedding_shape": list(embeddings.shape),
            "feature_shape": list(features.shape),
        }
        if retained_by_view is not None:
            metadata["retained_genes_per_sample_per_view"] = retained_by_view
        return CellFMExtractionResult(features=features, metadata=metadata)


def write_cellfm_artifacts(*, out_dir: Path, features: pd.DataFrame,
                           metadata: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "cellfm_features.csv"
    metadata_path = out_dir / "cellfm_metadata.json"
    features.to_csv(feature_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return feature_path, metadata_path
