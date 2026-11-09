from __future__ import annotations

import importlib
import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from omiclass.config.schema import BulkFormerCfg
from omiclass.errors.errors import DataError


@dataclass
class BulkFormerPreparedInput:
    input_df: pd.DataFrame
    mask_prob: float
    missing_genes: list[str]
    gene_list: list[str]


@dataclass
class BulkFormerExtractionResult:
    features: pd.DataFrame
    metadata: dict[str, Any]


def load_bulkformer_gene_list(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "ensg_id" not in df.columns:
        raise DataError(f"BulkFormer gene info must contain an 'ensg_id' column: {path}")
    genes = df["ensg_id"].dropna().astype(str).str.strip()
    genes = genes[genes != ""]
    genes = genes[~genes.duplicated(keep="first")]
    if genes.empty:
        raise DataError(f"BulkFormer gene info contains no usable genes: {path}")
    return genes.tolist()


def load_bulkformer_gene_lengths(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if {"ensg_id", "length"}.issubset(df.columns):
        ids = df["ensg_id"].astype(str)
        values = pd.to_numeric(df["length"], errors="coerce")
    elif df.shape[1] >= 2:
        ids = df.iloc[:, 0].astype(str)
        values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        raise DataError(f"BulkFormer gene length file must have gene IDs and lengths: {path}")

    out = {
        gene: float(length)
        for gene, length in zip(ids, values)
        if gene and np.isfinite(length) and float(length) > 0
    }
    if not out:
        raise DataError(f"BulkFormer gene length file contains no usable lengths: {path}")
    return out


def normalize_counts_to_log_tpm(
    counts: pd.DataFrame,
    gene_lengths: dict[str, float],
    *,
    default_gene_length: float = 1000.0,
) -> pd.DataFrame:
    if counts.empty:
        raise DataError("BulkFormer input count matrix is empty.")
    if default_gene_length <= 0:
        raise DataError("BulkFormer default gene length must be positive.")

    counts = counts.copy()
    counts.index = counts.index.map(str)
    counts.columns = counts.columns.map(str)
    counts = counts.apply(pd.to_numeric, errors="coerce")
    if counts.isna().any().any():
        raise DataError("BulkFormer input count matrix contains non-numeric values.")
    if (counts < 0).any().any():
        raise DataError("BulkFormer input count matrix contains negative counts.")

    gene_lengths_kb = np.array(
        [float(gene_lengths.get(str(gene), default_gene_length)) / 1000.0 for gene in counts.index],
        dtype=np.float64,
    )
    rates = counts.to_numpy(dtype=np.float64) / gene_lengths_kb[:, None]
    per_sample = rates.sum(axis=0)
    per_sample[per_sample == 0] = 1e-6
    tpm = rates / per_sample[None, :] * 1e6
    log_tpm = np.log1p(tpm)
    return pd.DataFrame(log_tpm, index=counts.index, columns=counts.columns)


def align_to_bulkformer_vocabulary(
    log_tpm: pd.DataFrame,
    bulkformer_gene_list: Sequence[str],
    *,
    fill_value: float = -10.0,
) -> BulkFormerPreparedInput:
    if log_tpm.empty:
        raise DataError("BulkFormer normalized expression matrix is empty.")
    if not bulkformer_gene_list:
        raise DataError("BulkFormer gene vocabulary is empty.")

    log_tpm = log_tpm.copy()
    log_tpm.index = log_tpm.index.map(str)
    log_tpm.columns = log_tpm.columns.map(str)
    wanted = [str(g) for g in bulkformer_gene_list]
    missing = [gene for gene in wanted if gene not in log_tpm.index]

    sample_by_gene = log_tpm.T.reindex(columns=wanted, fill_value=fill_value)
    sample_by_gene = sample_by_gene.astype("float32")
    mask_prob = len(missing) / len(wanted)
    return BulkFormerPreparedInput(
        input_df=sample_by_gene,
        mask_prob=float(mask_prob),
        missing_genes=missing,
        gene_list=wanted,
    )


def feature_matrix_from_embeddings(
    embeddings: np.ndarray,
    sample_ids: Iterable[str],
) -> pd.DataFrame:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise DataError(f"BulkFormer sample embeddings must be 2D, got shape {arr.shape}.")
    sample_ids = [str(s) for s in sample_ids]
    if arr.shape[0] != len(sample_ids):
        raise DataError(
            "BulkFormer embedding/sample mismatch: "
            f"embeddings={arr.shape[0]}, samples={len(sample_ids)}."
        )
    feature_names = [f"bulkformer_feature_{i:04d}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr.T, index=feature_names, columns=sample_ids)


def prepare_bulkformer_input_from_counts(counts: pd.DataFrame, cfg: BulkFormerCfg) -> BulkFormerPreparedInput:
    gene_lengths = load_bulkformer_gene_lengths(Path(cfg.gene_lengths_path))
    gene_list = load_bulkformer_gene_list(Path(cfg.gene_info_path))
    log_tpm = normalize_counts_to_log_tpm(counts, gene_lengths)
    return align_to_bulkformer_vocabulary(log_tpm, gene_list)


def write_bulkformer_artifacts(
    *,
    out_dir: Path,
    features: pd.DataFrame,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "bulkformer_features.csv"
    metadata_path = out_dir / "bulkformer_metadata.json"
    features.to_csv(feature_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return feature_path, metadata_path


def _validate_bulkformer_resources(cfg: BulkFormerCfg) -> None:
    required = {
        "root_dir": cfg.root_dir,
        "graph_path": cfg.graph_path,
        "graph_weights_path": cfg.graph_weights_path,
        "gene_embedding_path": cfg.gene_embedding_path,
        "checkpoint_path": cfg.checkpoint_path,
        "gene_info_path": cfg.gene_info_path,
        "gene_lengths_path": cfg.gene_lengths_path,
        "interested_gene_path": cfg.interested_gene_path,
    }
    missing = [f"{name}={path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise DataError("BulkFormer is enabled but required resources are missing: " + "; ".join(missing))


def _resolve_device(device: str):
    torch = _import_required("torch", "Install torch in the BulkFormer environment to enable extraction.")
    requested = str(device or "auto").strip()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested.startswith("cuda") and not torch.cuda.is_available():
        raise DataError("BulkFormer device was set to CUDA, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _import_required(module_name: str, hint: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DataError(f"BulkFormer is enabled but '{module_name}' is not importable. {hint}") from exc


@contextmanager
def _bulkformer_import_path(root_dir: Path):
    root = str(Path(root_dir).resolve())
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _load_bulkformer_model(cfg: BulkFormerCfg, device):
    torch = _import_required("torch", "Install torch in the BulkFormer environment to enable extraction.")
    torch_geometric_typing = _import_required(
        "torch_geometric.typing",
        "Install torch-geometric and its sparse tensor dependencies for BulkFormer.",
    )
    sparse_tensor_cls = getattr(torch_geometric_typing, "SparseTensor", None)
    if sparse_tensor_cls is None:
        raise DataError("BulkFormer requires torch_geometric.typing.SparseTensor.")

    with _bulkformer_import_path(Path(cfg.root_dir)):
        bulkformer_module = importlib.import_module("utils.BulkFormer")
        model_config = importlib.import_module("model.config")

    graph = torch.load(Path(cfg.graph_path), map_location="cpu", weights_only=False)
    weights = torch.load(Path(cfg.graph_weights_path), map_location="cpu", weights_only=False)
    graph = sparse_tensor_cls(row=graph[1], col=graph[0], value=weights).t().to(device)
    gene_emb = torch.load(Path(cfg.gene_embedding_path), map_location="cpu", weights_only=False)

    params = dict(getattr(model_config, "model_params"))
    params["graph"] = graph
    params["gene_emb"] = gene_emb

    model = bulkformer_module.BulkFormer(**params).to(device)
    ckpt_model = torch.load(Path(cfg.checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        state_dict[key[7:] if key.startswith("module.") else key] = value
    model.load_state_dict(state_dict)
    model.eval()
    return model, params


def _to_numpy_indices(index_obj: Any) -> np.ndarray | None:
    if index_obj is None:
        return None
    if hasattr(index_obj, "detach"):
        index_obj = index_obj.detach().cpu().numpy()
    arr = np.asarray(index_obj)
    if arr.size == 0:
        return None
    return arr.astype(int)


def _extract_sample_embeddings(
    *,
    model: Any,
    expr_array: np.ndarray,
    interested_gene_idx: np.ndarray | None,
    aggregate_type: str,
    device: Any,
    batch_size: int,
    mask_prob: float,
) -> np.ndarray:
    torch = _import_required("torch", "Install torch in the BulkFormer environment to enable extraction.")
    data_mod = _import_required("torch.utils.data", "Install torch to enable BulkFormer data loading.")
    tensor_dataset = getattr(data_mod, "TensorDataset")
    data_loader = getattr(data_mod, "DataLoader")

    expr_tensor = torch.tensor(expr_array, dtype=torch.float32)
    loader = data_loader(tensor_dataset(expr_tensor), batch_size=batch_size, shuffle=False)
    all_emb = []
    use_cuda = getattr(device, "type", str(device)) == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda):
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            gene_emb = model(batch_x, mask_prob=mask_prob, output_expr=False)
            gene_emb = gene_emb.detach().cpu().numpy()
            if interested_gene_idx is not None:
                gene_emb = gene_emb[:, interested_gene_idx, :]

            if aggregate_type == "max":
                final_emb = np.max(gene_emb, axis=1)
            elif aggregate_type == "mean":
                final_emb = np.mean(gene_emb, axis=1)
            elif aggregate_type == "median":
                final_emb = np.median(gene_emb, axis=1)
            elif aggregate_type == "all":
                final_emb = (
                    np.max(gene_emb, axis=1)
                    + np.mean(gene_emb, axis=1)
                    + np.median(gene_emb, axis=1)
                )
            else:
                raise DataError(f"Unsupported BulkFormer aggregate_type: {aggregate_type}")

            all_emb.append(final_emb)
            if use_cuda:
                torch.cuda.empty_cache()

    if not all_emb:
        raise DataError("BulkFormer extraction produced no embeddings.")
    return np.vstack(all_emb).astype(np.float32)


class BulkFormerFeatureExtractor:
    def __init__(self, cfg: BulkFormerCfg, *, observer=None, continuous=False) -> None:
        self.cfg = cfg
        self.observer = observer
        self.continuous = continuous

    def extract(self, counts: pd.DataFrame) -> BulkFormerExtractionResult:
        from omiclass.utils.reproducibility import random_state_scope
        with random_state_scope(cuda=self.cfg.device != 'cpu'):
            return self._extract(counts)

    def _extract(self, counts: pd.DataFrame) -> BulkFormerExtractionResult:
        cfg = self.cfg
        _validate_bulkformer_resources(cfg)
        prepared = (align_to_bulkformer_vocabulary(counts, load_bulkformer_gene_list(cfg.gene_info_path))
                    if self.continuous else prepare_bulkformer_input_from_counts(counts, cfg))

        torch = _import_required("torch", "Install torch in the BulkFormer environment to enable extraction.")
        device = _resolve_device(cfg.device)
        model, params = _load_bulkformer_model(cfg, device)
        interested_gene_idx = _to_numpy_indices(
            torch.load(Path(cfg.interested_gene_path), map_location="cpu", weights_only=False)
        )
        forward = model
        if self.observer is not None:
            position = 0
            gene_ids = np.asarray(prepared.gene_list)
            if interested_gene_idx is not None:
                gene_ids = gene_ids[interested_gene_idx]
            def forward(*args, **kwargs):
                nonlocal position
                encoded = model(*args, **kwargs)
                tokens = encoded if interested_gene_idx is None else encoded[:, interested_gene_idx, :]
                size = len(tokens)
                self.observer(tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device),
                              np.tile(gene_ids, (size, 1)), prepared.input_df.index[position:position+size], 0)
                position += size
                return encoded
        embeddings = _extract_sample_embeddings(
            model=forward,
            expr_array=prepared.input_df.to_numpy(dtype=np.float32),
            interested_gene_idx=interested_gene_idx,
            aggregate_type=str(cfg.aggregate_type),
            device=device,
            batch_size=int(cfg.batch_size),
            mask_prob=float(prepared.mask_prob),
        )
        features = feature_matrix_from_embeddings(embeddings, prepared.input_df.index)
        metadata = {
            "device": str(device),
            "batch_size": int(cfg.batch_size),
            "aggregate_type": str(cfg.aggregate_type),
            "checkpoint_path": str(cfg.checkpoint_path),
            "input_shape": list(counts.shape),
            "aligned_shape": list(prepared.input_df.shape),
            "embedding_shape": list(embeddings.shape),
            "feature_shape": list(features.shape),
            "mask_prob": float(prepared.mask_prob),
            "missing_vocabulary_genes": len(prepared.missing_genes),
            "model_dim": int(params.get("dim", embeddings.shape[1])),
            "gene_length": int(params.get("gene_length", len(prepared.gene_list))),
        }
        return BulkFormerExtractionResult(features=features, metadata=metadata)
