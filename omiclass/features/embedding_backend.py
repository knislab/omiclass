from __future__ import annotations

from typing import Any


def embedding_backend(cfg: Any) -> str | None:
    """Return the single enabled sample-embedding backend."""
    if bool(getattr(getattr(cfg, "bulkformer", None), "enabled", False)):
        return "bulkformer"
    if bool(getattr(getattr(cfg, "cellfm", None), "enabled", False)):
        return "cellfm"
    return None


def embedding_enabled(cfg: Any) -> bool:
    return embedding_backend(cfg) is not None


def embedding_config(cfg: Any):
    backend = embedding_backend(cfg)
    return getattr(cfg, backend) if backend is not None else None


def extraction_signature(backend: str, cfg: Any) -> dict[str, str]:
    if backend == "bulkformer":
        return {
            "aggregate_type": str(cfg.aggregate_type),
            "checkpoint_path": str(cfg.checkpoint_path),
            "gene_info_path": str(cfg.gene_info_path),
            "interested_gene_path": str(cfg.interested_gene_path),
        }
    if backend == "cellfm":
        return {
            "backend": backend,
            "pooling": "valid_gene_mean",
            "checkpoint_path": str(cfg.checkpoint_path),
            "gene_info_path": str(cfg.gene_info_path),
            "random_state": str(cfg.random_state),
        }
    raise ValueError(f"Unsupported embedding backend: {backend}")
