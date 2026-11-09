"""Read-only identities for encoder assets; no cache or checkpoint mutation."""
import hashlib
from pathlib import Path


def file_identity(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return dict(path=str(path), absent=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def source_identity(root, *, vendor=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return dict(path=str(root), absent=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.relative_to(root).parts):
            continue
        name = path.name.lower()
        if path.is_file() and (path.suffix == ".py" or
                (vendor and (name.endswith((".so", ".pyd", ".dll", ".dylib")) or ".so." in name))):
            files[str(path.relative_to(root))] = file_identity(path)
    return dict(path=str(root), files=files)


def encoder_identities(cfg):
    backend = "cellfm" if cfg.foundation.model.startswith("cellfm") else cfg.foundation.model
    encoder = getattr(cfg, backend)
    result = {}
    for name, value in encoder:
        if not isinstance(value, Path) or name in {"cache_dir", "tpm_input_path", "root_dir", "vendor_dir", "checkpoint_directory"}:
            continue
        result[name] = file_identity(value)
    result["implementation"] = source_identity(encoder.root_dir)
    if backend == "cellfm":
        if encoder.vendor_dir is not None:
            result["vendor_implementation"] = source_identity(encoder.vendor_dir, vendor=True)
        result["gene_symbol_mapping"] = file_identity(
            Path(__file__).resolve().parents[1] / "data/gencode_v34_GRCh38_hg38.csv")
    if backend == "bulkrnabert":
        checkpoint = encoder.checkpoint_directory / encoder.model_name
        result["checkpoint_config"] = file_identity(checkpoint / "config.json")
        result["checkpoint_weights"] = file_identity(checkpoint / "params.joblib")
    if cfg.normalization.gene_lengths_path is not None:
        result["normalization_gene_lengths"] = file_identity(cfg.normalization.gene_lengths_path)
    return result
