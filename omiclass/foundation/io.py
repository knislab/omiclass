"""Run-owned arrays and token records; no implicit manuscript paths."""
from contextlib import contextmanager
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False) + "\n")


def save_frame(path, frame):
    np.savez(path, values=frame.to_numpy(), index=np.asarray(frame.index, dtype=str), columns=np.asarray(frame.columns, dtype=str))


def load_frame(path):
    with np.load(path, allow_pickle=False) as z:
        return pd.DataFrame(z["values"], index=z["index"].astype(str), columns=z["columns"].astype(str))


def gene_order_digest(ids):
    return hashlib.sha256(json.dumps(np.asarray(ids).tolist(), ensure_ascii=True,
                                    separators=(',', ':')).encode()).hexdigest()


def token_schema(backend, shape, gene_ids=None):
    schema = dict(version=1, backend=backend, token_count=int(shape[0]), embedding_width=int(shape[1]),
                  identity_mode='sampled_positions' if backend == 'cellfm' else 'fixed_gene_order')
    if backend != 'cellfm':
        if gene_ids is None:
            raise ValueError('Fixed-position token schema requires gene identities')
        schema['gene_order_sha256'] = gene_order_digest(gene_ids)
    else:
        schema['position_interpretation'] = 'sampled positions; gene identities may vary across samples/views'
    return schema


def validate_tokens(tokens, mask, gene_ids=None, *, schema=None):
    x, raw_mask = np.asarray(tokens), np.asarray(mask)
    if (x.ndim != 2 or raw_mask.shape != (len(x),) or not np.isin(raw_mask, [0, 1]).all()
            or not raw_mask.any() or not np.isfinite(x).all()):
        raise ValueError('Token shape, finite values or binary valid mask mismatch')
    valid = raw_mask.astype(bool)
    if gene_ids is not None and np.asarray(gene_ids).shape != (len(x),):
        raise ValueError('Token gene identities must match every token position')
    if schema is not None:
        if (schema.get('version') != 1 or x.shape != (schema['token_count'], schema['embedding_width'])
                or gene_ids is None):
            raise ValueError('Token schema or gene identity metadata mismatch')
        genes = np.asarray(gene_ids)
        if schema['identity_mode'] == 'sampled_positions' and schema['backend'] == 'cellfm':
            if (genes.dtype.kind not in 'iu' or (genes[valid] <= 0).any()
                    or (genes[~valid] != 0).any() or len(np.unique(genes[valid])) != int(valid.sum())
                    or not np.array_equal(valid, np.arange(len(valid)) < valid.sum())):
                raise ValueError('CellFM gene IDs, padding or valid-mask ordering mismatch')
            if (genes[valid] > schema.get('vocabulary_size', np.iinfo(np.int64).max)).any():
                raise ValueError('CellFM gene ID exceeds its vocabulary')
            if schema.get('sampling_strategy') == 'sha256_sample_id_global_seed_zero_based_view_index' and (np.diff(genes[valid].astype(np.int64)) <= 0).any():
                raise ValueError('CellFM selected token order differs from canonical sampling order')
        elif schema['identity_mode'] == 'fixed_gene_order':
            if not valid.all() or gene_order_digest(genes) != schema.get('gene_order_sha256'):
                raise ValueError('Fixed token gene order or mask changed')
        else:
            raise ValueError('Unsupported token identity protocol')
    return x.astype(np.float32, copy=True), valid.copy()


def load_token_manifest(cache, *, expected_schema=None):
    manifest = json.loads((Path(cache) / 'manifest.json').read_text())
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported cache format')
    views = manifest.get('metadata', {}).get('num_views')
    if not isinstance(views, int) or views < 1:
        raise ValueError('Token cache requires a positive number of views')
    schema = manifest.get('metadata', {}).get('token_schema')
    if expected_schema is not None and schema != expected_schema:
        raise ValueError('Prediction token protocol differs from the saved model')
    if schema is not None and schema.get('backend') != manifest.get('backend'):
        raise ValueError('Token schema backend mismatch')
    if any(type(row['view']) is not int for row in manifest['records']):
        raise ValueError('Token view indices must be integers')
    coordinates = [(str(row['sample_id']), row['view']) for row in manifest['records']]
    if len(coordinates) != len(set(coordinates)) or any(not 0 <= view < views for _, view in coordinates):
        raise ValueError('Duplicated or invalid token sample/view record')
    return manifest


def load_token_arrays(cache, record, *, schema=None):
    with token_record(cache, record) as values:
        return validate_tokens(values['tokens'], values['mask'],
            values['gene_ids'] if 'gene_ids' in values else None, schema=schema)


@contextmanager
def token_record(cache, record):
    cache = Path(cache).resolve()
    path = (cache / record["file"]).resolve()
    if not path.is_relative_to(cache):
        raise ValueError("Token record escapes its cache")
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            values = {"tokens": array, "mask": np.ones(array.shape[0], dtype=bool)}
            if 'gene_ids_file' in record:
                gene_path = (cache / record['gene_ids_file']).resolve()
                if not gene_path.is_relative_to(cache):
                    raise ValueError('Token gene identity file escapes its cache')
                values['gene_ids'] = np.load(gene_path, allow_pickle=False)
            yield values
        finally:
            del array
    else:
        with np.load(path, allow_pickle=False) as z:
            yield z
