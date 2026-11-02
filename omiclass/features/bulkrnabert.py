"""BulkRNABert adapter. This module deliberately never imports JAX or Haiku."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from omiclass.errors.errors import DataError

N_GENES = 19062
EMBED_DIM = 256
MIN_COMMON_GENES = 18000
CACHE_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ids(ids, description):
    ids = list(map(str, ids))
    if not ids or any(not x.strip() or x != x.strip() for x in ids):
        raise DataError(f'BulkRNABert: empty or whitespace {description}.')
    if len(ids) != len(set(ids)):
        raise DataError(f'BulkRNABert: duplicate {description}.')
    return ids


def normalize_gene_ids(ids):
    return validate_ids([re.sub(r'\.\d+$', '', str(x)) for x in ids], 'gene IDs after version removal')


def read_expression_csv(path: Path) -> pd.DataFrame:
    """Strict genes x samples CSV reader, before pandas can mangle duplicate headers."""
    with Path(path).open(newline='') as handle:
        header = next(csv.reader(handle), [])
    samples = validate_ids(header[1:], 'sample IDs')
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] != len(samples) or df.index.isna().any():
        raise DataError('BulkRNABert: malformed expression matrix.')
    df.columns = samples
    df.index = normalize_gene_ids(df.index)
    return validate_expression(df)


def validate_expression(df: pd.DataFrame) -> pd.DataFrame:
    validate_ids(df.columns, 'sample IDs')
    validate_ids(df.index, 'gene IDs')
    try:
        values = df.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError('BulkRNABert expression must be numeric.') from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataError('BulkRNABert requires finite, nonnegative, unlogged expression.')
    if (values.sum(axis=0) <= 0).any():
        raise DataError('BulkRNABert: empty-expression sample.')
    return pd.DataFrame(values, index=df.index, columns=df.columns)


def prepare_tpm(tpm: pd.DataFrame, vocabulary, sample_ids):
    """No data-dependent fitting: align each sample to the fixed official vocabulary."""
    vocabulary = validate_ids(vocabulary, 'vocabulary IDs')
    if len(vocabulary) != N_GENES:
        raise DataError(f'BulkRNABert vocabulary must contain {N_GENES} genes.')
    sample_ids = validate_ids(sample_ids, 'requested sample IDs')
    tpm = tpm.copy()
    tpm.index = normalize_gene_ids(tpm.index)
    tpm.columns = validate_ids(tpm.columns, 'sample IDs')
    tpm = validate_expression(tpm)
    missing_samples = sorted(set(sample_ids) - set(tpm.columns))
    if missing_samples:
        raise DataError(f'BulkRNABert TPM missing samples: {missing_samples[:10]}')
    present = set(tpm.index)
    missing_genes = [g for g in vocabulary if g not in present]
    common = len(vocabulary) - len(missing_genes)
    if common < MIN_COMMON_GENES:
        raise DataError(f'BulkRNABert vocabulary coverage {common} < {MIN_COMMON_GENES}.')
    aligned = tpm.loc[:, sample_ids].reindex(vocabulary, fill_value=0.0)
    aligned = validate_expression(aligned)
    values = np.ascontiguousarray(aligned.T.to_numpy(dtype=np.float32))
    if not np.isfinite(values).all():
        raise DataError('BulkRNABert TPM exceeds float32 range.')
    return values, {
        'matched_genes': common, 'missing_genes': missing_genes,
        'unmapped_genes': sorted(present - set(vocabulary)),
        'sample_ids': sample_ids, 'input_scale': 'TPM',
    }


def feature_matrix(embeddings, sample_ids):
    sample_ids = validate_ids(sample_ids, 'embedding sample IDs')
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (len(sample_ids), EMBED_DIM) or not np.isfinite(values).all():
        raise DataError(f'Invalid BulkRNABert embeddings: {values.shape}; expected ({len(sample_ids)}, 256), finite.')
    return pd.DataFrame(values.T, index=[f'bulkrnabert_feature_{i:04d}' for i in range(EMBED_DIM)], columns=sample_ids)


@dataclass
class BulkRNABertExtractionResult:
    features: pd.DataFrame
    metadata: dict


class BulkRNABertFeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg

    def _run_worker(self, values, sample_ids, workdir):
        cfg = self.cfg
        input_path = workdir / 'input.npz'
        output_path = workdir / 'output.npz'
        np.savez(input_path, tpm=values, sample_ids=np.asarray(sample_ids, dtype=str))
        command = [str(Path(cfg.python_executable).expanduser().resolve()),
                   str(Path(__file__).with_name('bulkrnabert_worker.py').resolve()),
                   '--root-dir', str(Path(cfg.root_dir).resolve()),
                   '--checkpoint-directory', str(Path(cfg.checkpoint_directory).resolve()),
                   '--model-name', cfg.model_name, '--input', str(input_path),
                   '--output', str(output_path), '--batch-size', str(cfg.batch_size),
                   '--device', cfg.device]
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['PYTHONNOUSERSITE'] = '1'
        env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        print('[bulkrnabert] Extracting frozen embeddings in separate Python environment.', flush=True)
        try:
            subprocess.run(command, env=env, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DataError(f'BulkRNABert worker failed; check python_executable, JAX device and checkpoint: {exc}') from exc
        with np.load(output_path, allow_pickle=False) as result:
            if result['sample_ids'].tolist() != sample_ids:
                raise DataError('BulkRNABert worker sample-ID mismatch.')
            embeddings = result['embeddings'].copy()
            worker_metadata = json.loads(str(result['metadata']))
        feature_matrix(embeddings, sample_ids)
        return embeddings, worker_metadata

    def extract(self, counts: pd.DataFrame) -> BulkRNABertExtractionResult:
        # The existing matrix supplies ONLY sample order. Its values are never consumed.
        cfg = self.cfg
        if cfg.tpm_input_path is None:
            raise DataError('BulkRNABert requires tpm_input_path, never processed Omiclass expression.')
        sample_ids = validate_ids(counts.columns, 'requested sample IDs')
        tpm_path = Path(cfg.tpm_input_path).resolve()
        vocabulary = Path(cfg.vocabulary_path).read_text().splitlines()
        values, mapping = prepare_tpm(read_expression_csv(tpm_path), vocabulary, sample_ids)
        checkpoint = Path(cfg.checkpoint_directory) / cfg.model_name
        source_files = sorted((Path(cfg.root_dir) / 'multiomics_open_research').rglob('*.py'))
        if not source_files:
            raise DataError('BulkRNABert official implementation not found under root_dir.')
        signature = {
            'cache_version': CACHE_VERSION, 'model_name': cfg.model_name,
            'config_sha256': sha256_file(checkpoint / 'config.json'),
            'weights_sha256': sha256_file(checkpoint / 'params.joblib'),
            'vocabulary_sha256': sha256_file(Path(cfg.vocabulary_path)),
            'implementation': {str(p.relative_to(cfg.root_dir)): sha256_file(p) for p in source_files},
            'worker_sha256': sha256_file(Path(__file__).with_name('bulkrnabert_worker.py')),
            'expression_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
            'sample_ids': sample_ids, 'device': cfg.device, 'batch_size': cfg.batch_size,
            'pooling': 'embeddings_4_gene_mean', 'preprocessing': 'official_log10_TPM_then_tokenizer',
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        cache_dir = Path(cfg.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f'{key}.npz'
        hit = cache_path.exists()
        if hit:
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    saved = json.loads(str(cached['metadata']))
                    if saved['signature'] != signature or cached['sample_ids'].tolist() != sample_ids:
                        raise ValueError('signature/sample IDs mismatch')
                    embeddings = cached['embeddings'].copy()
                    if saved['embeddings_sha256'] != hashlib.sha256(embeddings.tobytes()).hexdigest():
                        raise ValueError('embedding checksum mismatch')
                    worker_metadata = saved['worker']
            except Exception as exc:
                raise DataError(f'Invalid BulkRNABert cache {cache_path}: {exc}; remove this cache and retry.') from exc
        else:
            with tempfile.TemporaryDirectory(prefix='extract-', dir=cache_dir) as tmp:
                workdir = Path(tmp)
                embeddings, worker_metadata = self._run_worker(values, sample_ids, workdir)
                saved = {'signature': signature, 'worker': worker_metadata,
                         'embeddings_sha256': hashlib.sha256(embeddings.tobytes()).hexdigest()}
                pending = workdir / 'cache.npz'
                np.savez_compressed(pending, embeddings=embeddings,
                                    sample_ids=np.asarray(sample_ids, dtype=str),
                                    metadata=json.dumps(saved, sort_keys=True))
                os.replace(pending, cache_path)
        features = feature_matrix(embeddings, sample_ids)
        metadata = {'backend': 'bulkrnabert', **mapping, 'signature': signature,
                    'worker': worker_metadata, 'cache_path': str(cache_path), 'cache_hit': hit,
                    'tpm_input_path': str(tpm_path), 'tpm_file_sha256': sha256_file(tpm_path),
                    'feature_shape': list(features.shape)}
        provenance = tpm_path.with_suffix(tpm_path.suffix + '.json')
        if provenance.exists():
            metadata['input_provenance'] = json.loads(provenance.read_text())
        return BulkRNABertExtractionResult(features, metadata)


def write_bulkrnabert_artifacts(*, out_dir, features, metadata):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / 'bulkrnabert_features.csv'
    metadata_path = out_dir / 'bulkrnabert_metadata.json'
    features.to_csv(feature_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return feature_path, metadata_path
