"""Standalone worker: execute with BulkRNABert's Python, not Omiclass's Python."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    parser.add_argument('--model-name', default='bulk_rna_bert_gtex_encode')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--input-scale', choices=['tpm', 'log10_tpm'], default='tpm')
    parser.add_argument('--token-dir', type=Path)
    parser.add_argument('--shape-only', action='store_true', help='Load weights and trace output shape without allocating attention tensors.')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    sys.path.insert(0, str(args.root_dir.resolve()))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if args.device != 'auto':
        os.environ['JAX_PLATFORMS'] = args.device

    import numpy as np
    import pandas as pd
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from multiomics_open_research.bulk_rna_bert.pretrained import get_bulkrnabert_pretrained_model
    from multiomics_open_research.common.preprocess import preprocess_omic

    parameters, forward_fn, tokenizer, config = get_bulkrnabert_pretrained_model(
        args.model_name, embeddings_layers_to_save=(4,),
        checkpoint_directory=str(args.checkpoint_directory.resolve()),
    )
    if (config.n_genes, config.embed_dim, config.num_layers) != (19062, 256, 4):
        raise ValueError('Checkpoint must have n_genes=19062, embed_dim=256, num_layers=4.')
    if not config.use_log_normalization or not config.use_max_normalization:
        raise ValueError('Checkpoint must enable official log and max normalization.')
    with np.load(args.input, allow_pickle=False) as bundle:
        tpm = bundle['tpm'].copy()
        ids = bundle['sample_ids'].copy()
    if tpm.shape != (len(ids), 19062) or not len(ids) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid worker input shape or sample IDs.')
    if not np.isfinite(tpm).all() or (tpm < 0).any() or (tpm.sum(axis=1) <= 0).any():
        raise ValueError('Invalid worker TPM values.')
    forward = hk.transform(forward_fn)
    key = jax.random.PRNGKey(0)

    def pooled(tokens):
        result = forward.apply(parameters, key, tokens)['embeddings_4']
        if result.shape != (tokens.shape[0], 19062, 256):
            raise ValueError(f'Unexpected embeddings_4 shape: {result.shape}')
        return (result, result.mean(axis=1)) if args.token_dir is not None else result.mean(axis=1)

    infer = jax.jit(pooled)
    embeddings = []
    records = []
    if args.token_dir is not None:
        args.token_dir.mkdir(parents=True, exist_ok=False)
    for start in range(0, len(ids), args.batch_size):
        batch = tpm[start:start + args.batch_size].copy()
        normalized = preprocess_omic(pd.DataFrame(batch), config) if args.input_scale == "tpm" else batch
        tokens = tokenizer.batch_tokenize(normalized)
        if tokens.min() < 0 or tokens.max() >= config.n_expressions_bins:
            raise ValueError('TPM exceeds checkpoint tokenizer bins; do not clip silently. Check input scale/checkpoint.')
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        if args.shape_only:
            shape = jax.eval_shape(lambda x: forward.apply(parameters, key, x)['embeddings_4'], tokens).shape
            print(json.dumps({'embeddings_4_shape': list(shape), 'jax': jax.__version__}), flush=True)
            return
        inferred = infer(tokens)
        if args.token_dir is not None:
            contextual, pooled_values = inferred
            for offset, values in enumerate(np.asarray(contextual, dtype=np.float32)):
                filename = f"record_{len(records):08d}.npy"
                np.save(args.token_dir / filename, values, allow_pickle=False)
                records.append(dict(sample_id=str(ids[start+offset]), view=0, file=filename))
            inferred = pooled_values
        embeddings.append(np.asarray(inferred, dtype=np.float32))
        print(f'[bulkrnabert] Extracted {min(start + args.batch_size, len(ids))}/{len(ids)} samples', flush=True)
    embeddings = np.concatenate(embeddings)
    if embeddings.shape != (len(ids), 256) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid pooled embeddings.')
    metadata = {'embeddings_4_shape': [min(args.batch_size, len(ids)), 19062, 256],
                'pool_axis': 1, 'normalization_factor': float(config.normalization_factor),
                'jax': jax.__version__, 'haiku': hk.__version__, 'numpy': np.__version__,
                'devices': [str(d) for d in jax.devices()]}
    if args.token_dir is not None:
        manifest = dict(format_version=1, backend="bulkrnabert", metadata=dict(num_views=1), records=records)
        (args.token_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(args.output, embeddings=embeddings, sample_ids=ids,
                        metadata=json.dumps(metadata, sort_keys=True))


if __name__ == '__main__':
    main()
