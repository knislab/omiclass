"""Article-only corrected values in the original count-selected CellFM views."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from omiclass.foundation.io import write_json


def continuous_expression(counts, cfg):
    """Full mapped, capped-count matrix; same float32 normalize as Prepare.normalize.

    The denominator includes every mapped capped gene, before view selection.
    Match the existing multiview canonical order before summing float32 counts.
    No model/vendor imports are needed in the parent Omiclass interpreter.
    """
    from omiclass.features import cellfm as backend
    vocabulary = backend.load_cellfm_vocabulary(Path(cfg.gene_info_path))
    prepared = backend.prepare_cellfm_input(counts, vocabulary)
    order = (np.argsort(prepared.gene_ids, kind='stable') if int(cfg.num_views) > 1
             else np.arange(len(prepared.gene_ids)))
    data = prepared.counts[:, order].astype(np.float32)
    read = data.sum(axis=1, dtype=np.float32)
    values = np.log1p(data / read[:, None] * 1e4).astype(np.float32)
    return pd.DataFrame(values.T, index=[prepared.mapped_genes[i] for i in order],
                        columns=prepared.sample_ids)


def extract_corrected(counts, corrected, cfg, directory, observer=None):
    """Reuse core selection/pooling; substitute scalars only on this model instance.

    Replay ALL original samples in original order, including samples not retained
    by the downstream split, so legacy single-view NumPy draws stay identical.
    Non-target samples keep original values and their embeddings are discarded.
    """
    import torch
    from omiclass.features import cellfm as backend
    backend._validate_resources(cfg)
    vocabulary = backend.load_cellfm_vocabulary(Path(cfg.gene_info_path))
    prepared = backend.prepare_cellfm_input(counts, vocabulary)
    original = continuous_expression(counts, cfg)
    if not set(corrected.columns).issubset(prepared.sample_ids):
        raise ValueError('Corrected CellFM sample IDs are absent from original counts')
    utils, model_module = backend._import_cellfm(cfg)
    device = backend._resolve_device(torch, cfg.device)
    model_cfg = utils.Config_80M()
    model_cfg.ecs = True
    model_cfg.ecs_threshold = 0.8
    model_cfg.add_zero = True
    model_cfg.pad_zero = True
    model_cfg.mask_ratio = 0.0
    model_cfg.recompute = False
    model_cfg.device = str(device)
    model_cfg.ckpt_path = str(cfg.checkpoint_path)
    prep = utils.Prepare(pad_len=model_cfg.nonz_len, pad=0, mask_ratio=0.0,
                         dw=False, random=False)
    if (int(model_cfg.nonz_len), int(model_cfg.enc_dims)) != (2048, 1536):
        raise ValueError('Unsupported CellFM sequence length/embedding width')
    # Verify the locally reproduced full normalization against the vendor function.
    order = (np.argsort(prepared.gene_ids, kind='stable') if int(cfg.num_views) > 1
             else np.arange(len(prepared.gene_ids)))
    for i, row in enumerate(prepared.counts[:, order].astype(np.float32)):
        normalized, _ = prep.normalize(row, np.asarray(row.sum(), dtype=np.float32))
        if not np.array_equal(normalized, original.iloc[:, i].to_numpy()):
            raise ValueError('Full CellFM normalization differs from existing Prepare.normalize')
    state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(cfg.random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.random_state))
    try:
        model = model_module.Cell_FM(n_gene=len(vocabulary), cfg=model_cfg,
            ckpt_path=str(cfg.checkpoint_path), device=str(device)).to(device)
        model.load_model(weight=True, moment=False)
    finally:
        torch.random.set_rng_state(state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    model.eval()
    model.requires_grad_(False)
    # The vendor encode ends with a NaN replacement. Fail before that fallback
    # rather than silently conceal a numerical incompatibility of corrected input.
    def require_finite(module, arguments, output):
        tensors = output if isinstance(output, (tuple, list)) else [output]
        if any(torch.is_tensor(value) and not torch.isfinite(value).all() for value in tensors):
            raise ValueError(f'Nonfinite corrected CellFM activation in {type(module).__name__}; '
                             'refusing the encoder NaN replacement fallback')
    hooks = [module.register_forward_hook(require_finite)
             for module in [model.net.value_enc, *model.net.encoder]]
    encode = model.net.encode
    position = 0
    sample_count = len(prepared.sample_ids)
    selection_hash = hashlib.sha256()
    audit_dir = directory / 'cellfm_views'
    audit_dir.mkdir()
    target_ids = set(corrected.columns)
    # Reindexing marks limma-filtered genes as unavailable, never imputes them.
    corrected_lookup = corrected.reindex(vocabulary)
    original_lookup = original.reindex(vocabulary)

    def fixed_view_encode(expression, genes, st_features, valid):
        nonlocal position
        view, start = divmod(position, sample_count)
        batch_count = len(expression)
        if start + batch_count > sample_count or view >= int(cfg.num_views):
            raise ValueError('Unexpected CellFM view traversal; cannot preserve sample/view identity')
        ids = prepared.sample_ids[start:start + batch_count]
        gene_array = genes.detach().cpu().numpy()
        valid_array = valid.detach().cpu().numpy()
        before = expression.detach().cpu().numpy()
        # Prepare.mask emits [unmask, expression] pairs even at mask_ratio=0.
        # Never replace the unmask channel (or the padding pairs).
        if before.ndim != 3 or before.shape[-1] != 2:
            raise ValueError('Unexpected CellFM [unmask, expression] token layout')
        after = before.copy()
        for i, sid in enumerate(ids):
            mask = valid_array[i].astype(bool)
            selected = gene_array[i, mask].astype(np.int64) - 1
            if (selected < 0).any() or (selected >= len(vocabulary)).any():
                raise ValueError('Invalid non-padding CellFM gene ID')
            expected = original_lookup[sid].to_numpy()[selected]
            if not np.array_equal(before[i, mask, 1], expected):
                raise ValueError('Count-selected values differ from full continuous normalization')
            if sid in target_ids:
                values = corrected_lookup[sid].to_numpy()[selected]
                if not np.isfinite(values).all():
                    missing = [vocabulary[j] for j in selected[~np.isfinite(values)]]
                    raise ValueError(f'Limma removed/unavailable genes selected for CellFM {sid}: '
                                     f'{missing[:10]}; refusing filling or changed selection')
                after[i, mask, 1] = values
        if not np.isfinite(after).all():
            raise ValueError('Corrected CellFM values exceed float32 range; no clipping permitted')
        selection_hash.update(gene_array.tobytes())
        selection_hash.update(valid_array.tobytes())
        # Persist exact gene order, padding, count-derived auxiliaries and both values.
        np.savez_compressed(audit_dir / f'view{view:03d}_sample{start:07d}.npz',
            sample_ids=np.asarray(ids, dtype=str), gene_ids=gene_array,
            valid=valid_array, st_features=st_features.detach().cpu().numpy(),
            original_values=before, corrected_values=after,
            corrected_samples=np.asarray([sid in target_ids for sid in ids]))
        position += batch_count
        encoded, extra = encode(torch.from_numpy(after).to(device=expression.device,
                      dtype=expression.dtype), genes, st_features, valid)
        if observer is not None:
            observer(encoded[:, 3:], valid, genes, ids, view, encoded[:, 0])
        return encoded, extra

    model.net.encode = fixed_view_encode
    try:
        embeddings, retained, seed_strategy, aggregation = backend._extract_configured_views(
            cfg=cfg, utils=utils, torch=torch, model=model, prep=prep,
            prepared=prepared, device=device)
    finally:
        model.net.encode = encode
        for hook in hooks:
            hook.remove()
    if position != sample_count * int(cfg.num_views):
        raise ValueError('Incomplete CellFM view traversal')
    metadata = dict(protocol='original_count_views_corrected_scalars_v1',
        selection_random_state=int(cfg.random_state), num_views=int(cfg.num_views),
        view_seed_strategy=seed_strategy, view_aggregation=aggregation,
        pooling='valid_gene_mean', selection_sha256=selection_hash.hexdigest(),
        selection_input='original capped/rounded mapped counts',
        normalization='Prepare.normalize: log1p(count / mapped_capped_read * 1e4), float32',
        original_sample_order=prepared.sample_ids, retained_genes=retained,
        unchanged_inputs=['gene IDs/order', 'valid/padding mask', 'unmask channel', 'ST library features'],
        negative_values='passed unchanged to ValueEncoder; no clipping/shift',
        filtered_genes='fail if selected; never fill or reselect',
        audit_directory='cellfm_views')
    write_json(directory / 'cellfm_fixed_views.json', metadata)
    result = backend.cellfm_feature_matrix(embeddings, prepared.sample_ids).T
    return result.loc[corrected.columns], metadata
