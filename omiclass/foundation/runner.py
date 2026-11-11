"""Opt-in API lifecycle for manuscript foundation protocols."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, average_precision_score

from omiclass.config.loaders import freeze_config
from omiclass.config.schema import Config
from omiclass.foundation import extraction, heads
from omiclass.foundation.io import save_frame, write_json
from omiclass.pipeline.runner_result import PipelineRunResult
from omiclass.utils.splits import holdout_split
from omiclass.train.selection import rank_candidates


def _metadata(cfg):
    from omiclass.utils.identifiers import validate_delimited_header, validate_frame_identifiers
    validate_delimited_header(cfg.project.meta_path, error_type=ValueError)
    meta = pd.read_csv(cfg.project.meta_path, index_col=0, dtype=str)
    validate_frame_identifiers(meta, error_type=ValueError)
    meta.index = meta.index.map(str)
    if not meta.index.is_unique:
        raise ValueError('Metadata sample IDs must be unique')
    return meta


def _resolve_paths(cfg):
    cfg = cfg.model_copy(deep=True)
    backend = getattr(cfg, extraction.backend_name(cfg))
    for name, value in backend:
        if isinstance(value, Path):
            setattr(backend, name, value.expanduser().resolve())
    if cfg.normalization.gene_lengths_path is not None:
        cfg.normalization.gene_lengths_path = Path(cfg.normalization.gene_lengths_path).expanduser().resolve()
    cfg.project.meta_path = Path(cfg.project.meta_path).expanduser().resolve()
    if cfg.foundation.split_path is not None:
        cfg.foundation.split_path = cfg.foundation.split_path.expanduser().resolve()
    return cfg


def _signature(cfg):
    foundation = cfg.foundation.model_dump(mode='json')
    for key in ['mode', 'run_dir', 'split_path', 'prediction_scope']:
        foundation.pop(key, None)
    encoder = getattr(cfg, extraction.backend_name(cfg)).model_dump(mode='json')
    for key in ['tpm_input_path', 'cache_dir']:
        encoder.pop(key, None)
    return dict(foundation=foundation, encoder=encoder, normalization=cfg.normalization.model_dump(mode='json'),
        correction=cfg.batch_correction.model_dump(mode='json') if cfg.foundation.correction != 'none' else None)



def _resource_digests(cfg, version=2):
    """Bind saved prediction to the same supplied encoder assets, not just paths."""
    if version == 2:
        from omiclass.foundation.assets import encoder_identities
        return encoder_identities(cfg)
    if version != 1:
        raise ValueError(f"Unsupported foundation artifact format: {version}")
    # Preserve the exact checks available in original Step 2 bundles. They
    # cannot retroactively verify sources or checkpoint-directory contents.
    result = {}
    encoder = getattr(cfg, extraction.backend_name(cfg))
    for name, value in encoder:
        if not isinstance(value, Path) or name in {'cache_dir', 'tpm_input_path', 'root_dir', 'vendor_dir'}:
            continue
        path = value.expanduser().resolve()
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            result[name] = dict(path=str(path), sha256=digest.hexdigest())
        else:
            result[name] = dict(path=str(path), absent=True)
    if cfg.normalization.gene_lengths_path is not None:
        path = Path(cfg.normalization.gene_lengths_path)
        result['normalization_gene_lengths'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def _split(counts, metadata, cfg):
    column = cfg.project.cluster_column
    if column not in metadata:
        raise ValueError(f'Missing classification column {column}')
    labels = metadata[column].reindex(counts.columns)
    classes = list(cfg.project.classes_names or sorted(labels.dropna().astype(str).unique()))
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('At least two distinct canonical classes are required')
    classes = list(map(str, classes))
    mapping = {name: i for i, name in enumerate(classes)}
    eligible = labels.index[labels.notna() & labels.isin(classes)].tolist()
    y = labels.loc[eligible].map(mapping).astype(int)
    if cfg.foundation.split_path is not None:
        split = json.loads(cfg.foundation.split_path.read_text())
    else:
        _, _, _, _, split = holdout_split(pd.DataFrame(index=y.index), y,
            method=cfg.runtime.split.method, train_ratio=cfg.runtime.split.train_ratio,
            random_state=cfg.runtime.random_state, hold_out_path=cfg.project.hold_out_path)
    train, test = list(map(str, split['train_index'])), list(map(str, split['test_index']))
    if not train or not test or len(set(train+test)) != len(train+test):
        raise ValueError('Training and holdout partitions must be nonempty, unique and disjoint')
    if set(train+test) - set(eligible):
        raise ValueError('Split contains absent samples, unknown classes or missing labels')
    if set(y.loc[train]) != set(range(len(classes))):
        raise ValueError('Training must contain every canonical class')
    split['train_index'], split['test_index'] = train, test
    return split, y, classes


def _probabilities(model, X, cache, ids, n_classes):
    from omiclass.eval.prediction import canonical_classes, validate_probabilities, align_probabilities
    if isinstance(model, heads.TokenModel):
        raw = model.predict_proba(cache, ids)
        predicted = model.classes_[raw.argmax(1)]
    else:
        raw = model.predict_proba(X.loc[ids])
        # SVC decision predictions may differ from probability argmax. Retain them.
        predicted = model.predict(X.loc[ids])
    classes = canonical_classes(model)
    p = align_probabilities(validate_probabilities(raw, classes, len(ids)), classes, np.arange(n_classes))
    if not set(predicted).issubset(set(classes)):
        raise ValueError('Saved classifier predictions differ from its class schema')
    return np.asarray(predicted, dtype=int), p


def _evaluation(y, predicted, p, classes):
    result = dict(accuracy=float(accuracy_score(y, predicted)),
        f1_macro=float(f1_score(y, predicted, average='macro', zero_division=0)),
        logloss=float(log_loss(y, p / p.sum(axis=1, keepdims=True), labels=list(range(len(classes))))),
        auroc_macro=None, average_precision_macro=None)
    if len(set(y)) == len(classes):
        if len(classes) == 2:
            result.update(auroc_macro=float(roc_auc_score(y, p[:, 1])),
                          average_precision_macro=float(average_precision_score(y, p[:, 1])))
        else:
            result.update(auroc_macro=float(roc_auc_score(y, p, labels=list(range(len(classes))), multi_class='ovr', average='macro')),
                average_precision_macro=float(average_precision_score(np.eye(len(classes))[np.asarray(y)], p, average='macro')))
    return result


def _prediction_frame(ids, model_name, predicted, probabilities, classes, truth=None):
    table = pd.DataFrame(dict(sample_id=ids, model=model_name, y_pred=predicted,
                              predicted_label=[classes[i] for i in predicted]))
    if truth is not None:
        table['y_true'] = np.asarray(truth, dtype=int)
    for j in range(len(classes)):
        table[f'probability_{j}'] = probabilities[:, j]
    return table


def _new_output(cfg, prediction=False):
    root = Path(cfg.project.out_dir).expanduser().resolve()
    if prediction and (root == cfg.foundation.run_dir.resolve() or root.is_relative_to(cfg.foundation.run_dir.resolve())):
        raise ValueError('Prediction output must be separate from the saved training run')
    prefix = 'prediction' if prediction else cfg.project.name
    if not prefix or Path(prefix).name != prefix:
        raise ValueError('Project name must be a directory name')
    out = root / f'{prefix}-{datetime.now().strftime("%Y%m%d-%H%M%S-%f")}-{uuid4().hex[:8]}'
    out.mkdir(parents=True, exist_ok=False)
    return out


def _verify_bundle(saved):
    manifest = json.loads((saved / 'foundation.json').read_text())
    if manifest.get('format_version') not in {1, 2}:
        raise ValueError('Unsupported foundation artifact format')
    if manifest.get('status') != 'complete':
        raise ValueError('Saved foundation run is not complete')
    required = {'config.freeze.yaml', *manifest['models'].values()}
    if not required.issubset(manifest['artifact_sha256']):
        raise ValueError('Saved foundation models/configuration lack checksums')
    for name, expected in manifest['artifact_sha256'].items():
        path = (saved / name).resolve()
        if not path.is_relative_to(saved) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Saved foundation artifact changed or escapes its run')
    return manifest


def _predict(expression_file, cfg):
    saved = cfg.foundation.run_dir.expanduser().resolve()
    manifest = _verify_bundle(saved)
    original_cfg = Config(**yaml.safe_load((saved / 'config.freeze.yaml').read_text()))
    if _signature(cfg) != _signature(original_cfg):
        raise ValueError('Prediction overrides change the fitted extraction/protocol configuration')
    version = manifest['format_version']
    if _resource_digests(cfg, version) != manifest['encoder_assets']:
        raise ValueError('Encoder assets changed since training')
    input_protocol = extraction.input_identifier_protocol(cfg, manifest)
    counts = extraction.read_expression(expression_file, protocol=input_protocol)
    known = set(manifest['split']['train_index'] + manifest['split']['test_index'])
    ids = counts.columns.tolist() if cfg.foundation.prediction_scope == 'all' else [s for s in counts.columns if s not in known]
    if not ids:
        raise ValueError('No external samples remain; use prediction_scope="all" only for explicit inference replay')
    if cfg.foundation.correction != 'none' and not set(manifest['source_ids']).issubset(counts.columns):
        raise ValueError('Historical external re-correction requires the full original input cohort plus external samples')
    out = _new_output(cfg, prediction=True)
    protocol = manifest.get('extraction_protocol', {})
    strategy = protocol.get('cellfm_single_view', 'legacy_global_numpy_random_state')
    legacy = (extraction.backend_name(cfg) == 'cellfm' and cfg.cellfm.num_views == 1
              and strategy == 'legacy_global_numpy_random_state')
    if extraction.backend_name(cfg) == 'cellfm' and cfg.cellfm.num_views == 1 and strategy not in {
            'legacy_global_numpy_random_state', 'sha256_sample_id_global_seed_zero_based_view_index'}:
        raise ValueError('Unsupported saved CellFM single-view sampling protocol')
    replay = {'legacy_cellfm_single_view': True} if legacy else {}
    extracted = extraction.extract(counts, cfg, out / 'extraction',
        _metadata(cfg) if cfg.foundation.correction != 'none' else None, **replay)
    if _resource_digests(cfg, version) != manifest['encoder_assets']:
        raise ValueError('Encoder assets changed during prediction extraction')
    if set(extracted.features.columns) != set(manifest['raw_features']):
        raise ValueError('Prediction embedding schema differs from training')
    extracted.features = extracted.features.loc[:, manifest['raw_features']]
    frames = []
    for name, relative in manifest['models'].items():
        model = joblib.load(saved / relative)
        predicted, proba = _probabilities(model, extracted.features, extracted.token_cache, ids, len(manifest['classes']))
        frames.append(_prediction_frame(ids, name, predicted, proba, manifest['classes']))
    path = out / 'predictions.csv'
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    write_json(out / 'prediction.json', dict(source_run=str(saved), samples=ids, refitted=False,
        correction='full supplied prediction input re-corrected' if cfg.foundation.correction != 'none' else 'none',
        source_format_version=version,
        input_identifier_protocol=input_protocol,
        encoder_verification='file_and_source_identities' if version == 2 else 'legacy_file_checks_only',
        input_expression=str(Path(expression_file).resolve()),
        tpm_input_path=str(cfg.bulkrnabert.tpm_input_path) if extraction.backend_name(cfg) == 'bulkrnabert' and cfg.bulkrnabert.tpm_input_path else None,
        classes=manifest['classes'], selected_model=manifest['selected_model']))
    return PipelineRunResult(out, None, None, {'predictions': path}, {}, path)


def run_foundation(expression_file, cfg):
    cfg = _resolve_paths(cfg)
    if cfg.foundation.mode == 'predict':
        return _predict(expression_file, cfg)
    input_protocol = extraction.input_identifier_protocol(cfg)
    counts = extraction.read_expression(expression_file, protocol=input_protocol)
    meta = _metadata(cfg)
    split, y, classes = _split(counts, meta, cfg)
    train, test = split['train_index'], split['test_index']
    out = _new_output(cfg)
    cfg.project.out_dir = out
    cfg.project.split_data_dir = out
    cfg.project.classes_names = classes
    freeze_config(cfg, out)
    write_json(out / 'split.json', split)
    encoder_assets = _resource_digests(cfg)
    extracted = extraction.extract(counts, cfg, out / 'extraction', meta)
    if _resource_digests(cfg) != encoder_assets:
        raise ValueError('Encoder assets changed during training extraction')
    X = extracted.features.loc[train+test]
    save_frame(out / 'extraction/features.npz', extracted.features)
    models, board = heads.fit(cfg, X, y, train, test, classes, extracted.token_cache, out / 'training')
    if not models:
        raise RuntimeError('No fitted foundation models')
    if cfg.foundation.classifier in {'omiclass', 'foundation_classifier'}:
        ranked, _ = rank_candidates(board, models, cfg.runtime.scoring)
        selected = str(ranked.iloc[0]['model'])
    elif len(models) == 1:
        selected = next(iter(models))
    else:
        raise ValueError('Standalone foundation protocol returned multiple models')
    model_dir = out / 'models'
    model_dir.mkdir()
    artifacts, paths, evaluations, prediction_frames = {}, {}, [], []
    for name, model in models.items():
        path = model_dir / f'{name}.joblib'
        joblib.dump(model, path)
        paths[name] = str(path.relative_to(out))
        artifacts[name] = path
        predicted, p = _probabilities(model, X, extracted.token_cache, test, len(classes))
        evaluations.append(dict(model=name, **_evaluation(y.loc[test], predicted, p, classes)))
        prediction_frames.append(_prediction_frame(test, name, predicted, p, classes, y.loc[test]))
    board.to_csv(out / 'leaderboard.csv', index=False)
    pd.DataFrame(evaluations).to_csv(out / 'test_metrics.csv', index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / 'holdout_predictions.csv', index=False)
    signed = [*paths.values(), 'config.freeze.yaml']
    seed, source = extraction.encoder_seed(cfg)
    protocol = dict(encoder_seed=seed, seed_source=source, input_identifiers=input_protocol)
    if extraction.backend_name(cfg) == 'cellfm' and cfg.cellfm.num_views == 1:
        protocol['cellfm_single_view'] = 'sha256_sample_id_global_seed_zero_based_view_index'
    manifest = dict(format_version=2, status='complete', backend=extraction.backend_name(cfg), encoder_assets=encoder_assets,
        extraction_protocol=protocol,
        model=cfg.foundation.model, representation=cfg.foundation.representation, pooling=cfg.foundation.pooling,
        classifier=cfg.foundation.classifier, classes=classes, split=split, source_ids=counts.columns.tolist(),
        raw_features=extracted.features.columns.tolist(), selected_model=selected, models=paths,
        diagnostic=False,
        evidence=('validated experimental protocol; no completed full-scale manuscript fits'
            if cfg.foundation.model == 'bulkrnabert' and cfg.foundation.representation == 'tokens'
            else 'implemented framework extension; outside recovered matched manuscript configurations'
            if cfg.foundation.pooling in {'max', 'median'} or (extraction.backend_name(cfg) == 'cellfm' and cfg.cellfm.num_views not in {1, 20})
            else 'saved manuscript protocol'),
        correction_transductive=cfg.foundation.correction != 'none',
        artifact_sha256={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in signed})
    write_json(out / 'foundation.json', manifest)
    return PipelineRunResult(out, out / 'split.json', out / 'leaderboard.csv', artifacts, {}, out / 'test_metrics.csv')
