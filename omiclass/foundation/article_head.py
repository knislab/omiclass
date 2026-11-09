"""Original fixed-epoch manuscript head, promoted to a reusable public component."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from omiclass.foundation.io import token_record, load_token_manifest, load_token_arrays


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resample_training(vectors, targets, train_ids, *, seed, k_neighbors, classes):
    """SMOTE sees only original training samples, never evaluation vectors."""
    import numpy as np
    try:
        import imblearn
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError('SMOTE requires imbalanced-learn in the training Python environment') from exc
    if k_neighbors < 1:
        raise ValueError('SMOTE k-neighbors must be positive')
    x = np.stack([vectors[sid] for sid in train_ids]).astype(np.float32)
    y = np.array([targets[sid] for sid in train_ids], dtype=np.int64)
    counts = np.bincount(y, minlength=len(classes))
    minority = counts[counts < counts.max()]
    effective = None
    if len(minority):
        if minority.min() < 2:
            raise ValueError('SMOTE needs at least two training samples in every class needing oversampling')
        effective = min(k_neighbors, int(minority.min()) - 1)
        x, y = SMOTE(sampling_strategy='auto', random_state=seed,
                     k_neighbors=effective).fit_resample(x, y)
    info = dict(method='smote', requested_k_neighbors=k_neighbors,
                effective_k_neighbors=effective, seed=seed,
                class_counts_before=dict(zip(classes, map(int, counts))),
                class_counts_after=dict(zip(classes, map(int, np.bincount(y, minlength=len(classes))))),
                synthetic_sample_count=len(y) - len(train_ids),
                imbalanced_learn_version=imblearn.__version__)
    return x, y, info


def make_head(width, token_count, hidden_dims, token_dim, dropout, n_classes):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.projection = nn.Linear(width, token_dim) if token_count else None
            dims = [token_count * token_dim if token_count else width] + hidden_dims
            layers = []
            for left, right in zip(dims, dims[1:]):
                layers.extend([nn.Linear(left, right), nn.GELU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(dims[-1], n_classes))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x, mask):
            x = self.norm(x)
            if self.projection is not None:
                x = torch.nn.functional.gelu(self.projection(x))
                x = (x * mask.unsqueeze(-1)).flatten(1)
            return self.mlp(x)

    return Head()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['full', 'mean'], required=True)
    p.add_argument('--cache', type=Path, help='Directory produced by extract.py')
    p.add_argument('--features', type=Path, help='Existing feature-by-sample CSV, for mean only')
    p.add_argument('--labels', type=Path, required=True, help='CSV containing sample ID and label columns')
    p.add_argument('--id-column', default='sample_id')
    p.add_argument('--label-column', required=True)
    p.add_argument('--split', type=Path, required=True, help='Existing Omiclass split.json; no new splits are generated')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hidden-dims', type=int, nargs='+', default=[64])
    p.add_argument('--token-dim', type=int, default=8, help='Full variant shared per-token projection width')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8, help='Number of sample views per batch for full')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--resampling', choices=['none', 'smote'], default='none')
    p.add_argument('--smote-k-neighbors', type=int, default=5)
    return p.parse_args()


def train(args):
    # Keep the historical seed and deterministic training choices inside a scope.
    # Validation failures and training/output failures restore the caller too.
    from omiclass.utils.reproducibility import random_state_scope
    with random_state_scope(getattr(args, 'seed', None), deterministic=True,
                            cudnn_benchmark=False, cublas=True,
                            cuda=str(getattr(args, 'device', 'cpu')) != 'cpu'):
        return _train(args)


def _train(args):
    if args.variant not in {'full', 'mean'}:
        raise ValueError('Variant must be full or mean')
    if args.smote_k_neighbors < 1:
        raise ValueError('SMOTE k-neighbors must be positive')
    if args.resampling == 'smote' and args.variant == 'full':
        raise ValueError('SMOTE supports mean sample vectors, not full tokens')
    if args.out.exists():
        raise FileExistsError(f'Output already exists: {args.out}')
    if min(args.hidden_dims + [args.token_dim, args.epochs, args.batch_size]) <= 0:
        raise ValueError('Dimensions, epochs and batch size must be positive')
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid dropout, learning rate or weight decay')
    if args.features and (args.variant != 'mean' or args.cache):
        raise ValueError('--features is for mean only and cannot be combined with --cache')
    if not args.features and not args.cache:
        raise ValueError('Provide --cache or, for mean, --features')
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn import metrics
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    device = torch.device(args.device)
    split = json.loads(args.split.read_text())
    partitions = {name: list(map(str, split[key])) for name, key in
                  [('train', 'train_index'), ('test', 'test_index')]}
    if 'validation_index' in split:
        partitions['validation'] = list(map(str, split['validation_index']))
    all_ids = [sid for ids in partitions.values() for sid in ids]
    if any(not ids for ids in partitions.values()) or len(set(all_ids)) != len(all_ids):
        raise ValueError('Split partitions must be nonempty, disjoint, and contain unique IDs')
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    if not {args.id_column, args.label_column}.issubset(labels.columns):
        raise ValueError('Missing ID/label column in labels CSV')
    if labels[args.id_column].duplicated().any():
        raise ValueError('Duplicate sample IDs in labels CSV')
    labels = labels.set_index(args.id_column)[args.label_column]
    if set(all_ids) - set(labels.index) or (labels.loc[all_ids].str.strip() == '').any():
        raise ValueError('Missing labels for split samples')
    classes = sorted(set(labels.loc[partitions['train']]))
    if len(classes) < 2 or set(labels.loc[all_ids]) - set(classes):
        raise ValueError('Training must contain at least two classes and cover every evaluation class')
    class_to_id = {label: i for i, label in enumerate(classes)}
    targets = {sid: class_to_id[labels[sid]] for sid in all_ids}
    manifest = None
    grouped = {}
    if args.cache:
        manifest = load_token_manifest(args.cache)
        for record in manifest['records']:
            sid = str(record['sample_id'])
            grouped.setdefault(sid, []).append(record)
        if set(all_ids) - set(grouped):
            raise ValueError('Cache is missing split samples')
        for sid in all_ids:
            views = [r['view'] for r in grouped[sid]]
            if sorted(views) != list(range(manifest['metadata']['num_views'])):
                raise ValueError(f'Incomplete or duplicate views for {sid}')
            if manifest['metadata'].get('token_schema') is not None:
                grouped[sid].sort(key=lambda record: int(record['view']))

    vectors = {}
    token_count = None
    if args.variant == 'full':
        with token_record(args.cache, grouped[all_ids[0]][0]) as z:
            token_count, width = z['tokens'].shape
    else:
        feature_path = args.features or args.cache / 'mean_features.csv'
        frame = pd.read_csv(feature_path, index_col=0)
        if frame.columns.duplicated().any() or set(all_ids) - set(frame.columns):
            raise ValueError('Feature CSV must contain unique columns for all split samples')
        vectors = {sid: frame[sid].to_numpy(dtype=np.float32) for sid in all_ids}
        width = len(vectors[all_ids[0]])
    if vectors and any(v.shape != (width,) or not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Vectors have inconsistent shape or non-finite values')

    class Samples(Dataset):
        def __init__(self, ids):
            self.rows = [(sid, r) for sid in ids for r in grouped[sid]] if args.variant == 'full' else [(sid, None) for sid in ids]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            sid, record = self.rows[index]
            if record is None:
                x, mask = vectors[sid], np.ones(1, dtype=bool)
            else:
                x, mask = load_token_arrays(args.cache, record,
                    schema=manifest['metadata'].get('token_schema'))
                if x.shape != (token_count, width) or mask.shape != (token_count,):
                    raise ValueError(f'Inconsistent token shape for {sid}')
                if not np.isfinite(x).all() or not mask.any():
                    raise ValueError(f'Invalid contextual representation for {sid}')
            return torch.from_numpy(x), torch.from_numpy(mask), targets[sid], sid

    training_data = Samples(partitions['train'])
    resampling = dict(method='none')
    if args.resampling == 'smote':
        train_x, train_y, resampling = resample_training(
            vectors, targets, partitions['train'], seed=args.seed,
            k_neighbors=args.smote_k_neighbors, classes=classes)
        training_data = TensorDataset(torch.from_numpy(train_x),
                                      torch.ones((len(train_y), 1), dtype=torch.bool),
                                      torch.from_numpy(train_y), torch.arange(len(train_y)))

    model = make_head(width, token_count, args.hidden_dims, args.token_dim, args.dropout, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    # Reject output collisions before training; never write into framework outputs.
    args.out.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for x, mask, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), mask.to(device))
            loss = nn.functional.cross_entropy(logits, y.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(train_loader.dataset)})
        print(json.dumps(history[-1]), flush=True)

    model.eval()
    results = []
    for partition, ids in partitions.items():
        if partition == 'train':
            continue
        sums, totals = {}, {}
        with torch.inference_mode():
            for x, mask, _, sids in DataLoader(Samples(ids), batch_size=args.batch_size, num_workers=0):
                probs = model(x.to(device), mask.to(device)).softmax(-1).cpu().numpy()
                for sid, prob in zip(sids, probs):
                    sums[sid] = sums.get(sid, np.zeros(len(classes), dtype=np.float64)) + prob
                    totals[sid] = totals.get(sid, 0) + 1
        proba = np.stack([sums[sid] / totals[sid] for sid in ids])
        proba /= proba.sum(axis=1, keepdims=True)
        truth = np.array([targets[sid] for sid in ids])
        pred = proba.argmax(1)
        row = dict(variant=args.variant, resampling=args.resampling, split=partition, n_test=len(ids),
                   accuracy=metrics.accuracy_score(truth, pred),
                   balanced_accuracy=metrics.balanced_accuracy_score(truth, pred),
                   f1_macro=metrics.f1_score(truth, pred, average='macro', zero_division=0),
                   f1_weighted=metrics.f1_score(truth, pred, average='weighted', zero_division=0),
                   precision_macro=metrics.precision_score(truth, pred, average='macro', zero_division=0),
                   recall_macro=metrics.recall_score(truth, pred, average='macro', zero_division=0),
                   log_loss=metrics.log_loss(truth, proba, labels=list(range(len(classes)))))
        if len(set(truth)) == len(classes):
            row['auroc'] = float(metrics.roc_auc_score(truth, proba[:, 1]) if len(classes) == 2 else
                                 metrics.roc_auc_score(truth, proba, multi_class='ovr', average='macro'))
            binary = np.eye(len(classes))[truth]
            row['average_precision'] = float(metrics.average_precision_score(truth, proba[:, 1]) if len(classes) == 2 else
                                             metrics.average_precision_score(binary, proba, average='macro'))
        else:
            row.update(auroc=None, average_precision=None, auc_unavailable_reason='Evaluation partition lacks one or more training classes')
        # Names used by the existing Omiclass holdout metric tables.
        row['auroc_ovr_macro'] = row['auroc']
        row['logloss'] = row['log_loss']
        results.append(row)
        predictions = pd.DataFrame(dict(sample_id=ids, y_true=[classes[t] for t in truth],
                                        y_pred=[classes[t] for t in pred], y_true_encoded=truth, y_pred_encoded=pred))
        for i in range(len(classes)):
            predictions[f'probability_{i}'] = proba[:, i]
        predictions.to_csv(args.out / f'{partition}_predictions.csv', index=False)
        report = metrics.classification_report(truth, pred, labels=list(range(len(classes))),
                                                target_names=classes, output_dict=True, zero_division=0)
        (args.out / f'{partition}_classification_report.json').write_text(json.dumps(report, indent=2))
        pd.DataFrame(metrics.confusion_matrix(truth, pred, labels=list(range(len(classes)))),
                     index=classes, columns=classes).to_csv(args.out / f'{partition}_confusion_matrix.csv')
    pd.DataFrame(results).to_csv(args.out / 'metrics.csv', index=False)
    (args.out / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    pd.DataFrame(history).to_csv(args.out / 'history.csv', index=False)
    configuration = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = dict(arguments=configuration, resampling=resampling, classes=classes, class_to_id=class_to_id,
                      token_count=token_count, embedding_width=width,
                      token_schema=manifest['metadata'].get('token_schema') if manifest else None,
                      token_num_views=manifest['metadata']['num_views'] if manifest else None,
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      split=split, split_sha256=digest(args.split), labels_sha256=digest(args.labels),
                      manifest_sha256=digest(args.cache / 'manifest.json') if args.cache else None,
                      features_sha256=digest(feature_path) if args.variant == 'mean' else None,
                      backend=manifest['backend'] if manifest else 'external_mean_features',
                      script_sha256=digest(__file__),
                      versions=dict(torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__),
                      selection='fixed epochs; evaluation partitions never used for fitting or model selection',
                      view_handling='full: equal per-view training loss, mean view probabilities at evaluation; mean: mean view embeddings')
    (args.out / 'run.json').write_text(json.dumps(provenance, indent=2))
    torch.save({'state_dict': model.cpu().state_dict(), 'run': provenance}, args.out / 'head.pt')


if __name__ == '__main__':
    train(parse_args())
