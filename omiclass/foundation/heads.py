"""Adapters retaining the manuscript's distinct training/evaluation protocols."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from omiclass.features.bulkformer_reduction import EmbeddingRepresentation
from omiclass.foundation.extraction import backend_name
from omiclass.foundation.io import token_record, write_json


def representation(cfg):
    backend = backend_name(cfg)
    reduction = getattr(cfg, backend).reduction.model_dump()
    reduction['strategy'] = 'full' if cfg.foundation.representation == 'cls' else cfg.foundation.representation
    if reduction['random_state'] is None:
        reduction['random_state'] = cfg.runtime.random_state
    return EmbeddingRepresentation(**reduction)


def safe_cv(y, desired, seed):
    minimum = int(pd.Series(y).value_counts().min())
    if minimum < 2:
        raise ValueError('Every training class needs at least two samples for CV')
    return StratifiedKFold(n_splits=max(2, min(desired, minimum)), shuffle=True, random_state=seed)


def fit_lr(cfg, X, y, out):
    from sklearn.metrics import f1_score, accuracy_score
    r, f = cfg.runtime, cfg.foundation
    def search(values, labels):
        pipeline = Pipeline([('representation', representation(cfg)), ('scaler', StandardScaler()),
            ('logistic_regression', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, random_state=r.random_state))])
        fit = GridSearchCV(pipeline, {'logistic_regression__C': f.lr_c_values,
            'logistic_regression__class_weight': [None, 'balanced']}, scoring=r.scoring,
            cv=safe_cv(labels, r.tuning.inner_cv or 3, r.random_state), n_jobs=r.n_jobs,
            refit=True, return_train_score=False, error_score='raise')
        fit.fit(values, labels.astype(int))
        return fit
    folds, oof = [], pd.DataFrame(index=X.index)
    scopes = []
    for number, (tr, va) in enumerate(safe_cv(y, r.cv_folds or 5, r.random_state).split(X, y), 1):
        fitted = search(X.iloc[tr], y.iloc[tr])
        predicted = fitted.predict(X.iloc[va])
        probabilities = fitted.predict_proba(X.iloc[va])
        folds.append(dict(fold=number, f1_macro=float(f1_score(y.iloc[va], predicted, average='macro', zero_division=0)),
            accuracy=float(accuracy_score(y.iloc[va], predicted)), best_score=float(fitted.best_score_),
            best_params=json.dumps(fitted.best_params_)))
        oof.loc[X.index[va], 'y_true'] = y.iloc[va].to_numpy()
        oof.loc[X.index[va], 'y_pred'] = predicted
        for j, label in enumerate(fitted.classes_):
            oof.loc[X.index[va], f'probability_{int(label)}'] = probabilities[:, j]
        scopes.append(dict(fold=number, representation_fit_ids=X.index[tr].tolist(),
            validation_ids=X.index[va].tolist(), note='representation/scaler remain inside every inner search fit'))
    fitted = search(X, y)
    pd.DataFrame(folds).to_csv(out / 'fold_metrics.csv', index=False)
    pd.DataFrame(fitted.cv_results_).to_csv(out / 'inner_search_results.csv', index=False)
    oof.to_csv(out / 'oof_predictions.csv', index_label='sample_id')
    write_json(out / 'fit_scope.json', dict(protocol='nested_cv_lr', folds=scopes,
        final_fit_ids=X.index.tolist(), holdout_used_for_fitting=False))
    return {'logistic_regression': fitted.best_estimator_}, pd.DataFrame([dict(model='logistic_regression',
        f1_macro_mean=float(np.mean([row['f1_macro'] for row in folds])), protocol='nested_cv_lr')])


class NativeModel:
    """Persist the native adapter's log decision and training-only constant filter."""
    def __init__(self, model, columns, did_log):
        self.model, self.columns, self.did_log = model, list(columns), bool(did_log)
        self.classes_ = np.asarray(model.classes_)

    def prepare(self, X):
        if set(self.columns) - set(X.columns):
            raise ValueError('Prediction lacks fitted embedding dimensions')
        if self.did_log:
            X = np.log1p(X)
        if not np.isfinite(X.loc[:, self.columns].to_numpy()).all():
            raise ValueError('Native embedding scale is incompatible with the saved log transform')
        return X.loc[:, self.columns]

    def predict(self, X):
        return self.model.predict(self.prepare(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.prepare(X))


def fit_native(cfg, X, y, train_ids, out):
    from omiclass.train.trainer import Trainer
    from omiclass.utils.pipeline import maybe_log1p_expression
    cfg = cfg.model_copy(deep=True)
    cfg.runtime.cv_folds = cfg.runtime.cv_folds or 5
    cfg.runtime.tuning.inner_cv = cfg.runtime.tuning.inner_cv or 3
    cfg.de.geneset_selection = False
    matrix, did_log, reason = maybe_log1p_expression(X.T, enabled=cfg.runtime.log_transform,
                                                   already_normalized_or_logged=False)
    matrix = matrix.T
    keep = matrix.loc[train_ids].nunique(dropna=False).gt(1)
    if not keep.any():
        raise ValueError('No nonconstant training embedding dimensions')
    matrix = matrix.loc[:, keep]
    trainer = Trainer(cfg, artifacts_dir=out / 'train_artifacts')
    result = trainer.run(matrix.loc[train_ids], y.loc[train_ids])
    board = result.metrics
    board.to_csv(out / 'native_leaderboard.csv', index=False)
    joblib.dump(trainer.cv_results, out / 'cv_results.joblib')
    if not result.artifact_paths or ('status' in board and not board.status.eq('ok').all()):
        raise RuntimeError('Native candidate failed; retained diagnostics do not constitute a completed suite')
    models = {name: NativeModel(joblib.load(path), matrix.columns, did_log) for name, path in result.artifact_paths.items()}
    write_json(out / 'fit_scope.json', dict(protocol='native_omiclass_cv_and_imbalance_dispatch',
        constant_filter_fit_ids=train_ids, features=matrix.columns.tolist(), did_log=did_log, reason=reason,
        selection='Existing Trainer including calibration, tuning, sampling and stacker is unchanged'))
    return models, board


class NeuralModel:
    """Portable numpy state and fitted representation; imports Torch only at use."""
    def __init__(self, state, architecture, classes, transform=None, trained_class_order=None):
        self.state, self.architecture, self.transform = state, architecture, transform
        self.classes_ = np.arange(len(classes))
        self.class_names = list(classes)
        self.column_order = [list(trained_class_order).index(c) for c in classes] if trained_class_order else list(range(len(classes)))

    def network(self):
        import torch
        if self.architecture['family'] == 'bert':
            from omiclass.foundation.bert_head import make_mlp
            model = make_mlp(len(self.classes_), self.architecture['width'])
        else:
            from omiclass.foundation.article_head import make_head
            a = self.architecture
            model = make_head(a['width'], a['token_count'], a['hidden_dims'], a['token_dim'], a['dropout'], len(self.classes_))
        model.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in self.state.items()})
        model.eval()
        return model

    def predict_proba(self, X):
        import torch
        from omiclass.foundation.bert_head import torch_reproducibility
        values = self.transform.transform(X) if self.transform is not None else X
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.architecture['width'] or not np.isfinite(values).all():
            raise ValueError('Saved neural-head input schema mismatch')
        with torch_reproducibility(0, 1), torch.inference_mode():
            model = self.network()
            result = []
            for array in np.array_split(values, max(1, int(np.ceil(len(values) / 1024)))):
                x = torch.from_numpy(array)
                logits = model(x) if self.architecture['family'] == 'bert' else model(x, torch.ones((len(x), 1), dtype=torch.bool))
                result.append(logits.softmax(-1).numpy())
        return np.concatenate(result)[:, self.column_order]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class TokenModel(NeuralModel):
    def predict_proba(self, cache, ids):
        import torch
        from omiclass.foundation.bert_head import torch_reproducibility
        manifest = json.loads((Path(cache) / 'manifest.json').read_text())
        grouped = {str(s): [] for s in ids}
        for record in manifest['records']:
            if str(record['sample_id']) in grouped:
                grouped[str(record['sample_id'])].append(record)
        expected_views = int(manifest['metadata']['num_views'])
        with torch_reproducibility(0, 1), torch.inference_mode():
            model = self.network()
            all_probabilities = []
            for sid in ids:
                records = grouped[str(sid)]
                if sorted(int(r['view']) for r in records) != list(range(expected_views)):
                    raise ValueError('Incomplete or duplicated prediction views')
                total = np.zeros(len(self.classes_), dtype=np.float64)
                for record in records:
                    with token_record(cache, record) as z:
                        x, mask = np.asarray(z['tokens'], dtype=np.float32), np.asarray(z['mask'], dtype=bool)
                        if x.shape != (self.architecture['token_count'], self.architecture['width']) or mask.shape != (len(x),) or not mask.any() or not np.isfinite(x).all():
                            raise ValueError('Token prediction shape/mask mismatch')
                        total += model(torch.from_numpy(x.copy())[None], torch.from_numpy(mask.copy())[None]).softmax(-1).numpy()[0]
                total /= len(records)
                all_probabilities.append(total / total.sum())
        return np.stack(all_probabilities)[:, self.column_order]

    def predict(self, cache, ids):
        return self.classes_[self.predict_proba(cache, ids).argmax(1)]


def fit_bert(cfg, X, y, train_ids, classes, out):
    from omiclass.foundation import bert_head
    config = bert_head.MLPConfig(**cfg.foundation.bert_head.model_dump(),
        resampling='smote' if cfg.foundation.classifier == 'mlp_smote' else 'none')
    inner, validation = bert_head.inner_partition(train_ids, y.loc[train_ids].to_numpy(), config, cfg.runtime.random_state)
    fit_ids = [train_ids[i] for i in inner]
    transform = representation(cfg).fit(X.loc[fit_ids], y.loc[fit_ids])
    vectors = transform.transform(X.loc[train_ids]).astype(np.float32)
    model, info = bert_head.fit_mlp(vectors.to_numpy(), y.loc[train_ids].to_numpy(), train_ids,
        len(classes), config, cfg.runtime.random_state, input_width=vectors.shape[1])
    if info['inner_train_ids'] != fit_ids:
        raise ValueError('BERT representation fit and head inner-training membership differ')
    pd.DataFrame(info.pop('history')).to_csv(out / 'history.csv', index=False)
    write_json(out / 'training.json', dict(configuration=config.metadata(), input_width=vectors.shape[1], **info))
    write_json(out / 'fit_scope.json', dict(protocol='bert_inner_validation_early_stopping',
        representation_fit_ids=fit_ids, inner_validation_ids=[train_ids[i] for i in validation],
        refit_on_all_training=False, holdout_used_for_fitting=False))
    estimator = NeuralModel({k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()},
        dict(family='bert', width=vectors.shape[1]), classes, transform)
    return {'bert_mlp': estimator}, pd.DataFrame([dict(model='bert_mlp', protocol='bert_inner_validation_early_stopping', f1_macro_mean=None)])


def fit_article(cfg, X, y, train_ids, test_ids, classes, cache, out):
    import torch
    from omiclass.foundation import article_head
    token = cfg.foundation.representation == 'tokens'
    transform = None
    features = None
    if not token:
        transform = representation(cfg).fit(X.loc[train_ids], y.loc[train_ids])
        features = out / 'head_features.csv'
        transform.transform(X.loc[train_ids+test_ids]).astype(np.float32).T.to_csv(features)
    split = out / 'head_split.json'
    labels = out / 'head_labels.csv'
    write_json(split, dict(train_index=train_ids, test_index=test_ids))
    pd.DataFrame({'sample_id': train_ids+test_ids, 'label': [classes[int(y[s])] for s in train_ids+test_ids]}).to_csv(labels, index=False)
    args = SimpleNamespace(**cfg.foundation.article_head.model_dump(), variant='full' if token else 'mean',
        features=features, cache=cache if token else None, labels=labels, id_column='sample_id', label_column='label',
        split=split, out=out / 'article_training', seed=cfg.runtime.random_state, device='cpu',
        resampling='smote' if cfg.foundation.classifier == 'mlp_smote' else 'none')
    article_head.train(args)
    saved = torch.load(args.out / 'head.pt', map_location='cpu', weights_only=False)
    run = saved['run']
    architecture = dict(family='article', width=run['embedding_width'], token_count=run['token_count'],
        hidden_dims=args.hidden_dims, token_dim=args.token_dim, dropout=args.dropout)
    cls = TokenModel if token else NeuralModel
    estimator = cls({k: v.numpy().copy() for k, v in saved['state_dict'].items()}, architecture,
        classes, transform, trained_class_order=run['classes'])
    write_json(out / 'fit_scope.json', dict(protocol='article_fixed_epochs', representation_fit_ids=train_ids if not token else [],
        head_training_ids=train_ids, smote_fit_ids=train_ids if args.resampling == 'smote' else [],
        evaluation_used_for_selection=False, token_views='equal per-view loss; mean view probabilities'))
    return {'article_mlp': estimator}, pd.DataFrame([dict(model='article_mlp', protocol='article_fixed_epochs', f1_macro_mean=None)])


def fit(cfg, X, y, train_ids, test_ids, classes, cache, out):
    out.mkdir(exist_ok=False)
    classifier = cfg.foundation.classifier
    if classifier in {'omiclass', 'foundation_classifier'}:
        return fit_native(cfg, X, y, train_ids, out)
    if classifier == 'lr':
        return fit_lr(cfg, X.loc[train_ids], y.loc[train_ids], out)
    if classifier == 'svm':
        model = Pipeline([('representation', representation(cfg)), ('scaler', StandardScaler()),
            ('svm', SVC(C=1.0, kernel='rbf', gamma='scale', class_weight=None, probability=True,
                decision_function_shape='ovr', random_state=cfg.runtime.random_state))])
        model.fit(X.loc[train_ids], y.loc[train_ids])
        if model['svm'].fit_status_:
            raise ValueError('SVC did not converge')
        write_json(out / 'fit_scope.json', dict(protocol='fixed_svm', representation_fit_ids=train_ids,
            scaler_fit_ids=train_ids, calibration_scope='libsvm internal training-only', holdout_used_for_fitting=False))
        return {'svm': model}, pd.DataFrame([dict(model='svm', protocol='fixed_svm', f1_macro_mean=None)])
    if backend_name(cfg) == 'bulkrnabert' and cfg.foundation.representation != 'tokens':
        return fit_bert(cfg, X, y, train_ids, classes, out)
    return fit_article(cfg, X, y, train_ids, test_ids, classes, cache, out)
