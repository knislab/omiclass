"""Compare existing head protocols using training-only out-of-fold predictions.

Encoders are frozen and sample-local (apart from explicitly transductive limma).
Only their unselected embeddings are shared. Each fit owns representation/PCA,
scaling, sampling, calibration, tuning and neural validation exactly as before.
"""
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from omiclass.foundation import heads
from omiclass.foundation.io import write_json
from omiclass.train.selection import fold_selection_score, selection_summary
from omiclass.eval.prediction import align_probabilities


def model_key(classifier, local):
    return f'{classifier}__{local}'


def _folds_from_oof(cfg, y, predicted):
    rows = []
    labels = np.arange(len(np.unique(y)))
    for number, (_, va) in enumerate(heads.safe_cv(y, cfg.runtime.cv_folds or 5, cfg.runtime.random_state).split(y, y), 1):
        actual, pred = y.iloc[va], np.asarray(predicted)[va]
        rows.append(dict(fold=number, accuracy=float(accuracy_score(actual, pred)),
            f1_macro=float(f1_score(actual, pred, labels=labels, average='macro', zero_division=0)),
            precision_macro=float(precision_score(actual, pred, labels=labels, average='macro', zero_division=0)),
            recall_macro=float(recall_score(actual, pred, labels=labels, average='macro', zero_division=0))))
    return rows


def _fixed_protocol_cv(cfg, X, y, classes, cache, out):
    """Outer evaluation wraps the existing fixed SVM/article/BERT fit routines."""
    from omiclass.foundation.runner import _probabilities
    folds, scopes = [], []
    predicted = np.full(len(y), -1, dtype=int)
    probabilities = np.full((len(y), len(classes)), np.nan)
    for number, (tr, va) in enumerate(heads.safe_cv(y, cfg.runtime.cv_folds or 5, cfg.runtime.random_state).split(X, y), 1):
        train, validation = X.index[tr].tolist(), X.index[va].tolist()
        models, _ = heads.fit(cfg, X, y, train, validation, classes, cache, out / f'fold_{number}')
        if len(models) != 1:
            raise ValueError('A standalone head returned multiple fold models')
        model = next(iter(models.values()))
        pred, p = _probabilities(model, X, cache, validation, len(classes))
        predicted[va], probabilities[va] = pred, p
        metrics = _folds_from_oof_subset(y.iloc[va], pred, len(classes))
        metrics.update(fold=number, selection_score=fold_selection_score(cfg.runtime.scoring,
            model, X.iloc[va], y.iloc[va], metrics))
        folds.append(metrics)
        scopes.append(dict(fold=number, fit_ids=train, validation_ids=validation))
    if (predicted < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError('Incomplete out-of-fold standalone predictions')
    write_json(out / 'cv_scope.json', dict(protocol='training_only_outer_cv_of_existing_head',
        classifier=cfg.foundation.classifier, folds=scopes, holdout_used=False,
        final_fit_protocol='unchanged fixed SVM/article epochs or BERT training-only early stopping'))
    return SimpleNamespace(oof_true=y.to_numpy(), oof_pred=predicted,
        oof_proba=probabilities, classes=np.arange(len(classes)), per_fold_metrics=folds)


def _folds_from_oof_subset(y, pred, n_classes):
    return dict(accuracy=float(accuracy_score(y, pred)),
        f1_macro=float(f1_score(y, pred, labels=np.arange(n_classes), average='macro', zero_division=0)),
        precision_macro=float(precision_score(y, pred, labels=np.arange(n_classes), average='macro', zero_division=0)),
        recall_macro=float(recall_score(y, pred, labels=np.arange(n_classes), average='macro', zero_division=0)))


def fit_comparison(cfg, X, y, train_ids, classes, cache, out):
    """No held-out IDs/outcomes are accepted by this fitting/selection boundary."""
    out.mkdir(exist_ok=False)
    X, y = X.loc[train_ids], y.loc[train_ids]
    models, results, families, rows = {}, {}, {}, []
    oof_tables, fold_tables = [], []
    for classifier in cfg.foundation.classifiers:
        local_cfg = cfg.model_copy(deep=True)
        local_cfg.foundation.classifiers = None
        local_cfg.foundation.classifier = classifier
        directory = out / classifier
        directory.mkdir()
        if classifier in {'omiclass', 'lr'}:
            fitted, board = heads.fit(local_cfg, X, y, train_ids, [], classes, cache, directory / 'final')
            if classifier == 'omiclass':
                cv_results = joblib.load(directory / 'final/cv_results.joblib')
            else:
                saved = pd.read_csv(directory / 'final/oof_predictions.csv', dtype={'sample_id': str}, index_col=0).reindex(train_ids)
                cv_results = {'logistic_regression': SimpleNamespace(oof_true=saved.y_true.to_numpy(),
                    oof_pred=saved.y_pred.to_numpy(dtype=int),
                    oof_proba=saved[[f'probability_{i}' for i in range(len(classes))]].to_numpy(),
                    classes=np.arange(len(classes)),
                    per_fold_metrics=pd.read_csv(directory / 'final/fold_metrics.csv').to_dict('records'))}
        else:
            cv = _fixed_protocol_cv(local_cfg, X, y, classes, cache, directory)
            fitted, board = heads.fit(local_cfg, X, y, train_ids, [], classes, cache, directory / 'final')
            if len(fitted) != 1:
                raise ValueError('Standalone classifier did not produce one final model')
            cv_results = {next(iter(fitted)): cv}
        for local, model in fitted.items():
            if local not in cv_results:
                raise ValueError(f'Missing genuine CV results for {classifier}/{local}')
            cv = cv_results[local]
            if not np.array_equal(cv.oof_true, y.to_numpy()) or len(cv.oof_pred) != len(y):
                raise ValueError('CV sample/class schema differs from training')
            key = model_key(classifier, local)
            # Retain the native selection scores; enrich all CV protocols with
            # the same four descriptive metrics computed from their own OOF data.
            enriched = _folds_from_oof(cfg, y, cv.oof_pred)
            if len(enriched) != len(cv.per_fold_metrics):
                raise ValueError('CV fold count differs between comparison protocols')
            for computed, native in zip(enriched, cv.per_fold_metrics):
                descriptive = dict(computed)
                computed.update(native)
                computed.update(descriptive)
                computed['selection_score'] = native.get('selection_score', native.get(cfg.runtime.scoring))
                if computed['selection_score'] is None:
                    raise ValueError('Classifier lacks a training-only CV selection score')
            cv.per_fold_metrics = enriched
            scores = selection_summary(enriched, cfg.runtime.scoring)
            row = dict(model=key, classifier=classifier, local_model=local, status='ok',
                protocol=str(board.loc[board.model.eq(local)].iloc[0].get('protocol', 'native_omiclass_cv')),
                **scores)
            for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
                values = [fold[metric] for fold in enriched]
                row[metric + '_mean'], row[metric + '_std'] = float(np.mean(values)), float(np.std(values, ddof=1))
            models[key], results[key], families[key] = model, cv, classifier
            rows.append(row)
            fold_tables.append(pd.DataFrame(enriched).assign(model=key, classifier=classifier))
            probability = align_probabilities(cv.oof_proba, cv.classes, np.arange(len(classes)))
            table = pd.DataFrame(dict(sample_id=train_ids, model=key, classifier=classifier,
                                      y_true=cv.oof_true, y_pred=cv.oof_pred))
            for i in range(len(classes)):
                table[f'probability_{i}'] = probability[:, i]
            oof_tables.append(table)
    return models, pd.DataFrame(rows), dict(cv_results=results, model_classifiers=families,
        fold_metrics=pd.concat(fold_tables, ignore_index=True),
        oof_predictions=pd.concat(oof_tables, ignore_index=True))
