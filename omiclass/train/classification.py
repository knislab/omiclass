"""Native classifier comparison using existing heads, CV and fitted preprocessing."""
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from omiclass.foundation import heads
from omiclass.foundation.io import write_json
from omiclass.train.cv_engine import CVResult, nested_oof_cv, calibrate_within_training
from omiclass.train.scoped import PartitionClassifier
from omiclass.train.selection import selection_summary
from omiclass.reporting.per_model_report import report_cv


class ArticleMLPClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable adapter around the existing fixed-epoch article head training."""
    def __init__(self, cfg, work_dir=None):
        self.cfg = cfg
        self.work_dir = work_dir

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.fit_ids_ = X.index.tolist()
        cfg = self.cfg.model_copy(deep=True)
        cfg.foundation.model = 'bulkformer'
        cfg.foundation.classifier = 'mlp'
        cfg.foundation.representation = 'full'
        cfg.foundation.article_head = cfg.classification.mlp.model_copy(deep=True)
        cfg.bulkformer.reduction.strategy = 'full'
        cfg.bulkformer.reduction.random_state = cfg.runtime.random_state
        labels = pd.Series(np.searchsorted(self.classes_, np.asarray(y)), index=X.index)
        names = list(map(str, self.classes_))
        root = Path(self.work_dir) if self.work_dir is not None else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        # Existing head IO/training is reused; only its portable fitted state is
        # retained. Temporary expression files are not deployment dependencies.
        from omiclass.utils.reproducibility import random_state_scope
        with random_state_scope(), TemporaryDirectory(prefix='native-mlp-', dir=root) as temporary:
            out = Path(temporary)
            fitted, _ = heads.fit_article(cfg, X, labels, self.fit_ids_, [], names, None, out)
            self.model_ = fitted['article_mlp']
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def _lr_result(out, ids, classes):
    saved = pd.read_csv(out/'oof_predictions.csv', dtype={'sample_id': str}, index_col=0).reindex(ids)
    return CVResult(oof_true=saved.y_true.to_numpy(), oof_pred=saved.y_pred.to_numpy(dtype=int),
        oof_proba=saved[[f'probability_{i}' for i in classes]].to_numpy(), classes=np.asarray(classes),
        per_fold_metrics=pd.read_csv(out/'fold_metrics.csv').to_dict('records'), fold_recipes=[])


def run_comparison(trainer, X, y):
    """Only training data enters this boundary. Holdouts belong to TrainAndEvalStep."""
    from omiclass.train.trainer import TrainSummary
    from omiclass.foundation.multiclassifier import _folds_from_oof
    cfg, r = trainer.cfg, trainer.cfg.runtime
    requested = cfg.classification.requested
    if 'omiclass' in requested:
        native = trainer._run_omiclass(X, y)
        board, paths = native.metrics.copy(), dict(native.artifact_paths)
        if not board.status.eq('ok').all():
            raise RuntimeError('A native Omiclass candidate failed; comparison is incomplete')
        board['classifier'] = 'omiclass'
        board['protocol'] = 'existing_native_omiclass'
    else:
        trainer.final_models.clear(); trainer.cv_results.clear(); trainer.builders.clear()
        board, paths = pd.DataFrame(), {}
    artifacts = trainer.artifacts_dir
    out = artifacts.parent
    classes = np.unique(y)

    def preprocess():
        return clone(trainer.preprocessor) if trainer.preprocessor is not None else None

    for classifier in requested:
        if classifier == 'omiclass':
            continue
        key = 'logistic_regression' if classifier == 'lr' else classifier
        if classifier == 'lr':
            directory = artifacts/'classifier_training/lr'
            directory.mkdir(parents=True, exist_ok=False)
            local = cfg.model_copy(deep=True)
            local.foundation.lr_c_values = cfg.classification.lr_c_values
            def pipeline():
                return heads.lr_pipeline(r.random_state, preprocessing_step=preprocess())
            models, _ = heads.fit_lr(local, X, y, directory, pipeline_factory=pipeline)
            model = models[key]
            cv = _lr_result(directory, X.index.tolist(), classes)
            protocol = 'existing_nested_cv_lr_with_native_fold_preprocessing'
        else:
            def builder():
                base = (heads.svm_pipeline(r.random_state, probability=False) if classifier == 'svm'
                    else ArticleMLPClassifier(cfg, artifacts/'classifier_training/mlp-tmp'))
                return PartitionClassifier(base, preprocess(), seed=r.random_state,
                    inner_cv=r.tuning.inner_cv or 3)
            svm = classifier == 'svm'
            cv = nested_oof_cv(model_builder=builder, X=X, y=y,
                outer_splits=r.cv_folds or 5, seed=r.random_state,
                tune_space={}, tune_iter=0, tune_scoring=r.scoring,
                tune_inner_cv=r.tuning.inner_cv or 3, n_jobs=r.n_jobs, feature_cfg=None,
                eval_calibrated_in_cv=svm, calib_method='sigmoid', calib_cv=r.cv_folds or 5,
                calibration_ensemble=False)
            model = (calibrate_within_training(fitted_or_builder=builder(), Xtr_used=X, ytr=y,
                method='sigmoid', cv=r.cv_folds or 5, seed=r.random_state, ensemble=False)
                if svm else builder().fit(X, y))
            protocol = ('fixed_rbf_svm_with_fold_local_sigmoid_calibration' if svm
                        else 'existing_article_mlp_fixed_epochs')
        path = artifacts/f'{key}.joblib'
        joblib.dump(model, path)
        paths[key], trainer.final_models[key], trainer.cv_results[key] = path, model, cv
        row = dict(model=key, classifier=classifier, protocol=protocol, status='ok', error='',
                   **selection_summary(cv.per_fold_metrics, r.scoring))
        board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)

    folds, oof = [], []
    for key, cv in trainer.cv_results.items():
        if not np.array_equal(cv.oof_true, np.asarray(y)) or not np.isfinite(cv.oof_proba).all():
            raise ValueError('Incomplete or misaligned genuine training-only CV predictions')
        metrics = _folds_from_oof(cfg, y, cv.oof_pred)
        if len(metrics) != len(cv.per_fold_metrics):
            raise ValueError('Candidate CV fold count differs from the common comparison folds')
        for original, descriptive in zip(cv.per_fold_metrics, metrics):
            original.update(descriptive)
        for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
            values = [row[metric] for row in cv.per_fold_metrics]
            board.loc[board.model.eq(key), metric+'_mean'] = float(np.mean(values))
            board.loc[board.model.eq(key), metric+'_std'] = float(np.std(values, ddof=1))
        family = board.loc[board.model.eq(key), 'classifier'].iloc[0]
        folds.append(pd.DataFrame(cv.per_fold_metrics).assign(model=key, classifier=family))
        table = pd.DataFrame(dict(sample_id=X.index, model=key, classifier=family,
            y_true=cv.oof_true, y_pred=cv.oof_pred))
        from omiclass.eval.prediction import align_probabilities
        probabilities = align_probabilities(cv.oof_proba, cv.classes, classes)
        for i, label in enumerate(classes):table[f'probability_{label}'] = probabilities[:, i]
        oof.append(table)
        from omiclass.reporting.per_model_report import report_json_path
        original_report = report_json_path(out, key)
        if family == 'omiclass' and original_report.is_file():
            import json
            doc = json.loads(original_report.read_text())
            doc['cv']['per_fold'] = cv.per_fold_metrics
            for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
                values = [row[metric] for row in cv.per_fold_metrics]
                doc['cv']['metrics_mean'][metric] = float(np.mean(values))
                doc['cv']['metrics_std'][metric] = float(np.std(values))
            write_json(original_report, doc)
            continue
        report_cv(cvres=cv, artifacts_dir=artifacts, model_name=key,
            estimator_cls_name=type(trainer.final_models[key]).__name__, X_shape=X.shape,
            y_series=y, seed=r.random_state,
            cfg_short=dict(classifier=family, protocol='training_only_native_comparison', scoring=r.scoring),
            class_name_map=dict(enumerate(cfg.project.classes_names or list(map(str,classes)))),
            extra_metrics=['precision_macro', 'recall_macro'])
    pd.concat(folds, ignore_index=True).to_csv(out/'cv_metrics.csv', index=False)
    pd.concat(oof, ignore_index=True).to_csv(out/'oof_predictions.csv', index=False)
    joblib.dump(trainer.cv_results, out/'cv_results.joblib')
    write_json(out/'classification.json', dict(classifiers=requested,
        model_classifiers=dict(zip(board.model, board.classifier)),
        models={key:str(path.relative_to(out)) for key,path in paths.items()},
        selection_source='training_outer_cv', scoring=r.scoring,
        protocols=dict(zip(board.model, board.get('protocol', pd.Series(dtype=str))))))
    return TrainSummary(board, paths)
