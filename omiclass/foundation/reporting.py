"""Native Omiclass report adapters for a shared-extraction classifier comparison."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from omiclass.eval.holdout import _metrics, write_holdout_csvs
from omiclass.reporting.per_model_report import report_cv, finalize_report_for_model
from omiclass.reporting.final_summary_report import render_final_summary_html
from omiclass.reporting.components import card, kv_table, esc
from omiclass.reporting.sidebar_finalize import finalize_unified_sidebar
from omiclass.plotting.plotter import FinalSummaryPlotter


def write_comparison_reports(cfg, out, X_train, y_train, y_test, classes, models,
                             board, selected, payloads, comparison):
    artifacts = out / 'train_artifacts'
    artifacts.mkdir(exist_ok=True)
    class_names = dict(enumerate(classes))
    per_train = {}
    for key, model in models.items():
        cv = comparison['cv_results'][key]
        per_train[key] = dict(y_pred=cv.oof_pred, y_proba=cv.oof_proba)
        report_cv(cvres=cv, artifacts_dir=artifacts, model_name=key,
            estimator_cls_name=type(model).__name__, X_shape=X_train.shape,
            y_series=y_train, seed=cfg.runtime.random_state,
            cfg_short=dict(foundation=cfg.foundation.model_dump(mode='json'),
                classifier=comparison['model_classifiers'][key], selection='training_outer_cv'),
            class_name_map=class_names, extra_metrics=['precision_macro', 'recall_macro'])
        payload = payloads[key]
        metrics = _metrics(y_test, payload['y_pred'], payload['y_proba'], np.arange(len(classes)))
        write_holdout_csvs(artifacts, key, metrics=metrics)
        table = artifacts / 'tables' / f'{key}_test_metrics.csv'
        values = pd.read_csv(table)
        values['precision_macro'] = metrics['report']['macro avg']['precision']
        values['recall_macro'] = metrics['report']['macro avg']['recall']
        values.to_csv(table, index=False)
        finalize_report_for_model(out, key, test_metrics_csv=table,
            per_class_csv=artifacts / 'tables' / f'{key}_per_class.csv',
            model_rel_path=f'../models/{key}.joblib', cfg=cfg)
    plotter = FinalSummaryPlotter(labels=[str(i) for i in range(len(classes))],
        y_true=y_test, per_model=payloads, per_model_train=per_train,
        y_true_train=y_train, out_dir=out, class_name_map=class_names)
    context = SimpleNamespace(cache=dict(best_model=selected, leaderboard_csv=out / 'leaderboard.csv',
                                          combined_test_csv=out / 'test_metrics.csv'))
    html = render_final_summary_html(plotter=plotter, ctx=context,
        meta=dict(encoder=cfg.foundation.model, representation=cfg.foundation.representation,
            classifiers=cfg.foundation.classifiers, seed=cfg.runtime.random_state,
            selection_scoring=cfg.runtime.scoring, selection_source='training outer CV only',
            correction_transductive=cfg.foundation.correction != 'none'))
    cv_columns = ['model', 'classifier', 'accuracy_mean', 'f1_macro_mean',
                  'precision_macro_mean', 'recall_macro_mean', 'selection_score_mean']
    test = pd.read_csv(out / 'test_metrics.csv')
    test_columns = ['model', 'n_test', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    extra = card('CV-selected final model', kv_table(dict(model=selected,
        classifier=comparison['model_classifiers'][selected], scoring=cfg.runtime.scoring,
        selection_source='training_outer_cv', holdout_used_for_selection=False)))
    extra += card('Training CV — every classifier', board[cv_columns].to_html(index=False, classes='kv', escape=True))
    extra += card('Holdout test — every classifier', test[test_columns].to_html(index=False, classes='kv', escape=True))
    extra += card('Classifier reports', '<ul>' + ''.join(
        f'<li><a href="reports/{esc(key)}.html">{esc(key)}</a></li>' for key in models) + '</ul>')
    html.write_text(html.read_text().replace('</main>', extra + '</main>', 1))
    finalize_unified_sidebar(out)
