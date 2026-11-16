"""Small native HTML extension shared by classifier comparison workflows."""
from omiclass.reporting.components import card, kv_table


def append_comparison_summary(out, board, tests, selected, scoring):
    cv_columns = ['model', 'classifier', 'accuracy_mean', 'f1_macro_mean',
                  'precision_macro_mean', 'recall_macro_mean', 'selection_score_mean']
    test_columns = ['model', 'n_test', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    html = out/'final_report.html'
    extra = card('CV-selected final model', kv_table(dict(model=selected, scoring=scoring,
        selection_source='training_outer_cv', holdout_used_for_selection=False)))
    extra += card('Training CV — every classifier', board[cv_columns].to_html(index=False, classes='kv', escape=True))
    extra += card('Holdout test — every classifier', tests[test_columns].to_html(index=False, classes='kv', escape=True))
    html.write_text(html.read_text().replace('</main>', extra+'</main>', 1))
