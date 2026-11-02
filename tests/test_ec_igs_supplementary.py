"""Precision and source-figure checks for the EC/IGS package."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'final_results_runs'))
import build_ec_igs_supplementary as mod


def test_csv_roundtrip_preserves_missing_tiny_values_and_identifiers(tmp_path):
    frame = pd.DataFrame({'value': [0.12345678901234568, np.nan, np.inf, 1e-320],
                          'sample_id': ['001', 'NA', '=SUM(A1)', 'x']})
    path = mod.Builder(tmp_path).csv('values.csv', frame)
    assert '001' in path.read_text()
    assert '9.9998886718268301e-321' in path.read_text()


def test_pdf_matches_exact_model_name_and_rejects_changed_metric(monkeypatch):
    text = '''SMOTE RF 0.791 0.741 0.710 0.834
SVM SMOTE RF (AUC=0.954) SVM SMOTE RF (AP=0.841)
BL-SMOTE RF (AUC=0.947) BL-SMOTE RF (AP=0.812)
SMOTE RF (AUC=0.950) SMOTE RF (AP=0.836)
'''
    monkeypatch.setattr(mod.subprocess, 'check_output', lambda *a, **kw: text)
    frame = pd.DataFrame([dict(model='SMOTE RF', accuracy=.791, macro_f1=.741,
        macro_recall=.710, macro_precision=.834, macro_roc_auc=.950,
        macro_average_precision=.836)])
    mod.validate_pdf(Path('reference.pdf'), frame)
    frame.loc[0, 'macro_roc_auc'] = .951
    with pytest.raises(ValueError, match='AUC mismatch'):
        mod.validate_pdf(Path('reference.pdf'), frame)
