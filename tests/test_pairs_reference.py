"""Compare with the retained development snapshot, without its old checkout."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from omiclass.features.pairs import OvRTSPSelector

REFERENCE_REVISION = "2ef206af43fe0673f65d0708d8075568eb23827c"


@pytest.fixture
def reference_selector(tmp_path):
    source = subprocess.run(
        ["git", "show", f"{REFERENCE_REVISION}:omiclass/features/pairs.py"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True)
    if source.returncode:
        pytest.skip("Archived development revision unavailable in exported source")
    path = tmp_path / "reference_pairs.py"
    path.write_bytes(source.stdout)
    spec = importlib.util.spec_from_file_location("reference_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OvRTSPSelector


@pytest.mark.parametrize("n_classes", [2, 3, 12])
@pytest.mark.parametrize("reuse", [1, 2])
def test_exact_reference_candidates_scores_orientation_features_and_predictions(reference_selector, n_classes, reuse):
    rng = np.random.RandomState(20261116)
    counts = [8] + [4] * (n_classes - 1)
    y = pd.Series(np.repeat(np.arange(n_classes), counts), name="class")
    X = pd.DataFrame(rng.uniform(1, 3, (len(y), n_classes * 4)),
                     columns=[f"gene_{i:02d}" for i in range(n_classes * 4)])
    for cls in range(n_classes):
        X.loc[y == cls, f"gene_{2 * cls:02d}"] += 5
        X.loc[y != cls, f"gene_{2 * cls + 1:02d}"] += 5
    X["constant"], X["low"] = 0., .001
    options = dict(g=4, k=3, max_gene_reuse=reuse, min_expression=.1, min_fraction=.25)
    expected, actual = reference_selector(**options).fit(X, y), OvRTSPSelector(**options).fit(X, y)
    assert actual.candidate_genes_ == expected.candidate_genes_
    assert actual.selected_pairs_ == expected.selected_pairs_
    assert actual.required_genes_ == expected.required_genes_
    pd.testing.assert_frame_equal(actual.selection_report_, expected.selection_report_)
    external = X.iloc[:5].copy()
    # Exact ties and a one-ULP difference exercise signed features.
    a, c = actual.selected_pairs_[0]
    external.loc[0, a] = external.loc[0, c]
    external.loc[1, a] = np.nextafter(external.loc[1, c], np.inf)
    external = external[external.columns[::-1]]
    pd.testing.assert_frame_equal(actual.transform(external), expected.transform(external))
    models = [Pipeline([("pairs", selector(**options)),
                        ("classifier", RandomForestClassifier(n_estimators=7, random_state=80))]).fit(X, y)
              for selector in (reference_selector, OvRTSPSelector)]
    np.testing.assert_array_equal(models[0].predict(external), models[1].predict(external))
    np.testing.assert_array_equal(models[0].predict_proba(external), models[1].predict_proba(external))
