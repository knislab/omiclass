import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from omiclass.features.bulkformer_reduction import (
    BulkFormerRepresentation,
    SupervisedEmbeddingSelector,
    benjamini_hochberg,
)


def test_binary_selector_uses_both_signed_welch_directions():
    X = pd.DataFrame({
        "positive": [5, 6, 4, 0, 1, 0],
        "negative": [0, 1, 0, 5, 6, 4],
        "constant": [1, 1, 1, 1, 1, 1],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    selector = SupervisedEmbeddingSelector(n_dimensions=2).fit(X, y)

    assert selector.selected_features_ == ["positive", "negative"]
    report = selector.selection_report_.set_index("dimension")
    assert report.loc["positive", "direction"] == "positive"
    assert report.loc["negative", "direction"] == "negative"
    assert report.loc["positive", "t_statistic"] > 0
    assert report.loc["negative", "t_statistic"] < 0


def test_ranking_is_t_statistic_not_mean_difference_or_pvalue():
    # large_effect has much higher variance; stable has the larger Welch t.
    X = pd.DataFrame({
        "large_effect": [20, -10, 20, 0, 0, 0],
        "stable": [4, 5, 6, 0, 1, 2],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])
    selector = SupervisedEmbeddingSelector(n_dimensions=1).fit(X, y)
    assert selector.selected_features_ == ["stable"]
    assert "mean_difference" in selector.selection_report_.columns
    assert "p_value" in selector.selection_report_.columns
    assert "fdr_adjusted_p_value" in selector.selection_report_.columns


def test_bh_is_reportable_and_monotone():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(adjusted[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted[3])


def test_multiclass_selector_finds_close_pairs_and_respects_budget():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(np.vstack([
        rng.normal(0, .1, (5, 6)),
        rng.normal(.2, .1, (5, 6)),
        rng.normal(5, .1, (5, 6)),
    ]), columns=[f"bulkformer_feature_{i:04d}" for i in range(6)])
    y = pd.Series(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    selector = SupervisedEmbeddingSelector(
        n_dimensions=4, close_pair_count=1, random_state=7
    ).fit(X, y)
    assert len(selector.selected_features_) == 4
    assert selector.selected_close_pairs_[0][:2] == ("a", "b")
    assert selector.one_vs_rest_budget_ + selector.close_pair_budget_ == 4


def test_representation_schema_and_joblib_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(20, 8)), columns=[f"bulkformer_feature_{i:04d}" for i in range(8)])
    y = pd.Series([0] * 10 + [1] * 10)
    model = Pipeline([
        ("bulkformer_representation", BulkFormerRepresentation(
            strategy="supervised_pca", supervised_dimensions=5,
            pca_components=3, random_state=2,
        )),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=2)),
    ]).fit(X, y)
    before = model.predict_proba(X)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert np.allclose(before, loaded.predict_proba(X))
    with pytest.raises(ValueError, match="schema mismatch"):
        loaded.predict(X.drop(columns=X.columns[0]))
