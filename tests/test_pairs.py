from collections import Counter
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from skopt.space import Categorical

from omiclass.features.pairs import OvRTSPSelector, fitted_pair_selectors, prepare_pair_prediction
from omiclass.config.schema import Config
from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.train.cv_engine import nested_oof_cv, _safe_stratified_kfold
from omiclass.train.trainer import Trainer


def data(n_classes=2, per_class=12):
    rng = np.random.RandomState(11)
    y = pd.Series(np.repeat(np.arange(n_classes), per_class))
    X = pd.DataFrame(rng.uniform(1, 2, (len(y), n_classes * 4)),
                     columns=[f"g{i}" for i in range(n_classes * 4)])
    for cls in range(n_classes):
        X.loc[y == cls, f"g{cls * 2}"] += 5
        X.loc[y != cls, f"g{cls * 2 + 1}"] += 5
    return X, y


def config(tmp_path, method="pairs", **runtime):
    options = dict(cv_folds=2, bayesian_iterations=1, tuning=dict(inner_cv=2),
                   feature_selection=dict(enabled=False), feature_reduction=dict(enabled=False))
    options.update(runtime)
    return Config(project=dict(meta_path=tmp_path / "meta.csv", cluster_column="target", out_dir=tmp_path),
        marker_selection=dict(method=method, g=4, k=3, max_gene_reuse=2),
        batch_correction=dict(algorithm="none"),
        runtime=options,
        models=dict(random_forest=dict(defaults=dict(calibration=dict(calibrate=False),
            search_space=dict(n_estimators=[5, 5], max_depth=[3, 3], min_samples_split=[2, 2],
                              min_samples_leaf=[1, 1], max_features=["sqrt"])))))


def test_binary_scoring_ties_and_orientation():
    X = pd.DataFrame({"a": [8, 8, 8, 1, 2, 3], "b": [1, 2, 3, 8, 8, 8]})
    selector = OvRTSPSelector(g=1, k=1).fit(X, pd.Series([0]*3 + [1]*3))
    assert selector.selected_pairs_ == [("a", "b")]
    assert list(selector.candidate_genes_) == [0]  # one comparison only
    assert selector.selection_report_.iloc[0].score == 1
    external = pd.DataFrame({"a": [2, 1, 1, 0], "b": [1, 2, 1, 0]})
    assert selector.transform(external).iloc[:, 0].tolist() == [1, -1, 0, 0]
    assert selector.transform(external).dtypes.iloc[0] == np.int8
    near = pd.DataFrame({"a": [np.nextafter(1., 2.)], "b": [1.]})
    assert selector.transform(near).iat[0, 0] == 1  # strict comparison, no epsilon


@pytest.mark.parametrize("n_classes", [2, 3, 12])
def test_multiclass_balanced_score_reuse_and_union(n_classes):
    X, y = data(n_classes, 4)
    # Deliberately imbalance one class, preserving ordering frequencies.
    X = pd.concat([X, X.loc[y == 0].set_axis(range(len(X), len(X) + 4))])
    y = pd.concat([y, pd.Series([0]*4, index=X.index[-4:])])
    selector = OvRTSPSelector(g=5, k=4, max_gene_reuse=2).fit(X, y)
    keys = [frozenset(p) for p in selector.selected_pairs_]
    assert len(keys) == len(set(keys))
    report = selector.selection_report_
    assert set(report.target_class) == ({0} if n_classes == 2 else set(range(n_classes)))
    for cls, sub in report.groupby("target_class"):
        assert len(sub) <= 4
        assert max(Counter(sub.gene_a.tolist() + sub.gene_b.tolist()).values()) <= 2
        for row in sub.itertuples():
            freq = (X[row.gene_a] > X[row.gene_b]).groupby(y).mean()
            assert row.score == pytest.approx(freq[cls] - freq.drop(cls).mean())


def test_reversed_pairs_are_eliminated_after_per_class_selection():
    X = pd.DataFrame({"a": [9, 8, 1, 2, 5, 5], "b": [1, 2, 9, 8, 5, 5]})
    selector = OvRTSPSelector(g=1, k=1).fit(X, pd.Series([0, 0, 1, 1, 2, 2]))
    assert selector.selected_pairs_ == [("a", "b")]
    assert selector.selection_report_.orientation.tolist() == [1, -1]
    assert selector.selection_report_.feature.nunique() == 1


def test_sample_independence_monotonicity_column_order_and_clone():
    X, y = data(3)
    selector = OvRTSPSelector(g=4, k=4).fit(X, y)
    expected = selector.transform(X)
    pd.testing.assert_frame_equal(expected.loc[[4]], selector.transform(X.loc[[4]]))
    pd.testing.assert_frame_equal(expected, selector.transform(X[X.columns[::-1]]))
    scales = np.linspace(.3, 2, len(X))[:, None]
    shifted = pd.DataFrame(np.exp(X.to_numpy() * scales) + np.arange(len(X))[:, None],
                           index=X.index, columns=X.columns)
    pd.testing.assert_frame_equal(expected, selector.transform(shifted))
    assert clone(selector).get_params() == selector.get_params()
    assert not hasattr(clone(selector), "selected_pairs_")


def test_selection_is_deterministic_and_consumes_no_random_numbers():
    X, y = data(3)
    before = np.random.get_state()
    selector = OvRTSPSelector(g=4, k=4).fit(X, y)
    reordered = OvRTSPSelector(g=4, k=4).fit(X[X.columns[::-1]], y)
    pd.testing.assert_frame_equal(selector.selection_report_, reordered.selection_report_)
    selector.transform(X)
    for a, b in zip(before, np.random.get_state()):
        np.testing.assert_array_equal(a, b)


def test_missing_nonfinite_duplicate_and_unused_genes():
    X, y = data()
    selector = OvRTSPSelector(g=2, k=1).fit(X, y)
    with pytest.raises(ValueError, match="Missing genes"):
        selector.transform(X.drop(columns=selector.required_genes_[0]))
    required = X[selector.required_genes_]
    pd.testing.assert_frame_equal(selector.transform(X), selector.transform(required))
    with pytest.raises(ValueError, match="finite"):
        selector.transform(required.assign(**{selector.required_genes_[0]: np.nan}))
    with pytest.raises(ValueError, match="unique"):
        selector.fit(pd.concat([X, X.iloc[:, [0]]], axis=1), y)
    with pytest.raises(ValueError, match="indices"):
        selector.fit(X, y[::-1])


def test_expression_filter_is_fitted_only_on_training():
    X, y = data()
    X["heldout_only"] = 0.
    X["low"] = .01
    selector = OvRTSPSelector(g=100, k=3, min_expression=.1).fit(X, y)
    candidates = selector.candidate_genes_[0]
    assert "heldout_only" not in candidates["up"] + candidates["down"]
    assert "low" not in candidates["up"] + candidates["down"]
    before = selector.selection_report_.copy()
    selector.transform(X.assign(heldout_only=10000))
    pd.testing.assert_frame_equal(before, selector.selection_report_)


def test_inner_outer_cv_and_calibration_refit_selector_without_leakage(monkeypatch):
    X, y = data(per_class=24)
    calls = []
    original = OvRTSPSelector.fit

    def spy(self, X, y):
        calls.append(frozenset(X.index))
        return original(self, X, y)

    monkeypatch.setattr(OvRTSPSelector, "fit", spy)
    builder = lambda: Pipeline([("pairs", OvRTSPSelector(g=3, k=2)),
                               ("classifier", RandomForestClassifier(n_estimators=5, random_state=11))])
    result = nested_oof_cv(model_builder=builder, X=X, y=y, outer_splits=3, seed=11,
        tune_space={"pairs__k": Categorical([1, 2]), "classifier__max_depth": Categorical([2, 3])},
        tune_iter=2, tune_scoring="f1_macro", tune_inner_cv=2, n_jobs=1, feature_cfg=None,
        eval_calibrated_in_cv=True, calib_method="sigmoid", calib_cv=2, pair_features=True)
    expected = set()
    for tr, va in _safe_stratified_kfold(y, 3, 11).split(X, y):
        train = X.iloc[tr]
        expected.add(frozenset(train.index))
        for inner_tr, inner_va in _safe_stratified_kfold(y.iloc[tr], 2, 11).split(train, y.iloc[tr]):
            expected.add(frozenset(train.iloc[inner_tr].index))
    assert calls and all(indices in expected for indices in calls)
    assert {len(indices) for indices in calls} == {16, 32}
    assert all(m["pair_selection"] and m["n_features"] < X.shape[1] for m in result.per_fold_metrics)
    assert len(result.oof_pred) == len(X)


@pytest.mark.parametrize("calibrate,ensemble", [(False, False), (True, False), (True, True)])
def test_trainer_serialization_and_external_prediction(tmp_path, calibrate, ensemble):
    X, y = data()
    cfg = config(tmp_path)
    cfg.models.random_forest.defaults.calibration.calibrate = calibrate
    cfg.models.random_forest.defaults.calibration.ensemble = ensemble
    cfg.marker_selection.search_space = {"k": [1, 2]}
    trainer = Trainer(cfg, tmp_path)
    row, path = trainer.run_rf_variant("plain_random_forest", X, y, "rf.joblib")
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(X), trainer.final_models["plain_random_forest"].predict_proba(X))
    assert loaded._omiclass_pair_features
    selectors = fitted_pair_selectors(loaded)
    assert len(selectors) == (2 if ensemble else 1)
    assert len(list(tmp_path.glob("rf.pairs_*.csv"))) == len(selectors)
    required = list(dict.fromkeys(g for s in selectors for g in s.required_genes_))
    ext = X.loc[[0], required]
    np.testing.assert_array_equal(loaded.predict(ext), loaded.predict(X.loc[[0]]))
    prepare_pair_prediction(loaded, ext)
    from omiclass.eval.holdout import evaluate_holdout
    assert evaluate_holdout(loaded, ext, y.loc[ext.index])["n_test"] == len(ext)
    with pytest.raises(ValueError, match="Missing genes"):
        prepare_pair_prediction(loaded, ext.drop(columns=required[0]))
    assert row["accuracy_mean"] > .9


def test_config_and_runtime_are_strictly_opt_in(tmp_path):
    cfg = config(tmp_path, method="de")
    assert "marker_selection" not in cfg.model_dump(mode="json")
    builder = lambda: RandomForestClassifier(random_state=11)
    trainer = Trainer(cfg, tmp_path)
    assert isinstance(trainer._with_bulkformer_reduction(builder)(), RandomForestClassifier)
    space = {"max_depth": Categorical([2, 3])}
    assert trainer._reduction_space(space) == space
    cfg = config(tmp_path)
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / "tpm.csv", {})
    assert opts.mode == "train-only" and not opts.run_de and not opts.run_batch
    with pytest.raises(ValueError, match="normalized"):
        create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / "counts.csv", {"raw_count": True})
    from omiclass.pipeline.steps.de_step import DEStep
    opts.run_de = True
    assert not DEStep().should_run(SimpleNamespace(cfg=cfg, opts=opts))
    raw = cfg.model_dump(mode="json")
    raw["batch_correction"]["algorithm"] = "limma"
    with pytest.raises(ValueError, match="algorithm='none'"):
        Config.model_validate(raw)
    with pytest.raises(ValueError, match="feature_selection"):
        config(tmp_path, feature_selection=dict(enabled=True))


def test_boosting_models_receive_only_pair_features(tmp_path):
    from omiclass.models.registry import get_builder
    X, y = data(3)
    trainer = Trainer(config(tmp_path), tmp_path)
    for key in ("xgboost", "light_gbm", "weighted_xgboost", "weighted_light_gbm", "smote_random_forest"):
        builder = get_builder(key, random_state=11, n_jobs=1, override_params={"n_estimators": 5})
        model = trainer._with_bulkformer_reduction(builder)().fit(X, y)
        assert set(np.unique(model.named_steps["pairs"].transform(X))) <= {-1, 0, 1}
        assert len(model.predict(X)) == len(y)


def test_pipeline_smoke_and_prediction_only_roundtrip(tmp_path):
    """Use real orchestrator, model routing, reporting, holdout, and external path."""
    from omiclass.config.loaders import freeze_config
    from omiclass.config.schema import xgboostParamGrid, LightGBMParamGrid
    from omiclass.pipeline.runner import run_pipeline

    X, y = data(per_class=16)
    X.index = y.index = pd.Index([f"s{i}" for i in range(len(X))])
    cfg = config(tmp_path)
    cfg.models.xgboost.param_grid = xgboostParamGrid(learning_rate=[.1, .11], n_estimators=[3, 3], max_depth=[2, 2],
        min_child_weight=[1., 1.1], gamma=[0., .1], reg_alpha=[.01, .02], reg_lambda=[1., 1.1],
        subsample=[.9, 1.], colsample_bytree=[.9, 1.], colsample_bylevel=[.9, 1.], colsample_bynode=[.9, 1.])
    cfg.models.light_gbm.param_grid = LightGBMParamGrid(learning_rate=[.1, .11], n_estimators=[3, 3],
        num_leaves=[3, 3], max_depth=[2, 2], feature_fraction=[.9, 1.], bagging_fraction=[.9, 1.],
        min_child_samples=[2, 2], bagging_freq=[0, 0], lambda_l1=[.01, .02], lambda_l2=[.01, .02])
    pd.DataFrame({"target": y}).to_csv(cfg.project.meta_path)
    expression = tmp_path / "tpm.csv"
    X.T.to_csv(expression)
    freeze_config(cfg, tmp_path)
    result = run_pipeline(expression, output_dir=str(tmp_path))
    leaderboard = pd.read_csv(result.leaderboard_csv)
    assert set(leaderboard.model) == {"plain_random_forest", "xgboost", "light_gbm"}
    assert set(leaderboard.status) == {"ok"}
    assert result.combined_test_csv.exists()
    assert list((result.out_dir / "train_artifacts").glob("*.pairs_0.csv"))
    ext = X.iloc[:6].copy()
    ext.index = [f"external_{i}" for i in range(len(ext))]
    external_path = tmp_path / "external.csv"
    ext.T.to_csv(external_path)
    # Prediction works even when metadata has no external labels.
    repeat = run_pipeline(external_path, output_dir=str(result.out_dir))
    assert repeat.out_dir == result.out_dir
    predictions = pd.read_csv(result.out_dir / "survival_prediction/model_predictions.csv", index_col=0)
    assert predictions.index.tolist() == ext.index.tolist()


@pytest.mark.parametrize("key", ["weighted_xgboost", "weighted_light_gbm"])
def test_multiclass_weighted_boosting_trainer(tmp_path, key):
    from omiclass.config.schema import xgboostParamGrid, LightGBMParamGrid
    X, y = data(3)
    cfg = config(tmp_path)
    cfg.models.xgboost.param_grid = xgboostParamGrid(learning_rate=[.1, .11], n_estimators=[3, 3], max_depth=[2, 2],
        min_child_weight=[1., 1.1], gamma=[0., .1], reg_alpha=[.01, .02], reg_lambda=[1., 1.1],
        subsample=[.9, 1.], colsample_bytree=[.9, 1.], colsample_bylevel=[.9, 1.], colsample_bynode=[.9, 1.])
    cfg.models.light_gbm.param_grid = LightGBMParamGrid(learning_rate=[.1, .11], n_estimators=[3, 3],
        num_leaves=[3, 3], max_depth=[2, 2], feature_fraction=[.9, 1.], bagging_fraction=[.9, 1.],
        min_child_samples=[2, 2], bagging_freq=[0, 0], lambda_l1=[.01, .02], lambda_l2=[.01, .02])
    model_conf = cfg.models.xgboost if "xgboost" in key else cfg.models.light_gbm
    trainer = Trainer(cfg, tmp_path)
    row, path = trainer.run_boosting(key, X, y, "boost.joblib", model_conf)
    assert np.isfinite(row["accuracy_mean"])
    loaded = joblib.load(path)
    assert loaded.predict_proba(X).shape == (len(X), 3)
    np.testing.assert_allclose(loaded.predict_proba(X).sum(axis=1), 1., atol=3 * np.finfo(np.float32).eps)
    assert fitted_pair_selectors(loaded)


def test_existing_short_override_meanings_are_preserved():
    from omiclass.config.overrides import normalize_user_overrides
    assert normalize_user_overrides({"search_space": {"n_estimators": [3, 5]}}) == {
        "models": {"random_forest": {"defaults": {"search_space": {"n_estimators": [3, 5]}}}}}
    assert normalize_user_overrides({"marker_method": "pairs", "pair_g": 30}) == {
        "marker_selection": {"method": "pairs", "g": 30}}




@pytest.mark.parametrize("input_scale,log_transform", [("tpm", True), ("log1p_tpm", True), ("tpm", False)])
def test_explicit_input_scale_on_training_and_scoped_prediction(tmp_path, input_scale, log_transform):
    from omiclass.data.processor import DataProcessor
    from omiclass.features.pairs import pair_input_expression
    from omiclass.pipeline.steps.rebatch_step import RebatchCorrectionStep
    import json

    X, y = data()
    cfg = config(tmp_path)
    cfg.batch_correction.input_scale = input_scale
    cfg.runtime.log_transform = log_transform
    expected = np.log1p(X) if input_scale == "tpm" and log_transform else X
    pd.testing.assert_frame_equal(pair_input_expression(X, cfg), expected)
    # A public saved scoped run must still apply the pair scale once.
    (tmp_path / "preprocessing").mkdir()
    (tmp_path / "preprocessing/scoped.json").write_text(json.dumps(dict(correction_transductive=False)))
    ctx = SimpleNamespace(cfg=cfg, out_dir=tmp_path, cache=dict(full_expr=X.T))
    RebatchCorrectionStep().run(ctx)
    pd.testing.assert_frame_equal(ctx.cache["X_recorrected"].T, expected)
    # No float32 cast, imputation or constant filtering before selector fitting.
    X["constant"] = 0.
    pd.DataFrame({"target": y}).to_csv(cfg.project.meta_path)
    prepared, _ = DataProcessor(cfg).build_xy_from_corrected(X.T, X.columns.tolist())
    pd.testing.assert_frame_equal(prepared, X)


def test_pair_external_required_schema_failures_and_upstream_prep_rejection(tmp_path):
    from omiclass.eval.prediction import match_features
    X, y = data()
    trainer = Trainer(config(tmp_path), tmp_path)
    _, path = trainer.run_rf_variant("plain_random_forest", X, y, "rf.joblib")
    loaded = joblib.load(path)
    required = fitted_pair_selectors(loaded)[0].required_genes_
    with pytest.raises(ValueError, match="Missing genes"):
        match_features(X.drop(columns=required[0]), loaded)
    with pytest.raises(ValueError, match="finite"):
        match_features(X.assign(**{required[0]: np.nan}), loaded)
    with pytest.raises(ValueError, match="upstream preprocessor"):
        Trainer(config(tmp_path), tmp_path, preprocessor=object())
