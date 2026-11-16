"""Exact default-path regression against the corrected public starting commit."""
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from omiclass.config.schema import Config
from omiclass.train.trainer import Trainer


BASE_REVISION = "07752936aa02ebbfffa889520667c702edcb02c2"


def original_module(tmp_path, source_path, module_name):
    result = subprocess.run(["git", "show", f"{BASE_REVISION}:{source_path}"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True)
    if result.returncode:
        pytest.skip("Original Git revision unavailable (e.g. source archive)")
    path = tmp_path / (module_name.rsplit(".", 1)[-1] + ".py")
    path.write_bytes(result.stdout)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", ["plain_random_forest", "smote_random_forest", "xgboost", "light_gbm"])
def test_original_trainer_predictions_metrics_artifacts_and_rng_are_exact(tmp_path, variant):
    schema = original_module(tmp_path, "omiclass/config/schema.py", "omiclass.config._pre_pairs_schema")
    cv = original_module(tmp_path, "omiclass/train/cv_engine.py", "omiclass.train._pre_pairs_cv")
    trainer = original_module(tmp_path, "omiclass/train/trainer.py", "omiclass.train._pre_pairs_trainer")
    trainer.nested_oof_cv = cv.nested_oof_cv
    raw = dict(project=dict(meta_path=tmp_path / "meta.csv", cluster_column="target", out_dir=tmp_path),
        runtime=dict(random_state=80, n_jobs=1, cv_folds=3, bayesian_iterations=2,
                     tuning=dict(inner_cv=2), feature_selection=dict(enabled=False)),
        models=dict(random_forest=dict(defaults=dict(calibration=dict(calibrate=True),
            search_space=dict(n_estimators=[8, 12], max_depth=[2, 5], min_samples_split=[2, 4],
                              min_samples_leaf=[1, 2], max_features=["sqrt", .5])))))
    raw["models"]["xgboost"] = dict(param_grid=dict(learning_rate=[.1, .11], n_estimators=[3, 5], max_depth=[2, 3],
        min_child_weight=[1., 1.1], gamma=[0., .1], reg_alpha=[.01, .02], reg_lambda=[1., 1.1],
        subsample=[.9, 1.], colsample_bytree=[.9, 1.], colsample_bylevel=[.9, 1.], colsample_bynode=[.9, 1.]))
    raw["models"]["light_gbm"] = dict(param_grid=dict(learning_rate=[.1, .11], n_estimators=[3, 5],
        num_leaves=[3, 5], max_depth=[2, 3], feature_fraction=[.9, 1.], bagging_fraction=[.9, 1.],
        min_child_samples=[2, 3], bagging_freq=[0, 1], lambda_l1=[.01, .02], lambda_l2=[.01, .02]))
    current_cfg, old_cfg = Config(**raw), schema.Config(**raw)
    assert current_cfg.model_dump(mode="json") == old_cfg.model_dump(mode="json")
    rng = np.random.RandomState(80)
    X = pd.DataFrame(rng.lognormal(size=(48, 12)), columns=[f"g{i}" for i in range(12)])
    y = pd.Series([0, 1]*24)
    X.loc[y == 1, "g0"] += 2
    outputs = []
    for cls, cfg, label in [(trainer.Trainer, old_cfg, "old"), (Trainer, current_cfg, "new")]:
        np.random.seed(12345)
        instance = cls(cfg, tmp_path / label)
        if variant.endswith("forest"):
            row, path = instance.run_rf_variant(variant, X, y, "model.joblib")
        else:
            model_conf = getattr(cfg.models, variant)
            row, path = instance.run_boosting(variant, X, y, "model.joblib", model_conf)
        model = joblib.load(path)
        outputs.append((row, instance.cv_results[variant], model.predict(X), model.predict_proba(X),
                        np.random.get_state(), hashlib.sha256(path.read_bytes()).hexdigest()))
    old, new = outputs
    pd.testing.assert_frame_equal(pd.DataFrame([old[0]]), pd.DataFrame([new[0]]))
    assert old[1].per_fold_metrics == new[1].per_fold_metrics
    assert [vars(r) for r in old[1].fold_recipes] == [vars(r) for r in new[1].fold_recipes]
    np.testing.assert_array_equal(old[1].oof_pred, new[1].oof_pred)
    np.testing.assert_array_equal(old[1].oof_proba, new[1].oof_proba)
    for a, b in zip(old[2:4], new[2:4]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(old[4], new[4]):
        np.testing.assert_array_equal(a, b)
    assert old[5] == new[5]
