import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from omiclass.models.rf_smote import (
    build_rf_adasyn,
    build_rf_bl_smote,
    build_rf_smote,
    build_rf_svm_smote,
)
from omiclass.train.cv_engine import (
    _feature_rank_by_importance,
    calibrate_within_training,
)
from omiclass.train.trainer import Trainer


def _classification_data():
    rng = np.random.RandomState(80)
    X = pd.DataFrame(
        rng.normal(size=(24, 6)), columns=[f"gene_{i}" for i in range(6)]
    )
    y = pd.Series([0, 1] * 12)
    splits = [(np.arange(4, 24), np.arange(4))]
    return X, y, splits


def test_smote_family_feature_ranking_falls_back_to_all_de_features():
    X, y, splits = _classification_data()
    builders = (
        build_rf_smote,
        build_rf_adasyn,
        build_rf_svm_smote,
        build_rf_bl_smote,
    )

    for builder in builders:
        ranked = _feature_rank_by_importance(
            lambda builder=builder: builder(random_state=80, n_jobs=1),
            X,
            y,
            {},
            splits,
        )
        assert ranked is None


def test_non_sampler_pipeline_feature_ranking_remains_available():
    X, y, splits = _classification_data()

    ranked = _feature_rank_by_importance(
        lambda: SklearnPipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=5, random_state=80)),
            ]
        ),
        X,
        y,
        {},
        splits,
    )

    assert sorted(ranked.tolist()) == list(range(X.shape[1]))


def test_cv_and_final_random_forest_calibration_use_ensemble():
    X, y, _ = _classification_data()
    calibrated_cv = calibrate_within_training(
        fitted_or_builder=RandomForestClassifier(n_estimators=5, random_state=80),
        Xtr_used=X,
        ytr=y,
        method="sigmoid",
        cv=3,
        seed=80,
    )

    trainer = Trainer.__new__(Trainer)
    calibrated_final = trainer._fit_final(
        lambda: RandomForestClassifier(n_estimators=5, random_state=80),
        X,
        y,
        calibrate=True,
        method="sigmoid",
        cv=3,
        seed=80,
    )

    assert calibrated_cv.ensemble is True
    assert calibrated_final.ensemble is True


def test_all_smote_family_builders_propagate_framework_seed():
    builders = (
        build_rf_smote,
        build_rf_adasyn,
        build_rf_svm_smote,
        build_rf_bl_smote,
    )

    for builder in builders:
        pipeline = builder(random_state=80, n_jobs=1)
        assert pipeline.named_steps["sampler"].random_state == 80
        assert pipeline.named_steps["rf"].random_state == 80
