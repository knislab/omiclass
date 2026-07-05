from __future__ import annotations

from typing import Any


MODEL_DISPLAY_NAMES = {
    "plain_random_forest": "Plain Random Forest",
    "weighted_random_forest": "Weighted Random Forest",
    "smote_random_forest": "SMOTE Random Forest",
    "adasyn_random_forest": "ADASYN Random Forest",
    "svm_smote_random_forest": "SVM-SMOTE Random Forest",
    "bl_smote_random_forest": "Borderline-SMOTE Random Forest",
    "light_gbm": "LightGBM",
    "weighted_light_gbm": "Weighted LightGBM",
    "xgboost": "XGBoost",
    "weighted_xgboost": "Weighted XGBoost",
    "stacked_ensemble": "Stacked Ensemble",
}
def _model_key(value: Any) -> str:
    text = str(value).replace("_", " ").replace("-", " ").strip().casefold()
    return " ".join(text.split())


MODEL_DISPLAY_BY_KEY = {
    _model_key(raw): display
    for raw, display in MODEL_DISPLAY_NAMES.items()
}
MODEL_DISPLAY_BY_KEY.update(
    {
        _model_key(display): display
        for display in MODEL_DISPLAY_NAMES.values()
    }
)


def display_model_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return MODEL_DISPLAY_NAMES.get(text) or MODEL_DISPLAY_BY_KEY.get(_model_key(text)) or text.replace("_", " ").title()