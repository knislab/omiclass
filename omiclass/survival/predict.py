# omiclass/survival/predict.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd

from omiclass.config.schema import SurvivalPredictionCfg
from omiclass.utils.io import load_survival_matrix, load_matrix
from omiclass.survival.plotter import plot_confidence_histogram, plot_confidence_violin, plot_km_and_forest
from omiclass.survival.report import render_survival_prediction_html
from omiclass.errors.errors import SurvivalPredictionError
from omiclass.utils.debug import debug_prediction_and_survival_inputs


@dataclass
class SurvivalPredictionResult:
    prediction: pd.Series
    probabilities: pd.DataFrame
    survival_df: pd.DataFrame
    conf_hist_svg: str
    conf_violin_svg: str
    km_svg: Optional[str]
    cox_forest_svg: Optional[str]
    html_path: Path

def _build_proba_df(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of predicted probabilities (index = samples)."""
    proba = model.predict_proba(X)
    if hasattr(model, "classes_"):
        cols = list(model.classes_)
    else:
        cols = list(range(1, proba.shape[1] + 1))
    proba_df = pd.DataFrame(proba, index=X.index, columns=cols)
    return proba_df

def _predict_with_thresholds(proba_df: pd.DataFrame, threshold_info: Dict[str, Any]) -> pd.Series:
    thresholds = threshold_info.get("thresholds") or threshold_info

    scores = proba_df.copy()
    for cls in scores.columns:
        thr = float(thresholds.get(str(cls), thresholds.get(cls, 0.5)))
        scores[cls] = scores[cls] / max(thr, 1e-12)

    pred = scores.idxmax(axis=1)
    pred.name = "threshold_optimized_prediction"
    return pred

def run_survival_prediction(X: Union[pd.DataFrame, Path, str],
    *,
    model_path: Union[str, Path],
    survival_path: Union[str, Path],
    time_col: str,
    event_col: str,
    time_scale: float,
    artifacts_dir: Union[str, Path],
    ctx: Optional[Any] = None,
    class_names: Optional[list[str]] = None,
    threshold_info: Optional[Dict[str, Any]] = None,
) -> SurvivalPredictionResult:
    
    if class_names is not None:
        class_names = list(class_names)
    
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- load model ---
    model_path = Path(model_path)
    model = joblib.load(model_path)
    model_name = model_path.stem

    # --- load X ---
    if isinstance(X, (str, Path)):
        X_df = load_matrix(Path(X))
    else:
        X_df = X.copy()

    X_df = X_df.T
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is None:
        expected_features = getattr(model, "_feature_names_in", None)
    X_df = X_df.reindex(columns=expected_features)
    X_df = X_df.fillna(0.0)
    
    # --- predictions ---
    proba_df = _build_proba_df(model, X_df)

    # regular prediction always stays the main prediction
    y_pred = model.predict(X_df)
    y_pred_series = pd.Series(y_pred, index=X_df.index, name="prediction")

    # optional threshold-optimized prediction for extra survival plot
    threshold_pred_series = None
    if threshold_info is not None:
        threshold_pred_series = _predict_with_thresholds(proba_df, threshold_info)
        threshold_pred_series.index = X_df.index
        threshold_pred_series.name = "threshold_optimized_prediction"
    # --- plots: confidence ---
    conf_hist_svg = plot_confidence_histogram(
        y_proba_df=proba_df,
        model_name=model_name,
    )
    conf_violin_svg = plot_confidence_violin(
        y_proba_df=proba_df,
        model_name=model_name,
        class_names=class_names,
        cfg=ctx.cfg if ctx is not None else None,
    )

    km_svg = None
    cox_forest_svg = None
    km_stats: Dict[str, Any] = {}
    merged = None
    threshold_km_svg = None
    threshold_cox_forest_svg = None
    threshold_km_stats: Dict[str, Any] = {}
    try:
        # --- survival metadata ---
        survival_path = Path(survival_path)
        survival_df = load_survival_matrix(survival_path, X_df.index, time_col=time_col, event_col=event_col)
        merged = survival_df.merge(
            y_pred_series.to_frame(),
            how="inner",
            left_index=True,
            right_index=True,
            validate="one_to_one",
        )
        scaled_time_col = f"{time_col}_scaled"
        if time_col not in merged.columns:
            raise KeyError(
                f"time_col '{time_col}' not found in merged dataframe columns: {merged.columns.tolist()}"
            )
        if event_col not in merged.columns:
            raise KeyError(
                f"event_col '{event_col}' not found in merged dataframe columns: {merged.columns.tolist()}"
            )
        merged[scaled_time_col] = merged[time_col].astype(float) / float(time_scale)
        
        if not merged.empty:
            ref_class = None
            if class_names is not None:
                class_names = list(class_names)  
                if "normal" in class_names:
                    ref_class = "normal"
                else:
                    ref_class = class_names[0] if len(class_names) > 0 else None

            km_svg, cox_forest_svg, km_stats = plot_km_and_forest(
                df=merged,
                time_col=scaled_time_col,
                event_col=event_col,
                group_col="prediction",
                time_unit_label="years",
                ref_class=ref_class,
                cfg=ctx.cfg if ctx is not None else None,
            )
            if threshold_pred_series is not None:
                merged_threshold = survival_df.merge(
                    threshold_pred_series.to_frame(),
                    how="inner",
                    left_index=True,
                    right_index=True,
                    validate="one_to_one",
                )

                merged_threshold[scaled_time_col] = (
                    merged_threshold[time_col].astype(float) / float(time_scale)
                )

                if not merged_threshold.empty:
                    threshold_km_svg, threshold_cox_forest_svg, threshold_km_stats = plot_km_and_forest(
                        df=merged_threshold,
                        time_col=scaled_time_col,
                        event_col=event_col,
                        group_col="threshold_optimized_prediction",
                        time_unit_label="years",
                        ref_class=ref_class,
                    )
    except Exception as e:
        pass
    # --- HTML ---
    html_path = render_survival_prediction_html(
        out_dir=artifacts_dir,
        model_name=model_name,
        n_samples=int(len(X_df)),
        conf_hist_svg=conf_hist_svg,
        conf_violin_svg=conf_violin_svg,
        km_svg=km_svg,
        cox_forest_svg=cox_forest_svg,
        ctx=ctx,
        stats=km_stats,
        threshold_km_svg=threshold_km_svg,
        threshold_cox_forest_svg=threshold_cox_forest_svg,
        threshold_stats=threshold_km_stats,
    )
    y_pred_df = y_pred_series.to_frame()
    y_pred_df.columns = ["prediction"]
    y_pred_df["proba"] = proba_df.max(axis=1)
    y_pred_df["class_names"] = y_pred_df["prediction"].map(
        lambda x: class_names[x] if class_names is not None and x < len(class_names) else str(x)
    )
    y_pred_df.to_csv(artifacts_dir / "model_predictions.csv")
    return SurvivalPredictionResult(
        prediction=y_pred_series,
        probabilities=proba_df,
        survival_df=merged,
        conf_hist_svg=conf_hist_svg,
        conf_violin_svg=conf_violin_svg,
        km_svg=km_svg,
        cox_forest_svg=cox_forest_svg,
        html_path=html_path,
    )