from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from omiclass.train.cv_engine import _bulkformer_representation_from_model
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss

from .thresholds import apply_multiclass_thresholds
from ..utils.toolbox import as_numpy_y as _as_numpy_y

def _match_features_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Ensures that X has the features expected by the model,
    reordering columns as needed and adding missing columns with zeros.
    """
    from omiclass.train.scoped import fitted_partitions
    if getattr(model, '_stack_schema_version_', None) == 1:
        return X  # Each stack base and passthrough validates its own fitted schema.
    if fitted_partitions(model):
        return X  # New fitted preprocessors validate and order their own inputs.
    representation = _bulkformer_representation_from_model(model)
    if representation is not None:
        expected = list(representation.feature_names_in_)
        missing = [feature for feature in expected if feature not in X.columns]
        extra = [feature for feature in X.columns if feature not in set(expected)]
        strict_extra = getattr(representation, "strategy", "full") != "full"
        if missing or (extra and strict_extra):
            raise ValueError(
                f"Embedding schema mismatch; missing={missing[:5]}, extra={extra[:5]}."
            )
        return X.loc[:, expected]

    if not hasattr(model, "feature_names_in_"):
        return X

    model_features = list(model.feature_names_in_)
    X_cols_set = set(X.columns)

    # Identify missing features
    missing_features = [f for f in model_features if f not in X_cols_set]

    # Add missing features with zeros
    for f in missing_features:
        X[f] = 0.0

    # Reorder columns to match model's expected feature order
    X_matched = X[model_features]

    return X_matched

def _predict_and_proba(model, X) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_pred, y_proba) with y_proba shape (n, K).
    - Uses predict_proba if available, else decision_function normalized per row,
      else one-hot of hard predictions (for plotting only).
    - Handles binary (n,), (n,1), (n,2) and aligns columns.
    """
    # hard predictions
    y_pred = model.predict(X)
    if isinstance(y_pred, (list, tuple)):
        y_pred = np.asarray(y_pred)
    if y_pred.dtype.kind in {"U", "S", "O"}:
        try:
            y_pred = y_pred.astype(int)
        except Exception:
            pass

    # probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        D = model.decision_function(X)
        D = np.atleast_2d(D)
        # normalize to [0,1] with per-row sum=1 (for plotting, not calibration)
        D = (D - D.min(axis=1, keepdims=True))
        denom = D.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        y_proba = D / denom
    else:
        # fallback: one-hot from hard predictions
        K_guess = int(np.max(y_pred)) + 1 if y_pred.size else 2
        y_proba = np.eye(K_guess, dtype=float)[y_pred.astype(int)]

    # ensure 2D probs
    if y_proba.ndim == 1:  # binary vector
        y_proba = np.column_stack([1 - y_proba, y_proba])

    return y_pred, np.asarray(y_proba)

# ---------- public API ----------
def evaluate_holdout(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Returns metrics + per-class report + confusion matrix,
    and a 'plot_payload' with y_pred/y_proba/labels for plotting.
    """
    y_true = _as_numpy_y(y_test)
    X_test_matched = _match_features_to_model(X_test, model)
    y_pred, y_proba = _predict_and_proba(model, X_test_matched)

    # harmonize K with label set
    labels_idx = np.array(sorted(np.unique(y_true.astype(int))), dtype=int)
    K = len(labels_idx)

    if hasattr(model, "classes_"):
        model_classes = np.asarray(model.classes_, dtype=int)
        # map each desired label -> column index in model_proba
        col_idx = []
        for lab in labels_idx:
            hits = np.where(model_classes == lab)[0]
            col_idx.append(int(hits[0]) if len(hits) else -1)

        # build aligned proba with zeros for missing labels
        aligned = np.zeros((y_proba.shape[0], len(labels_idx)), dtype=float)
        for j, src in enumerate(col_idx):
            if src != -1 and src < y_proba.shape[1]:
                aligned[:, j] = y_proba[:, src]
        y_proba = aligned

    # fix proba columns → (n, K)
    if y_proba.shape[1] != K:
        if y_proba.shape[1] > K:
            y_proba = y_proba[:, :K]
        else:
            pad = np.zeros((y_proba.shape[0], K - y_proba.shape[1]))
            y_proba = np.hstack([y_proba, pad])

    y_proba_safe = np.asarray(y_proba, dtype=float)
    y_proba_safe = np.clip(y_proba_safe, 1e-15, 1 - 1e-15)
    row_sums = y_proba_safe.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_proba_safe = y_proba_safe / row_sums

    auroc = np.nan
    ll = np.nan
    try:
        if K == 2:
            # binary AUROC expects score/proba for the positive class
            auroc = float(roc_auc_score(y_true, y_proba_safe[:, 1]))
        else:
            # multiclass AUROC (macro OVR)
            auroc = float(roc_auc_score(y_true, y_proba_safe, multi_class="ovr", average="macro"))
    except Exception:
        pass

    try:
        # log_loss needs explicit label order to match proba columns
        ll = float(log_loss(y_true, y_proba_safe, labels=labels_idx))
    except Exception:
        pass

    metrics: Dict[str, Any] = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc_ovr_macro": auroc,
        "logloss": ll,
    }

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    # keep prior shape for backwards compatibility
    metrics["report"] = rep
    metrics["confusion_matrix"] = cm.tolist()

    # NEW: payload for plotting (directly feed into FinalSummaryPlotter)
    metrics["plot_payload"] = {
        "labels": [str(i) for i in labels_idx.tolist()],
        "y_pred": y_pred,           # ndarray (n,)
        "y_proba": y_proba,         # ndarray (n, K)
    }
    return metrics


def evaluate_holdout_with_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Same holdout metrics as evaluate_holdout, but hard predictions are derived
    by applying OOF-learned thresholds to holdout predict_proba output.
    """
    y_true = _as_numpy_y(y_test)
    X_test_matched = _match_features_to_model(X_test, model)
    _, y_proba = _predict_and_proba(model, X_test_matched)

    threshold_classes = np.asarray(threshold_info.get("classes", []), dtype=int)
    threshold_values = threshold_info.get("thresholds", {}) or {}
    if threshold_classes.size == 0:
        raise ValueError("threshold_info is missing classes")

    if hasattr(model, "classes_"):
        model_classes = np.asarray(model.classes_, dtype=int)
    else:
        model_classes = threshold_classes

    aligned = np.zeros((y_proba.shape[0], len(threshold_classes)), dtype=float)
    for j, lab in enumerate(threshold_classes):
        hits = np.where(model_classes == lab)[0]
        if len(hits) and int(hits[0]) < y_proba.shape[1]:
            aligned[:, j] = y_proba[:, int(hits[0])]

    y_proba_safe = np.asarray(aligned, dtype=float)
    y_proba_safe = np.clip(y_proba_safe, 1e-15, 1 - 1e-15)
    row_sums = y_proba_safe.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_proba_safe = y_proba_safe / row_sums

    thresholds = [float(threshold_values[str(cls)]) for cls in threshold_classes]
    y_pred = apply_multiclass_thresholds(y_proba_safe, thresholds, threshold_classes)

    labels_idx = np.array(sorted(np.unique(y_true.astype(int))), dtype=int)
    K = len(labels_idx)

    auroc = np.nan
    ll = np.nan
    try:
        if K == 2 and y_proba_safe.shape[1] == 2:
            pos_col = int(np.where(threshold_classes == labels_idx[1])[0][0])
            auroc = float(roc_auc_score(y_true, y_proba_safe[:, pos_col]))
        else:
            score_cols = []
            for lab in labels_idx:
                hits = np.where(threshold_classes == lab)[0]
                score_cols.append(int(hits[0]))
            auroc = float(roc_auc_score(y_true, y_proba_safe[:, score_cols], multi_class="ovr", average="macro"))
    except Exception:
        pass

    try:
        score_cols = []
        for lab in labels_idx:
            hits = np.where(threshold_classes == lab)[0]
            score_cols.append(int(hits[0]))
        ll = float(log_loss(y_true, y_proba_safe[:, score_cols], labels=labels_idx))
    except Exception:
        pass

    metrics: Dict[str, Any] = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc_ovr_macro": auroc,
        "logloss": ll,
    }

    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    metrics["report"] = rep
    metrics["confusion_matrix"] = cm.tolist()
    metrics["thresholds"] = threshold_info
    metrics["plot_payload"] = {
        "labels": [str(i) for i in labels_idx.tolist()],
        "y_pred": y_pred,
        "y_proba": y_proba_safe,
    }
    return metrics


def _per_class_rows(report_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for cls, stats in report_dict.items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(stats, dict):
            continue
        rows.append({
            "class": cls,
            "precision": float(stats.get("precision", np.nan)),
            "recall": float(stats.get("recall", np.nan)),
            "f1": float(stats.get("f1-score", np.nan)),
            "support": int(stats.get("support", 0)),
        })
    return rows


def write_holdout_csvs(
    run_dir: Path,
    model_name: str,
    *,
    metrics: Dict[str, Any],
    threshold_metrics: Dict[str, Any] | None = None,
) -> dict:
    run_dir = Path(run_dir)
    ####
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # test_metrics.csv
    test_metrics_path = tables_dir / f"{model_name}_test_metrics.csv"
    pd.DataFrame([{
        "n_test": metrics.get("n_test"),
        "accuracy": metrics.get("accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
        "auroc_ovr_macro": metrics.get("auroc_ovr_macro"),
        "logloss": metrics.get("logloss"),
    }]).to_csv(test_metrics_path, index=False)

    # per_class.csv
    per_class_path = tables_dir / f"{model_name}_per_class.csv"
    per_class_rows = _per_class_rows(metrics.get("report", {}))
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

    out = {
        "test_metrics_csv": f"train_artifacts/tables/{test_metrics_path.name}",
        "per_class_csv": f"train_artifacts/tables/{per_class_path.name}",
    }

    if threshold_metrics is not None:
        threshold_metrics_path = tables_dir / f"{model_name}_threshold_optimized_test_metrics.csv"
        pd.DataFrame([{
            "n_test": threshold_metrics.get("n_test"),
            "accuracy": threshold_metrics.get("accuracy"),
            "f1_macro": threshold_metrics.get("f1_macro"),
            "f1_weighted": threshold_metrics.get("f1_weighted"),
            "auroc_ovr_macro": threshold_metrics.get("auroc_ovr_macro"),
            "logloss": threshold_metrics.get("logloss"),
        }]).to_csv(threshold_metrics_path, index=False)

        threshold_per_class_path = tables_dir / f"{model_name}_threshold_optimized_per_class.csv"
        threshold_per_class_rows = _per_class_rows(threshold_metrics.get("report", {}))
        pd.DataFrame(threshold_per_class_rows).to_csv(threshold_per_class_path, index=False)

        threshold_info = threshold_metrics.get("thresholds", {}) or {}
        threshold_values = threshold_info.get("thresholds", {}) or {}
        threshold_classes = threshold_info.get("classes", []) or []
        threshold_rows = [
            {
                "class": cls,
                "threshold": threshold_values.get(str(cls)),
                "method": threshold_info.get("method"),
                "optimized_metric": threshold_info.get("optimized_metric"),
                "baseline_oof_f1": threshold_info.get("baseline_oof_f1"),
                "optimized_oof_f1": threshold_info.get("optimized_oof_f1"),
            }
            for cls in threshold_classes
        ]
        threshold_values_path = tables_dir / f"{model_name}_threshold_optimized_thresholds.csv"
        pd.DataFrame(threshold_rows).to_csv(threshold_values_path, index=False)

        out.update({
            "threshold_test_metrics_csv": f"train_artifacts/tables/{threshold_metrics_path.name}",
            "threshold_per_class_csv": f"train_artifacts/tables/{threshold_per_class_path.name}",
            "threshold_values_csv": f"train_artifacts/tables/{threshold_values_path.name}",
        })

    return out
