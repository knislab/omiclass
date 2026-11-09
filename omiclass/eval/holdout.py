from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss

from .prediction import canonical_classes, match_features, predict_estimator, align_probabilities
from .thresholds import apply_multiclass_thresholds, ordered_thresholds


def _match_features_to_model(X, model):
    return match_features(X, model)


def _predict_and_proba(model, X):
    return predict_estimator(model, X)


def _metrics(y_true, y_pred, proba, classes):
    y_true, y_pred, classes = np.asarray(y_true), np.asarray(y_pred), np.asarray(classes)
    if y_true.shape != y_pred.shape or not len(y_true) or not set(y_true).issubset(set(classes)):
        raise ValueError('Evaluation labels differ from the fitted class schema or sample count')
    auroc, ll = np.nan, np.nan
    if proba is not None:
        # sklearn log_loss and multiclass AUROC sort their label encodings.
        # Reorder a metric-only copy; public probability columns stay canonical.
        sorted_classes = np.sort(classes)
        safe = np.clip(align_probabilities(proba, classes, sorted_classes), 1e-15, 1 - 1e-15)
        safe = safe / safe.sum(axis=1, keepdims=True)
        ll = float(log_loss(y_true, safe, labels=sorted_classes))
        if set(y_true) == set(classes):
            if len(classes) == 2:
                auroc = float(roc_auc_score(y_true == classes[1], np.asarray(proba)[:, 1]))
            else:
                auroc = float(roc_auc_score(y_true, safe, labels=sorted_classes, multi_class='ovr', average='macro'))
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    return dict(n_test=int(len(y_true)), classes=classes.tolist(),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0)),
        auroc_ovr_macro=auroc, logloss=ll, report=report,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        plot_payload=dict(labels=[str(label) for label in classes], y_pred=y_pred, y_proba=proba))


def evaluate_holdout(model, X_test, y_test):
    X = match_features(X_test, model)
    pred, proba = predict_estimator(model, X)
    return _metrics(y_test, pred, proba, canonical_classes(model))


def evaluate_holdout_with_thresholds(model, X_test, y_test, threshold_info):
    X = match_features(X_test, model)
    _, proba = predict_estimator(model, X)
    if proba is None:
        raise ValueError('Threshold evaluation requires fitted predict_proba probabilities')
    classes = canonical_classes(model)
    thresholds = ordered_thresholds(threshold_info, classes)
    pred = apply_multiclass_thresholds(proba, thresholds, classes)
    result = _metrics(y_test, pred, proba, classes)
    result['thresholds'] = threshold_info
    return result


def _per_class_rows(report_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for cls, stats in report_dict.items():
        if cls in ("accuracy", "micro avg", "macro avg", "weighted avg"):
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
