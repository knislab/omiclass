# omiclass/pipeline/steps/train_eval_step.py
from __future__ import annotations

from pathlib import Path
from pprint import pp
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from omiclass.utils.pipeline import make_or_load_split, maybe_log1p_expression
from omiclass.data.processor import DataProcessor
from omiclass.train.trainer import Trainer
from omiclass.eval.holdout import evaluate_holdout, evaluate_holdout_with_thresholds, write_holdout_csvs
from omiclass.eval.thresholds import learn_multiclass_thresholds
from omiclass.reporting.per_model_report import finalize_report_for_model
from omiclass.plotting.plotter import FinalSummaryPlotter
from omiclass.reporting.final_summary_report import render_final_summary_html
from omiclass.utils.toolbox import _build_class_name_map
from omiclass.eval.feature_reduction_runner import FeatureReductionStep
from omiclass.utils.pipeline import apply_batch_holdout
from omiclass.train.tune import resolve_cv_folds
from omiclass.config.loaders import freeze_config


if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext


class TrainAndEvalStep:
    name = "train_and_eval"
    
    @staticmethod
    def _is_higher_better_metric(metric_name: str) -> bool:
        m = str(metric_name or "").lower()
        if m.startswith("neg_"):
            return True
        lower_better_tokens = ("loss", "error", "rmse", "mae", "mse")
        return not any(tok in m for tok in lower_better_tokens)

    def should_run(self, ctx: RunContext) -> bool:
        if ctx.opts.mode == "clean_and_predict":
            return False
        if ctx.opts.run_training is not None:
            return ctx.opts.run_training
        return ctx.opts.mode in ("full", "train-only", "de-train", "batch-train")

    def run(self, ctx: RunContext) -> None:
        dp = DataProcessor(ctx.cfg)

        # Choose which expression matrix to use: corrected vs original
        if "X_corrected" in ctx.cache:
            X_full_raw = ctx.cache["X_corrected"]
        else:
            X_full_raw = ctx.cache["full_expr"]

        gene_list = ctx.cache["gene_list"]

        X_full_raw, did_log, log_reason = maybe_log1p_expression(
            X_full_raw,
            enabled=ctx.cfg.runtime.log_transform is True,
            already_normalized_or_logged=ctx.cache.get("_did_raw_count_norm", False)
            or ctx.cache.get("_did_log_transform", False),
        )
        if did_log:
            ctx.cache["_did_log_transform"] = True
        ctx.cache["_log_transform_reason"] = log_reason

        X_full_raw = apply_batch_holdout(
            X=X_full_raw,
            cfg=ctx.cfg,
            out_dir=ctx.out_dir,
        )

        # Build X, y for training
        X, y = dp.build_xy_from_corrected(
            X_full_raw=X_full_raw,
            gene_list=gene_list,
        )
        resolve_cv_folds(ctx.cfg, y)

        # Train/test split
        Xtr, Xte, ytr, yte, split_path, created = make_or_load_split(
            X=X,
            y=y,
            split_dir=ctx.split_out,
            cfg=ctx.cfg,
            reuse_split=ctx.opts.reuse_split,
        )
        # Train on training set
        train_dir = ctx.out_dir / "train_artifacts"
        trainer = Trainer(ctx.cfg, artifacts_dir=train_dir)
        summary = trainer.run(Xtr, ytr)
        # 4) Save leaderboard
        leaderboard_csv = ctx.out_dir / "leaderboard.csv"
        summary.metrics.to_csv(leaderboard_csv, index=False)

        print(f"[ 5][Train_and_eval]: Leaderboard saved at: {leaderboard_csv}")
        # 5) Evaluate all final models on test
        test_rows: list[Dict[str, Any]] = []
        test_reports: Dict[str, Path] = {}
        per_model_payload: Dict[str, Dict[str, np.ndarray]] = {}
        per_model_train_payload: Dict[str, Dict[str, np.ndarray]] = {}
        per_model_threshold_payload: Dict[str, Dict[str, np.ndarray]] = {}
        original_f1_scores: Dict[str, float] = {}
        threshold_f1_scores: Dict[str, float] = {}
        labels_from_first: list[str] = []
        train_label_index = np.array(sorted(np.unique(ytr.astype(int))), dtype=int)

        # Train payload from OOF CV outputs (same source used to create summary.metrics).
        for model_name in summary.metrics.get("model", pd.Series(dtype=str)).tolist():
            cvres = trainer.cv_results.get(model_name)
            if cvres is None:
                continue
            y_pred = np.asarray(getattr(cvres, "oof_pred", None))
            if y_pred.size == 0:
                continue
            y_proba = getattr(cvres, "oof_proba", None)
            if y_proba is None:
                # Fallback for estimators without predict_proba in CV: one-hot from predictions.
                y_proba = (y_pred[:, None] == train_label_index[None, :]).astype(float)
            per_model_train_payload[model_name] = {
                "y_pred": y_pred,
                "y_proba": np.asarray(y_proba),
            }

        i = 1
        for key, model_path in summary.artifact_paths.items():
            i += 1
            run_dir = ctx.out_dir
            model = joblib.load(model_path)

            report = evaluate_holdout(model, Xte, yte)
            cvres = trainer.cv_results.get(key)
            threshold_info = learn_multiclass_thresholds(cvres) if cvres is not None else None
            threshold_report = None
            if threshold_info is not None:
                threshold_report = evaluate_holdout_with_thresholds(
                    model=model,
                    X_test=Xte,
                    y_test=yte,
                    threshold_info=threshold_info,
                )

            paths = write_holdout_csvs(
                train_dir,
                key,
                metrics=report,
                threshold_metrics=threshold_report,
            )

            finalize_report_for_model(
                run_dir=run_dir,
                model_name=key,
                test_metrics_csv=run_dir / paths["test_metrics_csv"],
                per_class_csv=run_dir / paths["per_class_csv"],
                model_rel_path=ctx.out_dir / f"train_artifacts/{key}.joblib",
                cfg=ctx.cfg,
                threshold_metrics=threshold_report,

            )

            # regular final summary payload
            pp = report.get("plot_payload", {})
            if not labels_from_first:
                labels_from_first = list(pp.get("labels", []))
            per_model_payload[key] = {
                "y_pred": pp.get("y_pred"),
                "y_proba": pp.get("y_proba"),
            }

            # Track original F1 score
            original_f1_scores[key] = report.get("f1_macro")

            # threshold final summary payload (only if it improves F1)
            if threshold_report is not None:
                threshold_f1_scores[key] = threshold_report.get("f1_macro")
                tpp = threshold_report.get("plot_payload", {})
                per_model_threshold_payload[key] = {
                    "y_pred": tpp.get("y_pred"),
                    "y_proba": tpp.get("y_proba"),
                }

            test_rows.append({
                "model": key,
                "artifact": str(model_path),
                "n_test": report.get("n_test"),
                "accuracy": report.get("accuracy"),
                "f1_macro": report.get("f1_macro"),
                "f1_weighted": report.get("f1_weighted"),
                "threshold_accuracy": threshold_report.get("accuracy") if threshold_report else np.nan,
                "threshold_f1_macro": threshold_report.get("f1_macro") if threshold_report else np.nan,
                "threshold_f1_weighted": threshold_report.get("f1_weighted") if threshold_report else np.nan,
                "threshold_auroc_ovr_macro": threshold_report.get("auroc_ovr_macro") if threshold_report else np.nan,
                "threshold_logloss": threshold_report.get("logloss") if threshold_report else np.nan,
                "threshold_f1_macro_delta": (
                    threshold_report.get("f1_macro") - report.get("f1_macro")
                    if threshold_report and report.get("f1_macro") is not None
                    else np.nan
                ),
                "mean_true_class_proba": report.get("mean_true_class_proba", np.nan),
            })

        combined_test_csv = ctx.out_dir / "test_metrics.csv"
        final_test_res = pd.DataFrame(test_rows)
        final_test_res.to_csv(combined_test_csv, index=False)

        # Select best model ONLY from CV leaderboard (never from holdout ranking).
        scoring_metric = str(
            getattr(getattr(ctx.cfg.runtime, "tuning", None), "scoring_override", None)
            or ctx.cfg.runtime.scoring
        )
        cv_metric_col = f"{scoring_metric}_mean"
        fallback_cv_cols = ["f1_macro_mean", "accuracy_mean"]

        cv_leaderboard = summary.metrics.copy()
        if "status" in cv_leaderboard.columns:
            cv_leaderboard = cv_leaderboard[cv_leaderboard["status"] == "ok"]
        cv_leaderboard = cv_leaderboard[
            cv_leaderboard["model"].isin(summary.artifact_paths.keys())
        ]

        selected_metric_col = None
        if cv_metric_col in cv_leaderboard.columns and cv_leaderboard[cv_metric_col].notna().any():
            selected_metric_col = cv_metric_col
        else:
            for col in fallback_cv_cols:
                if col in cv_leaderboard.columns and cv_leaderboard[col].notna().any():
                    selected_metric_col = col
                    break
        if selected_metric_col is None:
            raise RuntimeError(
                "Could not select best model from CV leaderboard: no usable CV metric column found."
            )

        ascending = not self._is_higher_better_metric(selected_metric_col.replace("_mean", ""))
        cv_leaderboard = cv_leaderboard.sort_values(
            by=selected_metric_col,
            ascending=ascending,
            kind="mergesort",
        )
        best_model = str(cv_leaderboard.iloc[0]["model"])
        best_model_path = summary.artifact_paths[best_model]

        labels_for_plot = (
            labels_from_first
            if labels_from_first
            else [str(i) for i in sorted(np.unique(yte.astype(int)))]
        )

        class_name_map = _build_class_name_map(y, ctx.cfg)
        
        plotter = FinalSummaryPlotter(
            labels=labels_for_plot,
            y_true=yte,
            per_model=per_model_payload,
            out_dir=ctx.out_dir,
            class_name_map=class_name_map,
            per_model_train=per_model_train_payload,
            y_true_train=ytr,
        )

        ctx.cache["split_json"] = split_path
        ctx.cache["leaderboard_csv"] = leaderboard_csv
        ctx.cache["best_model"] = best_model
        ctx.cache["best_model_path"] = best_model_path
        
        fr = getattr(ctx.cfg.runtime, "feature_reduction", None)
        try:
            if bool(fr and fr.enabled):
                ctx.cache["X_train"] = Xtr
                ctx.cache["y_train"] = ytr
                fr_class = FeatureReductionStep()
                fr_class.run(ctx)
        except Exception as e:
            pass

        # Check if best model's threshold optimization improves F1
        best_model_improves_with_threshold = (
            best_model in threshold_f1_scores and 
            threshold_f1_scores[best_model] > original_f1_scores[best_model]
        )

        threshold_plotter = None
        if per_model_threshold_payload and best_model_improves_with_threshold:
            threshold_plotter = FinalSummaryPlotter(
                labels=labels_for_plot,
                y_true=yte,
                per_model=per_model_threshold_payload,
                out_dir=ctx.out_dir,
                class_name_map=class_name_map,
                per_model_train=per_model_train_payload,
                y_true_train=ytr,
            )

        _ = render_final_summary_html(
            plotter=plotter,
            threshold_plotter=threshold_plotter,
            include_threshold_appendix=bool(threshold_plotter is not None),
            meta={
                "train samples": int(len(ytr)),
                "test samples": int(len(yte)),
                "Number of classes": len(labels_for_plot),
                "Selected model": best_model.replace("_", " ").title(),
                "Selected metric": selected_metric_col,
                "Threshold optimized appendix": "yes" if threshold_plotter is not None else "no",
            },
            ctx=ctx,
        )

        # stash for orchestrator result
        ctx.cache["combined_test_csv"] = combined_test_csv
        ctx.cache["labels_for_plot"] = list(labels_for_plot)

        freeze_config(ctx.cfg, ctx.out_dir)
