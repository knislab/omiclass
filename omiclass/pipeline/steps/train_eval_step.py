# omiclass/pipeline/steps/train_eval_step.py
from __future__ import annotations

from pathlib import Path
from pprint import pp
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from omiclass.utils.pipeline import make_or_load_split
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

        # Optional log1p before train
        if ctx.cfg.runtime.log_transform is True and not ctx.cache.get("_did_raw_count_norm", False) and not ctx.cache.get("_did_log_transform", False):
            X_full_raw = np.log1p(X_full_raw)
            ctx.cache["_did_log_transform"] = True

        X_full_raw = apply_batch_holdout(
            X=X_full_raw,
            cfg=ctx.cfg,
            out_dir=ctx.out_dir,
        )

        # 1) Build X, y for ML
        X, y = dp.build_xy_from_corrected(
            X_full_raw=X_full_raw,
            gene_list=gene_list,
        )
        resolve_cv_folds(ctx.cfg, y)

        # 2) Train/test split
        Xtr, Xte, ytr, yte, split_path, created = make_or_load_split(
            X=X,
            y=y,
            split_dir=ctx.split_out,
            cfg=ctx.cfg,
            reuse_split=ctx.opts.reuse_split,
        )
        # 3) Train on train set
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
        per_model_threshold_payload: Dict[str, Dict[str, np.ndarray]] = {}
        labels_from_first: list[str] = []

        i = 1
        for key, model_path in summary.artifact_paths.items():
            i += 1
            run_dir = ctx.out_dir
            model = joblib.load(model_path)

            report = evaluate_holdout(model, Xte, yte)
            paths = write_holdout_csvs(train_dir, key, metrics=report)
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

            # threshold final summary payload
            if threshold_report is not None:
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
                "mean_true_class_proba": report.get("mean_true_class_proba", np.nan),
            })

        combined_test_csv = ctx.out_dir / "test_metrics.csv"
        final_test_res = pd.DataFrame(test_rows).sort_values(
            "f1_macro", ascending=False
        )
        best_model = final_test_res.iloc[0]["model"]
        best_model_path = summary.artifact_paths[best_model]
        final_test_res.to_csv(combined_test_csv, index=False)

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

        _ = render_final_summary_html(
            plotter=plotter,
            meta={"n_test": int(len(yte)), "k": len(labels_for_plot)},
            ctx=ctx
        )

        if per_model_threshold_payload:
            threshold_plotter = FinalSummaryPlotter(
                labels=labels_for_plot,
                y_true=yte,
                per_model=per_model_threshold_payload,
                out_dir=ctx.out_dir,
                class_name_map=class_name_map,
            )

            _ = render_final_summary_html(
                plotter=threshold_plotter,
            meta={
                "n_test": int(len(yte)),
                "k": len(labels_for_plot),
                "evaluation": "threshold optimized",
            },
            ctx=ctx,
            output_name="final_summary_threshold_optimized.html",
        )

        # stash for orchestrator result
        ctx.cache["combined_test_csv"] = combined_test_csv
        ctx.cache["labels_for_plot"] = list(labels_for_plot)

        freeze_config(ctx.cfg, ctx.out_dir)
