# omiclass/pipeline/steps/survival_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.survival.predict import run_survival_prediction
from omiclass.errors.errors import SurvivalPredictionError
from omiclass.utils.io import load_matrix
import json

class SurvivalStep:
    name = "survival_analysis"

    def should_run(self, ctx: RunContext) -> bool:
        if ctx.opts.mode == "clean_and_predict":
            return True
        sp = ctx.cfg.runtime.survival_prediction
        return bool(sp and sp.enabled)

    def run(self, ctx: RunContext) -> None:
        """
        Run survival analysis prediction and store results in ctx.cache["survival_results"].
        """
        try:
            final_test_res_path = Path(ctx.out_dir) / "test_metrics.csv"
            surv_cfg = ctx.cfg.runtime.survival_prediction
            surv_path = Path(ctx.cfg.project.meta_path)
            time_col = surv_cfg.time_col
            event_col = surv_cfg.event_col
            time_scale = float(surv_cfg.time_scale)

            X = None
            if ctx.opts.mode == "clean_and_predict":
                X = ctx.cache.get("X_recorrected", None)
            else:
                X = ctx.out_dir / "train_holdout" / "holdout_expression.csv"
                # check if X_path exists; if not, fall back to surv_cfg.X_path if both none dont do survival prediction and say enabled is true but no patient. and continue 
                if not X .exists():
                    if surv_cfg.X_path is not None:
                        X = Path(surv_cfg.X_path)
                    else:
                        print(
                            "[WARN] Survival prediction enabled but no valid expression data found; skipping survival prediction."
                        )
                        return
                    
            final_test_res = load_matrix(final_test_res_path, numeric=False)
            model_path = Path(final_test_res.iloc[0]["artifact"])
            labels_for_plot = ctx.cache.get("labels_for_plot", None)

            # optional threshold-optimized survival report
            model_name = model_path.stem
            report_json = ctx.out_dir / "reports" / f"{model_name}.report.json"

            if report_json.exists():
                data = json.loads(report_json.read_text(encoding="utf-8"))
                threshold_block = data.get("threshold_optimized_holdout") or {}
                threshold_info = threshold_block.get("thresholds")

                # Only use threshold info if it improves F1
                if threshold_info:
                    original_f1 = data.get("test", {}).get("metrics", {}).get("f1_macro", 0)
                    threshold_f1 = threshold_block.get("metrics", {}).get("f1_macro", 0)

                    if threshold_f1 > original_f1:
                        survival_results_threshold = run_survival_prediction(
                            X=X,
                            model_path=model_path,
                            survival_path=surv_path,
                            time_col=time_col,
                            event_col=event_col,
                            time_scale=time_scale,
                            artifacts_dir=ctx.out_dir / "survival_prediction",
                            ctx=ctx,
                            class_names=labels_for_plot,
                            threshold_info=threshold_info,
                        )
                        ctx.cache["survival_results"] = survival_results_threshold
                    else:
                        survival_results = run_survival_prediction(
                            X=X,
                            model_path=model_path,
                            survival_path=surv_path,
                            time_col=time_col,
                            event_col=event_col,
                            time_scale=time_scale,
                            artifacts_dir=ctx.out_dir / "survival_prediction",
                            ctx=ctx,
                            class_names=labels_for_plot,
                        )
                        ctx.cache["survival_results"] = survival_results
                else:
                    survival_results = run_survival_prediction(
                        X=X,
                        model_path=model_path,
                        survival_path=surv_path,
                        time_col=time_col,
                        event_col=event_col,
                        time_scale=time_scale,
                        artifacts_dir=ctx.out_dir / "survival_prediction",
                        ctx=ctx,
                        class_names=labels_for_plot,
                    )
                    ctx.cache["survival_results"] = survival_results
            else:
                survival_results = run_survival_prediction(
                    X=X,
                    model_path=model_path,
                    survival_path=surv_path,
                    time_col=time_col,
                    event_col=event_col,
                    time_scale=time_scale,
                    artifacts_dir=ctx.out_dir / "survival_prediction",
                    ctx=ctx,
                    class_names=labels_for_plot,
                )

                ctx.cache["survival_results"] = survival_results
        except SurvivalPredictionError as e:
            print(f"Survival prediction failed: {e}")
            ctx.cache["survival_results"] = None