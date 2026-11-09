# omiclass/pipeline/steps/survival_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.survival.predict import run_survival_prediction
from omiclass.errors.errors import SurvivalPredictionError
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
                    
            model_key = ctx.cache.get('best_model')
            model_base_path = ctx.cache.get('best_model_path')
            scope_path = ctx.out_dir / 'preprocessing/scoped.json'
            scope = json.loads(scope_path.read_text()) if scope_path.is_file() else {}
            if model_base_path is None and scope.get('model_path'):
                model_key = scope['selected_model']
                model_base_path = (ctx.out_dir / scope['model_path']).resolve()
                if not model_base_path.is_relative_to(ctx.out_dir.resolve()):
                    raise SurvivalPredictionError('Saved selected-model path escapes its run directory')
            if model_base_path is None:
                from omiclass.train.selection import rank_candidates
                import pandas as pd
                filenames = {key: value.filename for key, value in
                             vars(ctx.cfg.models.random_forest.variants).items()}
                filenames.update(xgboost=ctx.cfg.models.xgboost.filename,
                    weighted_xgboost=ctx.cfg.models.xgboost.weighted_filename,
                    light_gbm=ctx.cfg.models.light_gbm.filename,
                    weighted_light_gbm=ctx.cfg.models.light_gbm.weighted_filename,
                    stacked_ensemble=ctx.cfg.models.stacked_ensemble.filename)
                available = {key: ctx.out_dir / 'train_artifacts' / filename
                             for key, filename in filenames.items() if filename is not None
                             and (ctx.out_dir / 'train_artifacts' / filename).is_file()}
                if model_key is None:
                    board = pd.read_csv(ctx.out_dir / 'leaderboard.csv')
                    ranked, _ = rank_candidates(board, available, ctx.cfg.runtime.scoring)
                    model_key = str(ranked.iloc[0]['model'])
                model_base_path = available[model_key]
            model_base_path = Path(model_base_path)
            model_key = model_key or model_base_path.stem
            thresholds = ctx.out_dir / 'train_artifacts/thresholds.json'
            if thresholds.is_file():
                threshold_info = json.loads(thresholds.read_text()).get(model_key)
            else:
                # Historical report layout remains readable without consulting
                # its holdout scores to decide whether learned thresholds apply.
                report_json = ctx.out_dir / 'reports' / f'{model_key}.report.json'
                data = json.loads(report_json.read_text()) if report_json.is_file() else {}
                threshold_info = (data.get('threshold_optimized_holdout') or {}).get('thresholds')
            result = run_survival_prediction(X=X, model_path=model_base_path,
                survival_path=surv_path, time_col=time_col, event_col=event_col,
                time_scale=time_scale, artifacts_dir=ctx.out_dir / 'survival_prediction',
                ctx=ctx, class_names=ctx.cache.get('labels_for_plot'), threshold_info=threshold_info)
            ctx.cache['survival_results'] = result
        except SurvivalPredictionError:
            ctx.cache["survival_results"] = None
            raise
