# generic_classifier/pipeline/steps/feature_reduction_step.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import joblib
import pandas as pd

from generic_classifier.eval.feature_reduction import evaluate_feature_reduction
from generic_classifier.plotting.feature_reduction import plot_feature_reduction_curve

if TYPE_CHECKING:
    from generic_classifier.pipeline.orchestrator import RunContext


class FeatureReductionStep:
    name = "feature_reduction"
    
    def run(self, ctx: RunContext) -> None:
        # ---------- sanity ----------
        if "leaderboard_csv" not in ctx.cache:
            raise RuntimeError("FeatureReductionStep requires leaderboard_csv in ctx.cache")
        if "split_json" not in ctx.cache:
            raise RuntimeError("FeatureReductionStep requires split_json in ctx.cache")

        # ---------- load data ----------
        Xtr = ctx.cache["X_train"]
        ytr = ctx.cache["y_train"]

        # ---------- pick top model ----------
        model_path = ctx.cache["best_model_path"]
        best_model_name = ctx.cache["best_model"]
        if model_path is None:
            raise RuntimeError(f"Artifact for model '{model_path}' not found")

        model = joblib.load(model_path)

        # ---------- run feature reduction ----------
        fr_cfg = ctx.cfg.runtime.feature_reduction

        result = evaluate_feature_reduction(
            estimator=model,
            X=Xtr,
            y=ytr,
            k_ratios=fr_cfg.k_ratios,
            min_features=int(fr_cfg.min_features),
            n_splits=int(ctx.cfg.runtime.cv_folds),
            random_state=int(ctx.cfg.runtime.random_state),
        )

        svg = plot_feature_reduction_curve(
            result,
            title=f"Feature reduction – {best_model_name}",
        )

        # ---------- write artifacts ----------
        out_dir = ctx.out_dir / "feature_reduction"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "feature_reduction.csv"

        n_features = Xtr.shape[1]
        ratios_used = [k / float(n_features) for k in result.k_values]

        df = pd.DataFrame({
            "k": result.k_values,
            "ratio": ratios_used,
            **{f"{m}_mean": result.mean[m] for m in result.mean},
            **{f"{m}_std": result.std[m] for m in result.std},
        })
        df.to_csv(csv_path, index=False)

        # ---------- stash for report ----------
        ctx.cache["feature_reduction_svg"] = svg
        ctx.cache["feature_reduction_csv"] = csv_path
