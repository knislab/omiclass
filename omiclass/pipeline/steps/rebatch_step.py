# omiclass/pipeline/steps/batch_step.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any
import pandas as pd
import joblib

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.batch.limma_runner import LimmaRunner
from omiclass.reporting.limma_report import write_limma_report
from omiclass.utils.io import load_matrix
from omiclass.data.processor import DataProcessor
from omiclass.utils.splits import load_split
from omiclass.utils.pipeline import maybe_log1p_expression
from omiclass.features.embedding_backend import embedding_enabled


class RebatchCorrectionStep:
    name = "rebatch_correction"

    def should_run(self, ctx: RunContext) -> bool:
        if embedding_enabled(ctx.cfg):
            return False
        if ctx.opts.mode == "clean_and_predict":
            return True
        return False


    def run(self, ctx: RunContext) -> None:
        """
        Run limma batch correction + batch holdout, store result in ctx.cache["X_corrected"].
        """
        full_expr = ctx.cache["full_expr"]

        full_expr, did_log, log_reason = maybe_log1p_expression(
            full_expr,
            enabled=ctx.cfg.runtime.log_transform is True,
            already_normalized_or_logged=ctx.cache.get("_did_raw_count_norm", False)
            or ctx.cache.get("_did_log_transform", False),
        )
        if did_log:
            ctx.cache["_did_log_transform"] = True
        ctx.cache["_log_transform_reason"] = log_reason

        bc = ctx.cfg.batch_correction
        dp = DataProcessor(ctx.cfg)
        meta_df, cov_matrix, labels = dp.load_batch_metadata_prediction(
            X_expr=full_expr
        )
        batch_label_maps = getattr(meta_df, "attrs", {}).get("category_maps", {}) if meta_df is not None else {}
        if bc.batch_column and bc.batch_column in batch_label_maps:
            ctx.cache["batch_label_map"] = batch_label_maps[bc.batch_column]
            ctx.cache["batch_label_names"] = list(batch_label_maps[bc.batch_column].values())


        limma_runner = LimmaRunner()
        limma_out_dir = ctx.out_dir / "limma_re-correction"
        limma_out_dir.mkdir(parents=True, exist_ok=True)
        modality = str(getattr(ctx.cfg.project, "modality", "rnaseq"))
        feature_label_singular = "protein" if modality == "cptac_ms" else "gene"
        feature_label_plural = "proteins" if modality == "cptac_ms" else "genes"
        corrected_matrix_label = (
            "batch-corrected protein abundance"
            if modality == "cptac_ms"
            else "batch-corrected expression"
        )
        limma_out_csv = "limma_corrected_expression.csv"

        limma_run_res = limma_runner.run_limma(
            expr_df=full_expr,
            meta_df=meta_df,
            output_dir=limma_out_dir,
            cov_matrix=cov_matrix,
            batch_col=bc.batch_column or None,
            design_cols=bc.design_columns or None,
            n_pc=bc.n_pcs or 0,
            na_fill=bc.fill_na or None,
            out_csv=limma_out_csv,
            eval_cols=(ctx.cfg.de.covariates or None) if ctx.cfg.PDF else None,
            pc_pvalue_heatmap_max=getattr(ctx.cfg.plots, "pc_pvalue_heatmap_max", None),
            pdf_base_dir=(ctx.out_dir / "pdf_plots" / "base_files") if ctx.cfg.PDF else None,
        )
        if ctx.cfg.PDF and not limma_run_res.get("pdf_pca_base_saved", False):
            raise RuntimeError(
                "PDF export was enabled, but PCA regeneration data was not saved "
                "after batch re-correction."
            )

        X_full_raw = load_matrix(limma_out_dir / limma_out_csv)

        _ = write_limma_report(
            out_dir=limma_out_dir,
            expr_shape=X_full_raw.shape,
            meta_cols=list(meta_df.columns) if meta_df is not None else [],
            batch_col=bc.batch_column or None,
            design_cols=bc.design_columns or None,
            n_pc=bc.n_pcs or 10,
            na_fill=bc.fill_na or None,
            limma_result=limma_run_res,
            feature_label_singular=feature_label_singular,
            feature_label_plural=feature_label_plural,
            corrected_matrix_label=corrected_matrix_label,
        )
        try:
            split = load_split(ctx.cfg.project.split_data_dir / "split.json")
        except Exception as e:
            split = load_split(ctx.out_dir / "split.json")
        train_idx, test_idx = split["train_index"], split["test_index"]
        overall_idx = train_idx + test_idx
        X_full_raw = X_full_raw.drop(columns=overall_idx, errors='ignore')
        ctx.cache["X_recorrected"] = X_full_raw
