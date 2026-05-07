# generic_classifier/pipeline/steps/batch_step.py
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from generic_classifier.pipeline.orchestrator import RunContext
from generic_classifier.batch.limma_runner import LimmaRunner
from generic_classifier.reporting.limma_report import write_limma_report
from generic_classifier.utils.io import load_matrix
from generic_classifier.data.processor import DataProcessor

class BatchCorrectionStep:
    name = "batch_correction"

    def should_run(self, ctx: RunContext) -> bool:
        if ctx.opts.mode == "clean_and_predict":
            return False
        if ctx.opts.run_batch is not None:
            return ctx.opts.run_batch
        if ctx.opts.input_mode == "ready_for_training":
            return False
        bc = ctx.cfg.batch_correction and ctx.opts.input_mode == "batch_correction"
        
        return bool(bc and (bc.batch_column or bc.covariates_path or bc.n_pcs) and bc.method) and ctx.opts.mode in ("full", "batch-train")


    def run(self, ctx: RunContext) -> None:
        """
        Run limma batch correction + batch holdout, store result in ctx.cache["X_corrected"].
        """
        full_expr = ctx.cache["full_expr"]

        # Optional log1p before limma
        if ctx.cfg.runtime.log_transform is True and not ctx.cache.get("_did_raw_count_norm", False) and not ctx.cache.get("_did_log_transform", False):
            full_expr = np.log1p(full_expr)
            ctx.cache["_did_log_transform"] = True

        dp = DataProcessor(ctx.cfg)
        meta_df, cov_matrix = dp.load_batch_metadata(
            X_expr=full_expr
        )

        limma_runner = LimmaRunner()
        limma_out_dir = ctx.out_dir / "limma_correction"
        limma_out_dir.mkdir(parents=True, exist_ok=True)
        limma_out_csv = "limma_corrected_expression.csv"

        bc = ctx.cfg.batch_correction
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
        )
        ctx.cache["X_corrected"] = X_full_raw