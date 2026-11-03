# omiclass/pipeline/steps/load_step.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext

from omiclass.data.processor import DataProcessor


class LoadInputStep:
    name = "load_input"

    def should_run(self, ctx: RunContext) -> bool:
        # Always first step
        return True

    def run(self, ctx: RunContext) -> None:
        """
        Load expression (or counts) and gene list into ctx.cache:
        - ctx.cache["full_expr"]: expression matrix (samples in columns)
        - ctx.cache["gene_list"]: gene list associated with the matrix
        """
        dp = DataProcessor(ctx.cfg)
        if ctx.cfg.batch_correction.algorithm in {"frozen_quantile", "reference_ruv", "fsva_exact"}:
            from omiclass.batch.reference_pipeline import read_expression
            if ctx.opts.input_mode == "raw_counts":
                raise ValueError("Frozen references require explicit TPM or log1p(TPM) input, not raw counts")
            X = read_expression(ctx.opts.expression_path)
            from omiclass.utils.io import load_list
            genes = load_list(ctx.opts.gene_path) if ctx.opts.gene_path else X.index.tolist()
            if ctx.opts.surface_markers:
                from omiclass.utils.surface_markers import fillter_surface_markers
                genes = fillter_surface_markers(genes)
            ctx.cache.update(full_expr=X, full_expr_t=X.T, gene_list=genes, input_modality="rnaseq")
            return
        modality = getattr(ctx.cfg.project, "modality", "rnaseq")

        modality_label = (
            "CPTAC MS proteomics" if modality == "cptac_ms" else "RNA-seq"
        )
        print(f"[LoadInputStep] Input modality: {modality_label} ({modality})")

        if ctx.opts.input_mode == "raw_counts":
            # For now use the same loader; you can later plug in DE/normalization
            expr_bundle = dp.load_modality_input(
                expression_path=ctx.opts.counts_path,
                gene_path=ctx.opts.gene_path,
                surface_markers=ctx.opts.surface_markers,
            )
            full_expr, gene_list = expr_bundle.X, expr_bundle.genes

        else:
            # batch_corrected or ready_for_training both start from an expression matrix
            expr_bundle = dp.load_modality_input(
                expression_path=ctx.opts.expression_path,
                gene_path=ctx.opts.gene_path,
                surface_markers=ctx.opts.surface_markers,
            )
            full_expr, gene_list = expr_bundle.X, expr_bundle.genes

        ctx.cache["full_expr"] = full_expr
        ctx.cache["full_expr_t"] = full_expr.T  # samples in rows
        ctx.cache["gene_list"] = gene_list
        ctx.cache["input_modality"] = modality
