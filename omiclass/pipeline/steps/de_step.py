# omiclass/pipeline/steps/de_step.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.utils.io import load_matrix
from omiclass.data.processor import DataProcessor
from omiclass.de.limma_voom_runner import LimmaVoomRunner
from omiclass.utils.pipeline import make_or_load_split


class DEStep:
    name = "differential_expression"

    def should_run(self, ctx: RunContext) -> bool:
        if ctx.opts.mode == "clean_and_predict":
            return False
        if ctx.opts.run_de is not None:
            return ctx.opts.run_de
        if ctx.opts.input_mode == "ready_for_training":
            return False
        de = ctx.cfg.de
        return bool(de and de.geneset_selection) and ctx.opts.mode in ("full", "de-train")

    def run(self, ctx: RunContext) -> None:
        """
        Run differential expression analysis, store result in ctx.cache["de_results"].
        """
        full_expr = ctx.cache["full_expr"]

        dp = DataProcessor(ctx.cfg)
        meta_df, design_col, y = dp.load_de_data(
            X_expr=full_expr
        )
        common_index = meta_df.index.intersection(y.index)
        common_index = common_index.intersection(full_expr.columns)
        full_expr = full_expr[common_index]
        meta_df = meta_df.loc[common_index, :] 
        y = y.loc[common_index]

        Xtr, Xte, ytr, yte, split_path, created = make_or_load_split(
            X=full_expr.T,
            y=y,
            split_dir=ctx.split_out,
            cfg=ctx.cfg,
            reuse_split=ctx.opts.reuse_split,
        )
        de_cfg = ctx.cfg.de 

        limma = LimmaVoomRunner(de_cfg=de_cfg, cfg=ctx.cfg)
        min_count = getattr(de_cfg, 'min_count', 10)

        res_dict = limma.run_limma_voom(
            raw_counts_df=Xtr.T,
            meta_df=meta_df,
            output_dir=ctx.out_dir,
            design_col=design_col,
            covariates=de_cfg.covariates if de_cfg.covariates else None,
            min_count=min_count,
            min_samples_frac=de_cfg.min_samples_frac if de_cfg.min_samples_frac else 0.5,
            surface_markers=ctx.cfg.surface_markers.enabled if ctx.cfg.surface_markers and ctx.cfg.surface_markers.enabled else False
        )

        ctx.cache["gene_list"] = res_dict.get('selected_genes', [])