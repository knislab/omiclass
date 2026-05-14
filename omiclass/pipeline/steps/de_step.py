# omiclass/pipeline/steps/de_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.utils.io import load_matrix
from omiclass.data.processor import DataProcessor
from omiclass.de.limma_voom_runner import LimmaVoomRunner
from omiclass.utils.pipeline import make_or_load_split


class DEStep:
    name = "feature_selection"

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
        Run modality-aware differential feature analysis and store selected features.
        """
        full_expr = ctx.cache["full_expr"]
        modality = str(getattr(ctx.cfg.project, "modality", "rnaseq"))

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
        bc_cfg = ctx.cfg.batch_correction

        # Build DE-only covariates: biological DE covariates + batch covariates.
        de_covariates = list(de_cfg.covariates) if de_cfg.covariates else []
        de_covariates_seen = set(de_covariates)

        # Case 1: single batch column (e.g., cohort) from metadata.
        if bc_cfg.batch_column and bc_cfg.batch_column in meta_df.columns:
            if bc_cfg.batch_column not in de_covariates_seen:
                de_covariates.append(bc_cfg.batch_column)
                de_covariates_seen.add(bc_cfg.batch_column)

        # Case 2: multi-column batch representation from external covariate matrix.
        if bc_cfg.covariates_path:
            cov_matrix = load_matrix(Path(bc_cfg.covariates_path))
            cov_matrix = cov_matrix.loc[cov_matrix.index.intersection(common_index)]
            cov_matrix = cov_matrix.loc[meta_df.index]
            if not cov_matrix.empty:
                overlap_cols = [c for c in cov_matrix.columns if c in meta_df.columns]
                if overlap_cols:
                    cov_matrix = cov_matrix.drop(columns=overlap_cols)
                if not cov_matrix.empty:
                    meta_df = meta_df.join(cov_matrix, how="left")
                    for col in cov_matrix.columns:
                        if col not in de_covariates_seen:
                            de_covariates.append(col)
                            de_covariates_seen.add(col)

        limma = LimmaVoomRunner(de_cfg=de_cfg, cfg=ctx.cfg)
        min_count = getattr(de_cfg, 'min_count', 10)
        common_kwargs = dict(
            meta_df=meta_df,
            output_dir=ctx.out_dir,
            design_col=design_col,
            covariates=de_covariates if de_covariates else None,
            min_samples_frac=de_cfg.min_samples_frac if de_cfg.min_samples_frac else 0.5,
            surface_markers=ctx.cfg.surface_markers.enabled if ctx.cfg.surface_markers and ctx.cfg.surface_markers.enabled else False,
        )

        if modality == "cptac_ms":
            # CPTAC path: limma on log2-normalized protein abundance, no voom.
            res_dict = limma.run_limma(
                expr_df=Xtr.T,
                out_csv="limma_protein_de_results.csv",
                min_count=0,
                analysis_label="Differential Protein Abundance",
                feature_label_singular="protein",
                feature_label_plural="proteins",
                **common_kwargs,
            )
        else:
            # RNA path: keep current limma-voom behavior unchanged.
            res_dict = limma.run_limma_voom(
                raw_counts_df=Xtr.T,
                min_count=min_count,
                **common_kwargs,
            )

        selected_features = res_dict.get('selected_features', res_dict.get('selected_genes', []))
        ctx.cache["feature_list"] = selected_features
        # Backward compatibility for existing training/batch flows.
        ctx.cache["gene_list"] = selected_features
