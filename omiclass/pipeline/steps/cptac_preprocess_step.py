from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext

from omiclass.data.processor import DataProcessor


class CPTACPreprocessingStep:
    name = "cptac_ms_preprocessing"

    def should_run(self, ctx: RunContext) -> bool:
        modality = str(getattr(ctx.cfg.project, "modality", "rnaseq"))
        return modality == "cptac_ms"

    def run(self, ctx: RunContext) -> None:
        if 'split' in ctx.cache and ctx.opts.mode != 'clean_and_predict':
            ctx.cache['cptac_deferred'] = True
            return  # Fitted separately inside every estimator/DE training call.
        if (ctx.out_dir / 'preprocessing/scoped.json').is_file():
            return  # The saved estimator owns its fitted transform.
        full_expr = ctx.cache["full_expr"]
        dp = DataProcessor(ctx.cfg)
        processed, summary = dp.preprocess_cptac_ms(full_expr)
        ctx.cache["full_expr"] = processed
        ctx.cache["full_expr_t"] = processed.T
        ctx.cache["cptac_preprocessing"] = summary
        print(f"[CPTACPreprocessingStep] Applied CPTAC preprocessing: {summary}")
