# generic_classifier/pipeline/steps/raw_count_norm_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from generic_classifier.pipeline.orchestrator import RunContext
from generic_classifier.data.processor import DataProcessor

class RawCountNormalizationStep:
    name = "raw_count_normalization"

    def should_run(self, ctx: RunContext) -> bool:
        # Only normalize when starting from raw counts
        if ctx.opts.input_mode != "raw_counts":
            return False
        return not ctx.cache.get("_did_raw_count_norm", False)

    def run(self, ctx: RunContext) -> None:
        counts = ctx.cache["full_expr"]
        print(f"[DEBUG] [RawCountNormalizationStep] Normalizing raw counts with shape {counts.shape}")
        dp = DataProcessor(ctx.cfg)
        normalized_expr = dp.normalize_raw_counts(X_counts=counts)
        print(f"[DEBUG] [RawCountNormalizationStep] Finished normalization, result shape {normalized_expr.shape}")
        ctx.cache["full_expr"] = normalized_expr
        ctx.cache["_did_raw_count_norm"] = True