# omiclass/pipeline/steps/raw_count_norm_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.data.processor import DataProcessor
from omiclass.features.embedding_backend import embedding_enabled

class RawCountNormalizationStep:
    name = "raw_count_normalization"

    def should_run(self, ctx: RunContext) -> bool:
        if embedding_enabled(ctx.cfg):
            return False
        if str(getattr(ctx.cfg.project, "modality", "rnaseq")) == "cptac_ms":
            return False
        # Only normalize when starting from raw counts
        if ctx.opts.input_mode != "raw_counts":
            return False
        return not ctx.cache.get("_did_raw_count_norm", False)

    def run(self, ctx: RunContext) -> None:
        counts = ctx.cache["full_expr"]
        dp = DataProcessor(ctx.cfg)
        normalized_expr = dp.normalize_raw_counts(X_counts=counts)
        ctx.cache["full_expr"] = normalized_expr
        ctx.cache["_did_raw_count_norm"] = True
