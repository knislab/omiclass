# omiclass/pipeline/steps/raw_count_norm_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.data.processor import DataProcessor
from omiclass.features.embedding_backend import embedding_enabled
import joblib
import json
from omiclass.errors.errors import DataError

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
        saved = ctx.out_dir / 'preprocessing/raw_count_gene_lengths.joblib'
        predicting = ctx.opts.mode == 'clean_and_predict'
        scope_path = ctx.out_dir / 'preprocessing/scoped.json'
        scope = json.loads(scope_path.read_text()) if predicting and scope_path.is_file() else {}
        if scope.get('frozen_raw_count_normalization') and not saved.is_file():
            raise DataError('Saved raw-count normalization state is missing; prediction cannot rebuild it')
        lengths = joblib.load(saved) if predicting and saved.is_file() else None
        normalized_expr = dp.normalize_raw_counts(X_counts=counts, gene_lengths=lengths)
        if not predicting:
            saved.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(dp.normalization_gene_lengths_, saved)
        ctx.cache["full_expr"] = normalized_expr
        ctx.cache["_did_raw_count_norm"] = True
