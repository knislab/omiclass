from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext

from omiclass.features.bulkformer import BulkFormerFeatureExtractor, write_bulkformer_artifacts


class BulkFormerFeatureExtractionStep:
    name = "bulkformer_feature_extraction"

    def should_run(self, ctx: RunContext) -> bool:
        cfg = getattr(ctx.cfg, "bulkformer", None)
        if ctx.opts.mode == "clean_and_predict":
            return False
        return bool(cfg and cfg.enabled)

    def run(self, ctx: RunContext) -> None:
        counts = ctx.cache["full_expr"]
        extractor = BulkFormerFeatureExtractor(ctx.cfg.bulkformer)
        result = extractor.extract(counts)

        artifact_dir = ctx.out_dir / "bulkformer"
        feature_path, metadata_path = write_bulkformer_artifacts(
            out_dir=artifact_dir,
            features=result.features,
            metadata=result.metadata,
        )

        feature_list = list(result.features.index)
        ctx.cache["full_expr"] = result.features
        ctx.cache["full_expr_t"] = result.features.T
        ctx.cache["feature_list"] = feature_list
        ctx.cache["gene_list"] = feature_list
        ctx.cache["feature_source"] = "bulkformer"
        ctx.cache["bulkformer_metadata"] = result.metadata
        ctx.artifacts["bulkformer_features"] = feature_path
        ctx.artifacts["bulkformer_metadata"] = metadata_path
        print(f"[BulkFormer] Feature embeddings saved at: {feature_path}")
