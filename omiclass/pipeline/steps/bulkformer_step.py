from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext

from omiclass.features.bulkformer import BulkFormerFeatureExtractor, write_bulkformer_artifacts
from omiclass.features.cellfm import CellFMFeatureExtractor, write_cellfm_artifacts
from omiclass.features.embedding_backend import embedding_backend


class EmbeddingFeatureExtractionStep:
    name = "embedding_feature_extraction"

    def should_run(self, ctx: RunContext) -> bool:
        return embedding_backend(ctx.cfg) is not None

    def run(self, ctx: RunContext) -> None:
        counts = ctx.cache["full_expr"]
        backend = embedding_backend(ctx.cfg)
        if backend == "bulkformer":
            extractor = BulkFormerFeatureExtractor(ctx.cfg.bulkformer)
            writer = write_bulkformer_artifacts
        elif backend == "cellfm":
            extractor = CellFMFeatureExtractor(ctx.cfg.cellfm)
            writer = write_cellfm_artifacts
        else:  # protected by should_run; useful when invoked directly
            return
        result = extractor.extract(counts)

        artifact_dir = ctx.out_dir / backend
        if ctx.opts.mode == "clean_and_predict":
            artifact_dir = artifact_dir / "external_prediction"
        feature_path, metadata_path = writer(
            out_dir=artifact_dir,
            features=result.features,
            metadata=result.metadata,
        )

        feature_list = list(result.features.index)
        ctx.cache["full_expr"] = result.features
        ctx.cache["full_expr_t"] = result.features.T
        ctx.cache["feature_list"] = feature_list
        ctx.cache["gene_list"] = feature_list
        ctx.cache["feature_source"] = backend
        ctx.cache["embedding_metadata"] = result.metadata
        ctx.cache[f"{backend}_metadata"] = result.metadata
        if ctx.opts.mode == "clean_and_predict":
            ctx.cache["X_recorrected"] = result.features
        ctx.artifacts[f"{backend}_features"] = feature_path
        ctx.artifacts[f"{backend}_metadata"] = metadata_path
        print(f"[{backend}] Feature embeddings saved at: {feature_path}")


# Preserve the public class used by existing integrations.
BulkFormerFeatureExtractionStep = EmbeddingFeatureExtractionStep
