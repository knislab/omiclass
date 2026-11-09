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
            import json
            saved = ctx.out_dir / 'cellfm/cellfm_metadata.json'
            legacy = False
            if ctx.opts.mode == 'clean_and_predict' and ctx.cfg.cellfm.num_views == 1:
                if not saved.is_file():
                    raise ValueError('Saved CellFM single-view sampling metadata is required for replay')
                strategy = json.loads(saved.read_text()).get('view_seed_strategy', 'legacy_global_numpy_random_state')
                if strategy not in {'legacy_global_numpy_random_state', 'sha256_sample_id_global_seed_zero_based_view_index'}:
                    raise ValueError('Unsupported saved CellFM sampling protocol')
                legacy = strategy == 'legacy_global_numpy_random_state'
            extractor = CellFMFeatureExtractor(ctx.cfg.cellfm, **({'legacy_single_view': True} if legacy else {}))
            writer = write_cellfm_artifacts
        elif backend == "bulkrnabert":
            from omiclass.features.bulkrnabert import BulkRNABertFeatureExtractor, write_bulkrnabert_artifacts

            extractor = BulkRNABertFeatureExtractor(ctx.cfg.bulkrnabert)
            writer = write_bulkrnabert_artifacts
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
