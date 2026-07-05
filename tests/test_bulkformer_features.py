from types import SimpleNamespace

import numpy as np
import pandas as pd

from omiclass.config.loaders import load_config
from omiclass.features.bulkformer import (
    BulkFormerExtractionResult,
    align_to_bulkformer_vocabulary,
    feature_matrix_from_embeddings,
    normalize_counts_to_log_tpm,
)
from omiclass.pipeline.steps.bulkformer_step import BulkFormerFeatureExtractionStep


def test_normalize_counts_to_log_tpm_preserves_gene_sample_alignment():
    counts = pd.DataFrame(
        {
            "sample_a": [10, 0],
            "sample_b": [5, 5],
        },
        index=["ENSG1", "ENSG2"],
    )
    gene_lengths = {"ENSG1": 1000.0, "ENSG2": 2000.0}

    normalized = normalize_counts_to_log_tpm(counts, gene_lengths)

    assert normalized.index.tolist() == ["ENSG1", "ENSG2"]
    assert normalized.columns.tolist() == ["sample_a", "sample_b"]
    assert normalized.shape == counts.shape
    assert np.isfinite(normalized.to_numpy()).all()


def test_align_to_bulkformer_vocabulary_fills_missing_genes():
    log_tpm = pd.DataFrame(
        {
            "sample_a": [1.0, 2.0],
            "sample_b": [3.0, 4.0],
        },
        index=["ENSG1", "ENSG3"],
    )

    prepared = align_to_bulkformer_vocabulary(log_tpm, ["ENSG1", "ENSG2", "ENSG3"])

    assert prepared.input_df.index.tolist() == ["sample_a", "sample_b"]
    assert prepared.input_df.columns.tolist() == ["ENSG1", "ENSG2", "ENSG3"]
    assert prepared.input_df.loc["sample_a", "ENSG2"] == -10.0
    assert prepared.missing_genes == ["ENSG2"]
    assert prepared.mask_prob == 1 / 3


def test_feature_matrix_from_embeddings_is_features_by_samples():
    features = feature_matrix_from_embeddings(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        ["sample_a", "sample_b"],
    )

    assert features.index.tolist() == [
        "bulkformer_feature_0000",
        "bulkformer_feature_0001",
        "bulkformer_feature_0002",
    ]
    assert features.columns.tolist() == ["sample_a", "sample_b"]
    assert features.shape == (3, 2)


def test_bulkformer_step_updates_cache_with_mocked_extractor(tmp_path, monkeypatch):
    cfg = load_config(
        {
            "project_name": "mock_bulkformer",
            "output_dir": str(tmp_path / "out"),
            "meta_path": str(tmp_path / "meta.csv"),
            "class_col": "class",
            "bulkformer_enabled": True,
        }
    )
    input_counts = pd.DataFrame({"sample_a": [1], "sample_b": [2]}, index=["ENSG1"])
    mocked_features = pd.DataFrame(
        {
            "sample_a": [0.1, 0.2],
            "sample_b": [0.3, 0.4],
        },
        index=["bulkformer_feature_0000", "bulkformer_feature_0001"],
    )

    class FakeExtractor:
        def __init__(self, cfg):
            self.cfg = cfg

        def extract(self, counts):
            assert counts.equals(input_counts)
            return BulkFormerExtractionResult(
                features=mocked_features,
                metadata={"device": "cpu", "feature_shape": [2, 2]},
            )

    monkeypatch.setattr(
        "omiclass.pipeline.steps.bulkformer_step.BulkFormerFeatureExtractor",
        FakeExtractor,
    )
    ctx = SimpleNamespace(
        cfg=cfg,
        opts=SimpleNamespace(mode="train-only"),
        out_dir=tmp_path / "out",
        artifacts={},
        cache={"full_expr": input_counts},
    )

    step = BulkFormerFeatureExtractionStep()
    assert step.should_run(ctx) is True
    step.run(ctx)

    assert ctx.cache["feature_source"] == "bulkformer"
    assert ctx.cache["gene_list"] == mocked_features.index.tolist()
    assert ctx.cache["full_expr"].equals(mocked_features)
    assert ctx.artifacts["bulkformer_features"].exists()
    assert ctx.artifacts["bulkformer_metadata"].exists()
