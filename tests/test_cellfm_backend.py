from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.config.loaders import load_config
from omiclass.errors.errors import SchemaValidationError
from omiclass.features.bulkformer_reduction import build_embedding_representation
from omiclass.features.cellfm import (
    CellFMExtractionResult,
    cellfm_feature_matrix,
    prepare_cellfm_input,
)
from omiclass.pipeline.steps.bulkformer_step import EmbeddingFeatureExtractionStep


def _base(tmp_path):
    return {
        "project_name": "cellfm_test",
        "output_dir": str(tmp_path / "out"),
        "meta_path": str(tmp_path / "meta.csv"),
        "class_col": "class",
    }


def test_cellfm_config_aliases_and_runtime_route(tmp_path):
    overrides = _base(tmp_path)
    overrides.update({
        "cellfm_enabled": True,
        "cellfm_batch_size": 3,
        "cellfm_random_state": 17,
        "cellfm_reduction_strategy": "supervised_pca",
        "cellfm_supervised_dimensions": 12,
        "cellfm_pca_components": 4,
        "feature_selection_enabled": False,
        "batch_column": "batch",
    })
    cfg = load_config(overrides)

    assert cfg.cellfm.enabled is True
    assert cfg.bulkformer.enabled is False
    assert cfg.cellfm.batch_size == 3
    assert cfg.cellfm.random_state == 17
    assert cfg.cellfm.reduction.strategy == "supervised_pca"
    assert cfg.cellfm.reduction.supervised_dimensions == 12
    assert cfg.cellfm.reduction.pca_components == 4

    opts = create_runtime_options_from_path_conf_kwargs(
        cfg, tmp_path / "counts.csv", overrides
    )
    assert opts.mode == "train-only"
    assert opts.input_mode == "raw_counts"
    assert opts.run_de is False
    assert opts.run_batch is False
    assert opts.surface_markers is False


def test_cellfm_resource_defaults_follow_root_dir(tmp_path):
    root = tmp_path / "CellFM" / "CellFM-torch"
    cfg = load_config({**_base(tmp_path), "cellfm_root_dir": str(root)})
    assert cfg.cellfm.gene_info_path == root / "csv" / "expand_gene_info.csv"
    assert cfg.cellfm.checkpoint_path == root.parent / "weights" / "CellFM_80M_weight.ckpt"
    assert cfg.cellfm.vendor_dir == root.parent / "vendor"


def test_embedding_backends_are_mutually_exclusive(tmp_path):
    with pytest.raises(SchemaValidationError, match="enable only one"):
        load_config({**_base(tmp_path), "bulkformer_enabled": True, "cellfm_enabled": True})


def test_prepare_cellfm_input_maps_caps_and_preserves_samples():
    counts = pd.DataFrame(
        {"sample_a": [100_000, 100_000, 100_000], "sample_b": [2, 3, 5]},
        index=["G1", "G2", "NOT_IN_VOCAB"],
    )
    prepared = prepare_cellfm_input(counts, ["G1", "G2", "G3"])

    assert prepared.sample_ids == ["sample_a", "sample_b"]
    assert prepared.mapped_genes == ["G1", "G2"]
    assert prepared.unmapped_genes == ["NOT_IN_VOCAB"]
    assert prepared.gene_ids.tolist() == [1, 2]
    # Library cap uses all genes before discarding the unmapped feature.
    assert prepared.counts.tolist() == [[33333, 33333], [2, 3]]


def test_prepare_cellfm_input_maps_ensembl_ids_to_cellfm_symbols():
    counts = pd.DataFrame(
        {"sample_a": [10, 20], "sample_b": [1, 2]},
        index=["ENSG00000000003", "ENSG_DOES_NOT_EXIST"],
    )
    prepared = prepare_cellfm_input(counts, ["TSPAN6"])

    assert prepared.mapped_genes == ["TSPAN6"]
    assert prepared.unmapped_genes == ["ENSG_DOES_NOT_EXIST"]
    assert prepared.ensembl_mapped_genes == 1
    assert prepared.gene_ids.tolist() == [1]
    assert prepared.counts.tolist() == [[10], [1]]


def test_cellfm_feature_matrix_is_features_by_samples():
    result = cellfm_feature_matrix(
        np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32), ["a", "b"]
    )
    assert result.index.tolist() == [
        "cellfm_feature_0000", "cellfm_feature_0001", "cellfm_feature_0002"
    ]
    assert result.columns.tolist() == ["a", "b"]


def test_cellfm_step_enters_shared_feature_handoff(tmp_path, monkeypatch):
    cfg = load_config({**_base(tmp_path), "cellfm_enabled": True})
    counts = pd.DataFrame({"a": [1], "b": [2]}, index=["G1"])
    features = pd.DataFrame(
        {"a": [0.1, 0.2], "b": [0.3, 0.4]},
        index=["cellfm_feature_0000", "cellfm_feature_0001"],
    )

    class FakeExtractor:
        def __init__(self, extractor_cfg):
            assert extractor_cfg is cfg.cellfm

        def extract(self, actual):
            assert actual.equals(counts)
            return CellFMExtractionResult(features, {"backend": "cellfm"})

    monkeypatch.setattr(
        "omiclass.pipeline.steps.bulkformer_step.CellFMFeatureExtractor", FakeExtractor
    )
    context = SimpleNamespace(
        cfg=cfg, opts=SimpleNamespace(mode="train-only"), out_dir=tmp_path / "out",
        artifacts={}, cache={"full_expr": counts},
    )
    step = EmbeddingFeatureExtractionStep()
    step.run(context)

    assert context.cache["feature_source"] == "cellfm"
    assert context.cache["full_expr"].equals(features)
    assert context.cache["gene_list"] == features.index.tolist()
    assert context.artifacts["cellfm_features"].exists()
    assert context.artifacts["cellfm_metadata"].exists()


@pytest.mark.parametrize("strategy", ["full", "supervised", "pca", "supervised_pca"])
def test_cellfm_uses_shared_representation_strategies(tmp_path, strategy):
    overrides = {**_base(tmp_path), "cellfm_enabled": True,
                 "cellfm_reduction_strategy": strategy}
    if strategy != "full":
        overrides["feature_selection_enabled"] = False
    cfg = load_config(overrides)
    transformer = build_embedding_representation(cfg.cellfm, 11, backend="cellfm")
    rng = np.random.default_rng(5)
    X = pd.DataFrame(
        rng.normal(size=(18, 8)), columns=[f"cellfm_feature_{i:04d}" for i in range(8)]
    )
    y = np.repeat([0, 1, 2], 6)
    transformed = transformer.fit_transform(X, y)

    assert transformed.shape[0] == len(X)
    assert transformer.strategy == strategy
    assert transformer.extraction_signature_["backend"] == "cellfm"
