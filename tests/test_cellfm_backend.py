from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import omiclass.features.cellfm as cellfm_module
from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.config.loaders import load_config
from omiclass.config.schema import CellFMCfg
from omiclass.errors.errors import SchemaValidationError
from omiclass.features.bulkformer_reduction import build_embedding_representation
from omiclass.features.cellfm import (
    CELLFM_EMBEDDING_DIMENSION,
    CellFMExtractionResult,
    CellFMPreparedInput,
    _extract_configured_views,
    _extract_multiview,
    _stable_view_seed,
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
        "cellfm_num_views": 1,
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
    assert cfg.cellfm.num_views == 1
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
    assert cfg.cellfm.num_views == 20
    assert CellFMCfg(num_views=1).num_views == 1


def test_cellfm_num_views_accepts_cli_integer_string(tmp_path):
    cfg = load_config({**_base(tmp_path), "cellfm_num_views": "20"})
    assert cfg.cellfm.num_views == 20


@pytest.mark.parametrize("num_views", [0, -1, 1.5, True])
def test_cellfm_num_views_requires_positive_integer(tmp_path, num_views):
    with pytest.raises(SchemaValidationError, match="num_views"):
        load_config({**_base(tmp_path), "cellfm_num_views": num_views})


def test_cellfm_view_seeds_are_stable_and_distinct():
    seeds = [_stable_view_seed("sample_a", 17, view) for view in range(20)]
    assert seeds == [_stable_view_seed("sample_a", 17, view) for view in range(20)]
    assert len(set(seeds)) == 20
    assert _stable_view_seed("sample_b", 17, 0) != seeds[0]


def _prepared_for_multiview(sample_ids=("sample_a", "sample_b")):
    counts_by_id = {"sample_a": [1, 3], "sample_b": [2, 4]}
    return CellFMPreparedInput(
        counts=np.asarray([counts_by_id[sample_id] for sample_id in sample_ids]),
        gene_ids=np.asarray([2, 1], dtype=np.int32),
        sample_ids=list(sample_ids),
        mapped_genes=["G2", "G1"],
        unmapped_genes=[],
        raw_library_sizes=np.asarray([4, 6], dtype=np.float64),
    )


def test_cellfm_multiview_generates_20_views_and_averages_deterministically(monkeypatch):
    calls = []

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def to(self, _device):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakeInferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_torch = SimpleNamespace(
        from_numpy=FakeTensor,
        inference_mode=FakeInferenceMode,
    )

    def fake_tokenize(_prep, _counts, _gene_ids, *, seed):
        calls.append(seed)
        value = np.float32(seed % 997)
        return (
            np.full(2048, value, dtype=np.float32),
            np.ones(2048, dtype=np.int32),
            np.zeros(2, dtype=np.float32),
            np.ones(2048, dtype=np.float32),
        )

    def fake_pool(_model, expression, _genes, _st_features, _valid):
        return FakeTensor(
            np.repeat(expression.array[:, :1], CELLFM_EMBEDDING_DIMENSION, axis=1)
        )

    monkeypatch.setattr(cellfm_module, "_tokenize_sample_view", fake_tokenize)
    monkeypatch.setattr(cellfm_module, "_pool_cellfm_batch", fake_pool)
    prepared = _prepared_for_multiview()
    embeddings, retained = _extract_multiview(
        torch=fake_torch,
        model=object(),
        prep=object(),
        prepared=prepared,
        device="cpu",
        batch_size=1,
        random_state=17,
        num_views=20,
    )

    expected = [
        np.mean([_stable_view_seed(sample_id, 17, view) % 997 for view in range(20)])
        for sample_id in prepared.sample_ids
    ]
    assert len(calls) == len(prepared.sample_ids) * 20
    assert embeddings.shape == (2, CELLFM_EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32
    assert np.allclose(embeddings[:, 0], expected)
    assert np.all(embeddings == embeddings[:, :1])
    assert retained == [[2048] * 20, [2048] * 20]

    calls.clear()
    reordered = _prepared_for_multiview(("sample_b", "sample_a"))
    reordered_embeddings, _ = _extract_multiview(
        torch=fake_torch,
        model=object(),
        prep=object(),
        prepared=reordered,
        device="cpu",
        batch_size=2,
        random_state=17,
        num_views=20,
    )
    assert np.allclose(reordered_embeddings[0], embeddings[1])
    assert np.allclose(reordered_embeddings[1], embeddings[0])


def test_cellfm_num_views_one_can_replay_legacy_implementation(monkeypatch):
    expected = np.ones((2, CELLFM_EMBEDDING_DIMENSION), dtype=np.float32)
    loader = object()
    utils = SimpleNamespace(build_dataset=lambda *args, **kwargs: loader)
    monkeypatch.setattr(
        cellfm_module,
        "_extract_legacy_single_view",
        lambda **kwargs: (expected, [2048, 2048]),
    )
    monkeypatch.setattr(
        cellfm_module,
        "_extract_multiview",
        lambda **kwargs: pytest.fail("multiview path used for num_views=1"),
    )

    embeddings, retained, seed_strategy, aggregation = _extract_configured_views(
        cfg=SimpleNamespace(num_views=1, batch_size=2, random_state=17),
        utils=utils,
        torch=object(),
        model=object(),
        prep=object(),
        prepared=_prepared_for_multiview(),
        device="cpu",
        legacy_single_view=True,
    )
    assert embeddings is expected
    assert retained == [2048, 2048]
    assert seed_strategy == "legacy_global_numpy_random_state"
    assert aggregation == "single_view"


def test_cellfm_default_routes_to_20_view_mean(monkeypatch):
    expected = np.ones((2, CELLFM_EMBEDDING_DIMENSION), dtype=np.float32)
    captured = {}

    def fake_multiview(**kwargs):
        captured.update(kwargs)
        return expected, [[2048] * 20, [2048] * 20]

    monkeypatch.setattr(cellfm_module, "_extract_multiview", fake_multiview)
    monkeypatch.setattr(
        cellfm_module,
        "_extract_legacy_single_view",
        lambda **kwargs: pytest.fail("legacy path used for default num_views=20"),
    )
    prepared = _prepared_for_multiview()
    embeddings, _retained, seed_strategy, aggregation = _extract_configured_views(
        cfg=CellFMCfg(),
        utils=SimpleNamespace(),
        torch=object(),
        model=object(),
        prep=object(),
        prepared=prepared,
        device="cpu",
    )
    assert embeddings is expected
    assert captured["num_views"] == 20
    assert prepared.gene_ids.tolist() == [1, 2]
    assert seed_strategy == "sha256_sample_id_global_seed_zero_based_view_index"
    assert aggregation == "mean"


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
    assert transformer.extraction_signature_["num_views"] == "20"
