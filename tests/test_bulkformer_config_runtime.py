from pathlib import Path

from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.config.loaders import load_config


def _base_overrides(tmp_path):
    return {
        "project_name": "bulkformer_test",
        "output_dir": str(tmp_path / "out"),
        "meta_path": str(tmp_path / "meta.csv"),
        "class_col": "class",
    }


def test_bulkformer_defaults_disabled(tmp_path):
    cfg = load_config(_base_overrides(tmp_path))

    assert cfg.bulkformer.enabled is False
    assert cfg.bulkformer.device == "auto"
    assert cfg.bulkformer.batch_size == 4
    assert cfg.bulkformer.aggregate_type == "mean"
    assert cfg.bulkformer.checkpoint_path == Path("BulkFormer/model/bulkformer_147M.pt")


def test_bulkformer_aliases_map_and_device_auto_is_valid(tmp_path):
    overrides = _base_overrides(tmp_path)
    overrides.update(
        {
            "bulkformer_enabled": True,
            "bulkformer_root_dir": str(tmp_path / "BF"),
            "bulkformer_device": "auto",
            "bulkformer_batch_size": 8,
        }
    )

    cfg = load_config(overrides)

    assert cfg.bulkformer.enabled is True
    assert cfg.bulkformer.root_dir == tmp_path / "BF"
    assert cfg.bulkformer.graph_path == tmp_path / "BF" / "data" / "G_tcga.pt"
    assert cfg.bulkformer.batch_size == 8


def test_runtime_without_bulkformer_preserves_existing_decision(tmp_path):
    overrides = _base_overrides(tmp_path)
    overrides["batch_column"] = "batch"
    cfg = load_config(overrides)

    opts = create_runtime_options_from_path_conf_kwargs(
        cfg,
        expression_path=tmp_path / "counts.csv",
        kwargs=overrides,
    )

    assert opts.mode == "full"
    assert opts.input_mode == "raw_counts"
    assert opts.run_de is True
    assert opts.run_batch is True


def test_runtime_with_bulkformer_forces_embedding_training_path(tmp_path):
    overrides = _base_overrides(tmp_path)
    overrides.update(
        {
            "batch_column": "batch",
            "surface_markers_enabled": True,
            "bulkformer_enabled": True,
        }
    )
    cfg = load_config(overrides)

    opts = create_runtime_options_from_path_conf_kwargs(
        cfg,
        expression_path=tmp_path / "counts.csv",
        kwargs=overrides,
    )

    assert opts.mode == "train-only"
    assert opts.input_mode == "raw_counts"
    assert opts.run_de is False
    assert opts.run_batch is False
    assert opts.run_training is True
    assert opts.surface_markers is False
