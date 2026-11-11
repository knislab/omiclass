from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from omiclass.config.schema import Config, BatchCorrectionCfg
from omiclass.cli.runtime import RuntimeOptions, create_runtime_options_from_path_conf_kwargs
from omiclass.pipeline.steps.batch_step import BatchCorrectionStep
from omiclass.pipeline.steps.rebatch_step import RebatchCorrectionStep
from omiclass.pipeline.steps.load_step import LoadInputStep


def cfg(tmp_path, **batch):
    return Config(project=dict(meta_path=tmp_path / "meta.csv", cluster_column="target", out_dir=tmp_path),
                  batch_correction=batch)


def test_legacy_config_and_execution_modes(tmp_path):
    legacy = cfg(tmp_path, method=["batch", "design"], batch_column="cohort", design_columns=["age"])
    assert legacy.batch_correction.algorithm == "limma"
    assert legacy.batch_correction.method == ["batch", "design"]
    ctx = SimpleNamespace(cfg=legacy, opts=RuntimeOptions(mode="batch-train", input_mode="batch_correction", run_batch=True))
    assert BatchCorrectionStep().should_run(ctx)
    ctx.opts.input_mode = "ready_for_training"
    ctx.opts.run_batch = None
    assert not BatchCorrectionStep().should_run(ctx)
    ctx.opts.mode = "clean_and_predict"
    assert RebatchCorrectionStep().should_run(ctx)
    legacy.batch_correction.algorithm = "none"
    ctx.opts.mode = "batch-train"
    ctx.opts.run_batch = True
    assert not BatchCorrectionStep().should_run(ctx)
    for algorithm in ["frozen_quantile", "reference_ruv", "fsva_exact"]:
        config = cfg(tmp_path, algorithm=algorithm)
        opts = create_runtime_options_from_path_conf_kwargs(config, tmp_path / "tpm.csv", {})
        assert opts.mode == "batch-train" and opts.run_batch and not opts.run_de
        with pytest.raises(ValueError, match="TPM"):
            create_runtime_options_from_path_conf_kwargs(config, tmp_path / "counts.csv", {"raw_count": True})
        (tmp_path / "leaderboard.csv").touch()
        assert create_runtime_options_from_path_conf_kwargs(config, tmp_path / "tpm.csv", {}).mode == "clean_and_predict"
        (tmp_path / "leaderboard.csv").unlink()
    # Legacy flag validations retain their original semantics.
    with pytest.raises(ValueError, match="batch_column"):
        BatchCorrectionCfg(method="batch")


def test_load_fit_predict_stages_use_saved_reference_without_labels(tmp_path):
    train = pd.DataFrame([[1., 2.], [5., 7.], [12., 11.]], index=["a", "b", "c"], columns=["t1", "t2"])
    path = tmp_path / "input.csv"
    train.to_csv(path)
    ids = tmp_path / "ids.txt"
    ids.write_text("t1\nt2\n")
    config = cfg(tmp_path, algorithm="frozen_quantile", training_samples_path=ids)
    ctx = SimpleNamespace(cfg=config, opts=RuntimeOptions(mode="batch-train", expression_path=path,
        input_mode="batch_correction"), cache={}, artifacts={}, out_dir=tmp_path)
    LoadInputStep().run(ctx)
    BatchCorrectionStep().run(ctx)
    assert ctx.cache["_did_log_transform"]
    saved = (tmp_path / "preprocessing/reference.npz").read_bytes()
    ext = train.rename(columns={"t1": "DKFZ_1", "t2": "DKFZ_2"}) * 3
    ext.to_csv(path)
    ctx.opts.mode = "clean_and_predict"
    assert not BatchCorrectionStep().should_run(ctx)
    LoadInputStep().run(ctx)
    RebatchCorrectionStep().run(ctx)
    assert ctx.cache["X_recorrected"].columns.tolist() == ["DKFZ_1", "DKFZ_2"]
    assert saved == (tmp_path / "preprocessing/reference.npz").read_bytes()
    assert not config.project.meta_path.exists()


def test_legacy_limma_dispatch_arguments_unchanged(tmp_path, monkeypatch):
    import sys
    import omiclass.pipeline.steps.batch_step as module
    calls = []
    X = pd.DataFrame([[1.2, 2.3], [4.1, 3.2]], index=["g1", "g2"], columns=["s1", "s2"])
    metadata = pd.DataFrame({"cohort": ["a", "b"], "age": [50, 60]}, index=X.columns)
    class FakeRunner:
        def run_limma(self, **kwargs):
            calls.append(kwargs)
            X.to_csv(kwargs["output_dir"] / kwargs["out_csv"])
            return {}
    # Dispatch regression does not depend on a configured R runtime.
    monkeypatch.setitem(sys.modules, "omiclass.batch.limma_runner", SimpleNamespace(LimmaRunner=FakeRunner))
    monkeypatch.setattr(module, "DataProcessor", lambda config: SimpleNamespace(load_batch_metadata=lambda **kw: (metadata, None)))
    monkeypatch.setattr(module, "write_limma_report", lambda **kwargs: None)
    config = cfg(tmp_path, method=["batch", "design"], batch_column="cohort", design_columns=["age"])
    ctx = SimpleNamespace(cfg=config, opts=RuntimeOptions(mode="batch-train", input_mode="batch_correction"),
                          cache={"full_expr": X}, out_dir=tmp_path, artifacts={})
    BatchCorrectionStep().run(ctx)
    assert len(calls) == 1
    kw = calls[0]
    pd.testing.assert_frame_equal(kw["expr_df"], X)
    assert kw["meta_df"] is metadata
    assert kw["batch_col"] == "cohort" and kw["design_cols"] == ["age"]
    assert kw["n_pc"] == 0 and kw["na_fill"] is None and kw["cov_matrix"] is None
    assert kw["out_csv"] == "limma_corrected_expression.csv"
    pd.testing.assert_frame_equal(ctx.cache["X_corrected"], X)
