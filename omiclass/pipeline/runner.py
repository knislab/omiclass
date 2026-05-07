# omiclass/pipeline/runner.py
from __future__ import annotations
from datetime import datetime

from pathlib import Path
from typing import Any

from omiclass.config.loaders import load_config
from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.pipeline.orchestrator import PipelineOrchestrator
from omiclass.pipeline.runner_result import PipelineRunResult
from omiclass.utils.clean_and_pred import has_leaderboard

def run_pipeline(
    expression_file: Path,
    **kwargs: Any,
) -> PipelineRunResult:
    cfg = load_config(kwargs)

    project_out_dir = Path(cfg.project.out_dir)
    rebatch_info = None
    out_dir = None
    split_out = None

    if not has_leaderboard(project_out_dir):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = project_out_dir / f"pipeline-{ts}"

        split_dir = getattr(cfg.project, "split_data_dir", None)
        if split_dir is not None:
            split_out = Path(split_dir) / f"pipeline-{ts}"
        else:
            split_out = out_dir

    opts = create_runtime_options_from_path_conf_kwargs(
        cfg,
        expression_path=expression_file,
        kwargs=kwargs
    )

    orch = PipelineOrchestrator(
        cfg=cfg,
        opts=opts,
        out_dir=out_dir if out_dir is not None else cfg.project.out_dir,
        split_out=split_out if split_out is not None else cfg.project.split_data_dir,
    )

    if rebatch_info is not None:
        orch.ctx.cache["rebatch_info"] = rebatch_info
        for k, v in rebatch_info.items():
            orch.ctx.cache[k] = v

    res = orch.run()
    return res