# generic_classifier/pipeline/orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Protocol, List

from generic_classifier.config.schema import Config
from generic_classifier.cli.runtime import RuntimeOptions
from generic_classifier.pipeline.runner_result import PipelineRunResult
from generic_classifier.pipeline.steps.load_step import LoadInputStep
from generic_classifier.pipeline.steps.survival_step import SurvivalStep
from generic_classifier.pipeline.steps.batch_step import BatchCorrectionStep
from generic_classifier.pipeline.steps.train_eval_step import TrainAndEvalStep
from generic_classifier.pipeline.steps.raw_count_norm_step import RawCountNormalizationStep
from generic_classifier.pipeline.steps.de_step import DEStep
from generic_classifier.pipeline.steps.surface_merker_step import SurfaceMarkerStep
from generic_classifier.reporting.sidebar_finalize import finalize_unified_sidebar
from generic_classifier.pipeline.steps.rebatch_step import RebatchCorrectionStep
from generic_classifier.pipeline.steps.define_genecode_step import DefineGeneCodeStep



@dataclass
class RunContext:
    cfg: Config
    opts: RuntimeOptions
    out_dir: Path
    split_out: Path
    artifacts: Dict[str, Path]
    cache: Dict[str, Any]


class PipelineStep(Protocol):
    name: str

    def should_run(self, ctx: RunContext) -> bool:
        ...

    def run(self, ctx: RunContext) -> None:
        ...

class PipelineOrchestrator:
    """
    High-level orchestrator that glues together modular steps:
    - LoadInputStep (raw counts / expression)
    - BatchCorrectionStep (limma + batch holdout)
    - TrainAndEvalStep (train on train, evaluate on test, plots & survival)
    Additional steps (DE, gene-set optimization, prediction-only,
    standalone KM, etc.) can be plugged into `self.steps`.
    """

    def __init__(self, cfg: Config, opts: RuntimeOptions, out_dir: Path, split_out: Path,) -> None:
        self.cfg = cfg
        self.opts = opts
        self.out_dir = Path(out_dir)
        self.split_out = Path(split_out)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.split_out.mkdir(parents=True, exist_ok=True)

        self.steps: List[PipelineStep] = [
            DefineGeneCodeStep(),
            LoadInputStep(),
            DEStep(),
            RawCountNormalizationStep(),
            SurfaceMarkerStep(),
            BatchCorrectionStep(),
            TrainAndEvalStep(),
            RebatchCorrectionStep(),
            SurvivalStep(),
        ]

    def run(self) -> PipelineRunResult:
        ctx = RunContext(
            cfg=self.cfg,
            opts=self.opts,
            out_dir=self.out_dir,
            split_out=self.split_out,
            artifacts={},
            cache={},
        )
        for step in self.steps:
            if step.should_run(ctx):
                print(f"[DEBUG] [ORCHESTRATOR] running {step.name}")
                step.run(ctx)
                #print(f"[DEBUG] [ORCHESTRATOR] finished {step.name}")

        # render html reports 
        finalize_unified_sidebar(Path(ctx.out_dir))

        return PipelineRunResult(
            out_dir=self.out_dir,
            split_json=ctx.cache.get("split_json"),
            leaderboard_csv=ctx.cache.get("leaderboard_csv"),
            artifacts=ctx.artifacts,
            test_reports={},  
            combined_test_csv=ctx.cache.get("combined_test_csv"),
        )