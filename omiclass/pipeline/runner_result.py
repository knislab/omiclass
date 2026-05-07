# omiclass/pipeline/runner_result.py
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union


@dataclass
class PipelineRunResult:
    out_dir: Path
    split_json: Union[Path, None]
    leaderboard_csv: Union[Path, None]
    artifacts: Dict[str, Path]          # model_key -> model path
    test_reports: Dict[str, Path]       # model_key -> test json path (if you decide)
    combined_test_csv: Union[Path, None]
