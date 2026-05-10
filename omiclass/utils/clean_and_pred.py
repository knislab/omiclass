from datetime import datetime
from pathlib import Path
import re
from typing import Any, Optional


def extract_pipeline_ts(path: Path, cfg: Any) -> Optional[str]:
    proj_name = cfg.project.name
    m = re.match(rf"^{proj_name}-(\d{8}-\d{6})$", path.name)
    return m.group(1) if m else datetime.now().strftime("%Y%m%d-%H%M%S")


def has_leaderboard(path: Path) -> bool:
    return (Path(path) / "leaderboard.csv").is_file()