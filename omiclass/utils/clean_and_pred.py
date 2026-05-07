from datetime import datetime
from pathlib import Path
import re
from typing import Optional


def extract_pipeline_ts(path: Path) -> Optional[str]:
    m = re.match(r"^pipeline-(\d{8}-\d{6})$", path.name)
    return m.group(1) if m else datetime.now().strftime("%Y%m%d-%H%M%S")


def has_leaderboard(path: Path) -> bool:
    return (Path(path) / "leaderboard.csv").is_file()