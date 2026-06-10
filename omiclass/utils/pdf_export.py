from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


_PDF_DIR: Optional[Path] = None
_PDF_COUNTS: dict[str, int] = {}


def configure_pdf_export(out_dir: Path, *, enabled: bool) -> None:
    global _PDF_DIR, _PDF_COUNTS

    _PDF_COUNTS = {}
    if not enabled:
        _PDF_DIR = None
        return

    import matplotlib as mpl

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    _PDF_DIR = Path(out_dir) / "pdf_plots"
    _PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] PDF export enabled. Saving PDF plots to: {_PDF_DIR}")


def save_pdf_plot(fig, name: Optional[str] = None) -> Optional[Path]:
    if _PDF_DIR is None:
        return None

    base = _safe_pdf_stem(name or "plot")
    count = _PDF_COUNTS.get(base, 0) + 1
    _PDF_COUNTS[base] = count

    stem = base if count == 1 else f"{base}_{count}"
    path = _PDF_DIR / f"{stem}.pdf"
    while path.exists():
        count += 1
        _PDF_COUNTS[base] = count
        path = _PDF_DIR / f"{base}_{count}.pdf"

    fig.savefig(path, format="pdf", bbox_inches="tight")
    return path


def _safe_pdf_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    stem = stem.strip("._-")
    return stem or "plot"
