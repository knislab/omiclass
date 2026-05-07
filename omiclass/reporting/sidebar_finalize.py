from __future__ import annotations
from pathlib import Path
import re

from omiclass.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html

_ASIDE_RE = re.compile(r"(<aside\s+class=\"sidebar\"\s*>)(.*?)(</aside>)", re.DOTALL)

def finalize_unified_sidebar(run_dir: Path) -> None:
    run_dir = Path(run_dir).resolve()

    tree, _ = build_sidebar_from_run_dir(run_dir)  # MUST be non-strict (final may exist now anyway)

    html_files = []
    html_files += [run_dir / "final_report.html",  run_dir / "final_summary_threshold_optimized.html"]
    html_files += sorted((run_dir / "reports").glob("*.html")) if (run_dir / "reports").exists() else []
    html_files += sorted((run_dir / "limma_correction").glob("*.html")) if (run_dir / "limma_correction").exists() else []
    html_files += sorted((run_dir / "de_results").glob("*.html")) if (run_dir / "de_results").exists() else []
    html_files += sorted((run_dir / "survival_prediction").glob("*.html")) if (run_dir / "survival_prediction").exists() else []

    for page in [p for p in html_files if p.exists()]:
        current_page = page.relative_to(run_dir).as_posix()
        sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)

        text = page.read_text(encoding="utf-8")
        if not _ASIDE_RE.search(text):
            # page doesn't use the shell; skip
            continue

        new_text = _ASIDE_RE.sub(rf"\1\n{sidebar_html}\n\3", text, count=1)
        if new_text != text:
            page.write_text(new_text, encoding="utf-8")
