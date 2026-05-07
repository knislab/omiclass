from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class NavItem:
    label: str
    target: Optional[str] = None  # posix path relative to RUN ROOT
    children: List["NavItem"] = field(default_factory=list)
    expanded: bool = False


def _to_posix_rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def _rel_href(*, current_page: str, target: str) -> str:
    """
    current_page/target are posix paths relative to run root.
    Return href from current page folder -> target.
    """
    cur = PurePosixPath(current_page).parent
    tgt = PurePosixPath(target)

    cur_parts = cur.parts
    tgt_parts = tgt.parts

    i = 0
    for a, b in zip(cur_parts, tgt_parts):
        if a != b:
            break
        i += 1

    up = [".."] * (len(cur_parts) - i)
    down = list(tgt_parts[i:])
    return str(PurePosixPath(*(up + down)))


def render_sidebar_html(*, tree: List[NavItem], current_page: str) -> str:
    def render_item(it: NavItem) -> str:
        if it.children:
            inner = "\n".join(render_item(c) for c in it.children)
            open_attr = " open" if it.expanded else ""
            return f"""
            <details class="nav-group"{open_attr}>
              <summary class="nav-group-title">{it.label}</summary>
              <div class="nav-group-items">{inner}</div>
            </details>
            """.strip()

        if not it.target:
            return f"<div class='nav-item disabled'>{it.label}</div>"

        href = _rel_href(current_page=current_page, target=it.target)
        active = " active" if it.target == current_page else ""
        return f"<a class='nav-item{active}' href='{href}'>{it.label}</a>"

    return "<nav class='sidebar-nav'>\n" + "\n".join(render_item(x) for x in tree) + "\n</nav>"


def _pretty_model_name(stem: str) -> str:
    # "light_gbm" -> "Light Gbm" (keep simple; adjust if you want smarter)
    return stem.replace("_", " ").title()


def build_sidebar_from_run_dir(run_dir: Path,     current_page: Optional[str] = None) -> Tuple[List[NavItem], dict]:
    """
    Returns (tree, manifest).
    manifest contains resolved page paths for debugging.
    """
    run_dir = Path(run_dir)
    manifest: dict = {}

    # ----- required: final report at root -----
    final_pages: List[Tuple[str, str]] = []

    regular_final = run_dir / "final_report.html"
    if regular_final.exists() or current_page == "final_report.html":
        final_pages.append(("Global evaluation", "final_report.html"))

    threshold_final = run_dir / "final_summary_threshold_optimized.html"
    if threshold_final.exists() or current_page == "final_summary_threshold_optimized.html":
        final_pages.append(("Threshold optimized", "final_summary_threshold_optimized.html"))

    manifest["final_pages"] = [path for _, path in final_pages]


    # ----- models: reports/*.html -----
    reports_dir = run_dir / "reports"
    model_pages: List[Tuple[str, str]] = []
    if reports_dir.exists():
        for p in sorted(reports_dir.glob("*.html")):
            name = _pretty_model_name(p.stem)
            model_pages.append((f"Model: {name}", _to_posix_rel(run_dir, p)))
    manifest["models"] = [path for _, path in model_pages]

    # ----- optional: limma correction -----
    batch_dir = run_dir / "limma_correction"
    batch_pages: List[Tuple[str, str]] = []
    if (batch_dir / "index.html").exists():
        # include known pages if they exist (add more if you create them)
        candidates = [
            ("Overview", batch_dir / "index.html"),
            ("Diagnostic plots", batch_dir / "diagnostic_plots.html"),
            ("PCA comparison", batch_dir / "pca_comparison.html"),
        ]
        for label, fp in candidates:
            if fp.exists():
                batch_pages.append((label, _to_posix_rel(run_dir, fp)))
    manifest["batch"] = [path for _, path in batch_pages]

    # ----- optional: DE -----
    de_page = run_dir / "de_results" / "de_report.html"
    manifest["de"] = _to_posix_rel(run_dir, de_page) if de_page.exists() else None

    # ----- optional: prediction & KM (single report; pick first html) -----
    surv_dir = run_dir / "survival_prediction"
    pred_km_page: Optional[str] = None
    if surv_dir.exists():
        surv_htmls = sorted(surv_dir.glob("*.html"))
        if surv_htmls:
            pred_km_page = _to_posix_rel(run_dir, surv_htmls[0])
    manifest["pred_km"] = pred_km_page

    # ----- optional: limma re-correction -----
    limma_recorr_page = run_dir / "limma_re-correction"
    re_batch_pages: List[Tuple[str, str]] = []
    if (limma_recorr_page / "index.html").exists():
        # include known pages if they exist (add more if you create them)
        candidates = [
            ("Overview", limma_recorr_page / "index.html"),
            ("Diagnostic plots", limma_recorr_page / "diagnostic_plots.html"),
            ("PCA comparison", limma_recorr_page / "pca_comparison.html"),
        ]
        for label, fp in candidates:
            if fp.exists():
                re_batch_pages.append((label, _to_posix_rel(run_dir, fp)))
    manifest["limma_recorrection"] = [path for _, path in re_batch_pages]

    # ----- build tree -----
    tree: List[NavItem] = []

    final_children = []

    if final_pages:
        for lbl, path in final_pages:
            final_children.append(
                NavItem(label=lbl, target=path)
            )
    else:
        final_children.append(
            NavItem(label="Global evaluation", target=None)
        )

    tree.append(
        NavItem(
            label="Final summary",
            expanded=True,
            children=final_children,
        )
    )

    tree.append(
        NavItem(
            label="Models",
            expanded=True,
            children=[NavItem(label=lbl, target=path) for (lbl, path) in model_pages],
        )
    )

    if batch_pages:
        tree.append(
            NavItem(
                label="Batch correction",
                children=[NavItem(label=lbl, target=path) for (lbl, path) in batch_pages],
            )
        )

    if manifest["de"]:
        tree.append(
            NavItem(
                label="Differential expression",
                children=[NavItem(label="DE report", target=manifest["de"])],
            )
        )

    if manifest["pred_km"]:
        tree.append(
            NavItem(
                label="Prediction & Survival",
                children=[NavItem(label="Prediction + KM", target=manifest["pred_km"])],
            )
        )

    if re_batch_pages:
        tree.append(
            NavItem(
                label="Limma re-correction",
                children=[NavItem(label=lbl, target=path) for (lbl, path) in re_batch_pages],
            )
        )

    return tree, manifest
