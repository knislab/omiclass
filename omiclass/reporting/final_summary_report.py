# omiclass/reporting/final_summary_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from ..plotting.plotter import FinalSummaryPlotter
from .assets import wrap_report_html, get_inline_assets
from .components import card, svg_panel, json_kv, split_card_1_1, vcard, esc
from omiclass.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html


def _render_model_table(
    *,
    df: pd.DataFrame,
    selected_model: Optional[str],
) -> str:
    if df is None or df.empty:
        return "<div class='muted'>No data available.</div>"

    cols = [c for c in df.columns if c in ("model", "status", "f1_macro_mean", "accuracy_mean", "f1_macro", "accuracy")]
    if not cols:
        cols = list(df.columns)

    head = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"
    rows = []
    for _, r in df[cols].iterrows():
        model_name = str(r.get("model", ""))
        active = bool(selected_model and model_name == selected_model)
        row_style = " style='font-weight:700;background:#fff7e6;'" if active else ""
        cells = "".join(f"<td>{esc(r.get(c))}</td>" for c in cols)
        rows.append(f"<tr{row_style}>{cells}</tr>")

    note = (
        f"<div class='muted'>Highlighted: CV-selected model <code>{esc(selected_model)}</code>.</div>"
        if selected_model
        else ""
    )
    return f"{note}<table class='kv'>{head}{''.join(rows)}</table>"


def render_final_summary_html(
    *,
    plotter: FinalSummaryPlotter,
    models_order: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    ctx: Any = None,
    output_name: str = "final_report.html"
) -> Path:
    """
    Uses shared report components + assets.
    Shows:
      1) Leaderboard
      2) Confusion matrices grid
      3) Key metrics bar plot
      4) Macro PR + Macro ROC
      5) Feature reduction (optional)
    """
    # --- generate SVG strings ---
    lb_svg = plotter.plot_leaderboard_bar()
    cm_grid_svg = plotter.plot_confusion_matrices()
    pr_svg, roc_svg = plotter.plot_macro_pr_and_roc()
    metrics_svg = plotter.plot_metric_barplot()

    feature_reduction_svg = None
    selected_model = None
    leaderboard_df = None
    holdout_df = None
    if ctx is not None and hasattr(ctx, "cache"):
        feature_reduction_svg = ctx.cache.get("feature_reduction_svg")
        selected_model = ctx.cache.get("best_model")
        leaderboard_csv = ctx.cache.get("leaderboard_csv")
        holdout_csv = ctx.cache.get("combined_test_csv")
        try:
            if leaderboard_csv:
                leaderboard_df = pd.read_csv(leaderboard_csv)
        except Exception:
            leaderboard_df = None
        try:
            if holdout_csv:
                holdout_df = pd.read_csv(holdout_csv)
        except Exception:
            holdout_df = None

    # --- build body using components ---
    parts: List[str] = []

    # Run metadata (nice kv table)
    if meta:
        parts.append(
            card(
                "Run Metadata",
                json_kv(meta),
                subtitle="Configuration & run context",
            )
        )

    # 1) Leaderboard
    parts.append(
        card(
            "1) Leaderboard",
            svg_panel(
                title="Leaderboard",
                svg=lb_svg,
                open_by_default=True,
                collapsible=True,
                download_name="leaderboard.svg",
            ),
        )
    )
    if leaderboard_df is not None:
        parts.append(
            card(
                "1b) CV Leaderboard Table",
                _render_model_table(
                    df=leaderboard_df,
                    selected_model=selected_model,
                ),
                subtitle="Model selection is based on CV leaderboard only.",
            )
        )

    # 2) Confusion matrices
    parts.append(
        card(
            "2) Confusion matrices",
            svg_panel(
                title="Confusion matrices grid",
                svg=cm_grid_svg,
                open_by_default=False,
                collapsible=True,
                download_name="confusion_matrices_grid.svg",
            ),
            subtitle="Row-normalized heatmaps (per model)",
        )
    )

    # 3) Key metrics bar plot
    parts.append(
        card(
            "3) Key performance metrics",
            svg_panel(
                title="Metrics bar plot",
                svg=metrics_svg,
                open_by_default=False,
                collapsible=True,
                download_name="metrics_barplot.svg",
            ),
            subtitle="Includes accuracy / F1 / AUROC / LogLoss (as available)",
        )
    )
    if holdout_df is not None:
        parts.append(
            card(
                "3b) Holdout Metrics Table",
                _render_model_table(
                    df=holdout_df,
                    selected_model=selected_model,
                ),
                subtitle="All models are shown for transparency; highlighted row remains the CV-selected model.",
            )
        )

    # 4) Macro PR & ROC side-by-side
    parts.append(
        split_card_1_1(
            title="4) Macro PR & Macro ROC",
            subtitle="Overlay curves across models (macro averaging)",
            left_title="Macro PR",
            right_title="Macro ROC",
            left_html=svg_panel(
                title="Macro PR",
                svg=pr_svg,
                collapsible=True,
                open_by_default=False,
                download_name="macro_pr.svg",
            ),
            right_html=svg_panel(
                title="Macro ROC",
                svg=roc_svg,
                collapsible=True,
                open_by_default=False,
                download_name="macro_roc.svg",
            ),
        )
    )

    # 5) Feature reduction (optional)
    if feature_reduction_svg:
        parts.append(
            card(
                "5) Feature selection analysis",
                svg_panel(
                    title="Feature selection analysis",
                    svg=feature_reduction_svg,
                    open_by_default=False,
                    collapsible=True,
                    download_name="feature_reduction.svg",
                ),
                subtitle="Performance vs. #genes ranked by permutation importance on training folds",
            )
        )

    body_html = "\n".join(parts)
    out_html = plotter.out_dir / output_name
    current_page = Path(out_html).relative_to(plotter.out_dir).as_posix() 
    tree, _ = build_sidebar_from_run_dir(plotter.out_dir, current_page=current_page)
    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)


    # --- wrap into full report shell ---
    full_html = wrap_report_html(
        title="Final Summary Report",
        subtitle="Model comparison & diagnostics",
        body_html=body_html,
        sidebar_html=sidebar_html,
        assets=get_inline_assets(),
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(full_html, encoding="utf-8")
    return out_html
