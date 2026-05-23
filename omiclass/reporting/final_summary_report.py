# omiclass/reporting/final_summary_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..plotting.plotter import FinalSummaryPlotter
from .assets import wrap_report_html, get_inline_assets
from .components import card, svg_panel, json_kv, split_card_1_1, vcard, esc
from omiclass.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html


def _normalize_model_name(name: Optional[str]) -> str:
    if not name:
        return ""
    # Normalize cosmetic differences so selected model highlighting is robust:
    # underscores vs spaces, case differences, and accidental extra whitespace.
    return " ".join(str(name).replace("_", " ").strip().lower().split())

def _pretty_model_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return str(name).replace("_", " ").strip().title()

def _format_table_value(col: str, val: Any) -> Any:
    if col == "model":
        return _pretty_model_name(val)
    if col in ("F1 macro mean", "Accuracy mean", "f1_macro", "accuracy"):
        try:
            v = float(val)
            if pd.isna(v):
                return "—"
            return f"{v:.1f}"
        except Exception:
            return val
    return val


def _render_model_table(
    *,
    df: pd.DataFrame,
    selected_model: Optional[str],
) -> str:
    if df is None or df.empty:
        return "<div class='muted'>No data available.</div>"

    cols = [c for c in df.columns if c in ("model", "F1 macro mean", "Accuracy mean", "f1_macro", "accuracy")]
    if not cols:
        cols = list(df.columns)

    display_cols = ["Model" if c == "model" else c for c in cols]
    head = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in display_cols) + "</tr>"
    rows = []
    for _, r in df[cols].iterrows():
        model_name = str(r.get("model", ""))
        active = bool(
            selected_model
            and _normalize_model_name(model_name) == _normalize_model_name(selected_model)
        )
        row_style = " style='font-weight:700;background:#fff7e6;'" if active else ""
        cells = "".join(f"<td>{esc(_format_table_value(c, r.get(c)))}</td>" for c in cols)
        rows.append(f"<tr{row_style}>{cells}</tr>")

    note = (
        f"<div class='muted'>Highlighted: CV-selected model <code>{esc(_pretty_model_name(selected_model))}</code>.</div>"
        if selected_model
        else ""
    )
    return f"{note}<table class='kv'>{head}{''.join(rows)}</table>"

def clean_leaderboard_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only successful models first; failed/skipped rows often carry NaN metrics.
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    # Guardrail: only keep rows with usable score values.
    metric_candidates = [c for c in ("f1_macro_mean", "accuracy_mean", "f1_macro", "accuracy") if c in df.columns]
    if metric_candidates:
        df = df.dropna(subset=metric_candidates, how="all").copy()

    if "status" in df.columns:
        df = df.drop(columns=["status"])
    if "error" in df.columns:
        df = df.drop(columns=["error"])
    if "f1_macro_mean" in df.columns:
        df["f1_macro_mean"] = pd.to_numeric(df["f1_macro_mean"], errors="coerce") * 100
        df["F1 macro mean"] = df["f1_macro_mean"]
        df.drop(columns=["f1_macro_mean"], inplace=True)
    if "accuracy_mean" in df.columns:
        df["Accuracy mean"] = pd.to_numeric(df["accuracy_mean"], errors="coerce") * 100
        df.drop(columns=["accuracy_mean"], inplace=True)

    return df

def _best_model_holdout_score_html(plotter: FinalSummaryPlotter, selected_model: Optional[str]) -> str:
    if not selected_model:
        return "<div class='muted'>No selected model found.</div>"

    selected_clean = selected_model.replace("_", " ").title()
    selected_norm = _normalize_model_name(selected_model)
    res = None
    for key, payload in plotter.per_model.items():
        if _normalize_model_name(key) == selected_norm:
            res = payload
            break
    if not res:
        return f"<div class='muted'>Selected model <code>{esc(selected_clean)}</code> was not found in holdout payload.</div>"

    y_pred = res.get("y_pred")
    if y_pred is None:
        return f"<div class='muted'>No holdout predictions available for <code>{esc(selected_clean)}</code>.</div>"

    acc = float(accuracy_score(plotter.y_true, y_pred)) * 100.0
    f1m = float(f1_score(plotter.y_true, y_pred, average="macro", zero_division=0)) * 100.0

    return split_card_1_1(
        title="Best Model Holdout Score",
        subtitle=f"Model: {esc(selected_clean)}",
        left_title="Macro F1",
        right_title="Accuracy",
        left_html=vcard("", f"<div style='font-size:2rem;font-weight:700'>{f1m:.1f}%</div>"),
        right_html=vcard("", f"<div style='font-size:2rem;font-weight:700'>{acc:.1f}%</div>"),
    )

def _best_model_holdout_score_threshold_html(plotter: FinalSummaryPlotter, selected_model: Optional[str]) -> str:
    return _best_model_holdout_score_html(plotter, selected_model).replace(
        "Best Model Holdout Score",
        "Best Model Holdout Score (Threshold optimized)",
        1,
    )


def render_final_summary_html(
    *,
    plotter: FinalSummaryPlotter,
    threshold_plotter: Optional[FinalSummaryPlotter] = None,
    include_threshold_appendix: bool = False,
    models_order: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    ctx: Any = None,
    output_name: str = "final_report.html"
) -> Path:
    """
    Uses shared report components + assets.
    """
    # --- generate SVG strings ---
    feature_reduction_svg = None
    selected_model = None
    leaderboard_df = None
    holdout_df = None
    if ctx is not None and hasattr(ctx, "cache"):
        # Prefer feature reduction from the F1-optimized flow when available.
        feature_reduction_svg = (
            ctx.cache.get("feature_reduction_svg_f1_optimized")
            or ctx.cache.get("feature_reduction_svg")
        )
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
    if leaderboard_df is not None:
        leaderboard_df = clean_leaderboard_df(leaderboard_df)
    
    cm_grid_svg = plotter.plot_confusion_matrices(best_model_key=selected_model)
    pr_svg, roc_svg = plotter.plot_macro_pr_and_roc()
    metrics_svg = plotter.plot_metric_barplot()


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

    if leaderboard_df is not None:
        parts.append(
            card(
                "Leaderboard Table",
                _render_model_table(
                    df=leaderboard_df,
                    selected_model=selected_model,
                ),
                #  subtitle that its dispaly in percentage and 
                subtitle="Values displayed as percentages"
            )
        )
    # Key metrics bar plot
    parts.append(
        card(
            "Key performance metrics",
            svg_panel(
                title="Metrics bar plot",
                svg=metrics_svg,
                open_by_default=False,
                collapsible=True,
                download_name="metrics_barplot.svg",
            ),
            subtitle="Includes accuracy / F1 / precision / recall",
        )
    )

    # Macro PR & ROC side-by-side
    parts.append(
        split_card_1_1(
            title="Macro PR & Macro ROC",
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

    parts.append(_best_model_holdout_score_html(plotter, selected_model))

    # Confusion matrix
    parts.append(
        card(
            "Confusion matrix",
            svg_panel(
                title="Confusion matrix",
                svg=cm_grid_svg,
                open_by_default=False,
                collapsible=True,
                download_name="confusion_matrices_grid.svg",
            ),
            subtitle="Row-normalized heatmaps",
        )
    )

    if include_threshold_appendix and threshold_plotter is not None:
        threshold_cm_svg = threshold_plotter.plot_confusion_matrices(best_model_key=selected_model)
        parts.append(
            _best_model_holdout_score_threshold_html(threshold_plotter, selected_model)
        )
        parts.append(
            card(
                "Confusion matrix (Threshold optimized)",
                svg_panel(
                    title="Confusion matrix (threshold optimized)",
                    svg=threshold_cm_svg,
                    open_by_default=False,
                    collapsible=True,
                    download_name="confusion_matrices_grid_threshold_optimized.svg",
                ),
                subtitle="Shown only when threshold optimization improves the selected model F1",
            )
        )
    
    #Feature reduction (optional)
    if feature_reduction_svg:
        parts.append(
            card(
                "Feature selection analysis",
                svg_panel(
                    title="Feature selection analysis",
                    svg=feature_reduction_svg,
                    open_by_default=False,
                    collapsible=True,
                    download_name="feature_reduction.svg",
                ),
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
