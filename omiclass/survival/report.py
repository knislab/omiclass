from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from omiclass.reporting.assets import wrap_report_html, get_inline_assets
from omiclass.reporting.components import card, split_card_1_1, svg_panel, kv_table, esc, panel
from omiclass.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html

import pandas as pd

def _clean_stats_for_kv(stats: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for k, v in (stats or {}).items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        clean[k] = v
    return clean

def _format_threshold_stats(
    stats: Dict[Any, Any],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:

    def fmt_key(k: Any) -> str:
        if isinstance(k, tuple):
            try:
                parts = []
                for x in k:
                    ix = int(x)
                    if class_names is not None and ix < len(class_names):
                        parts.append(str(class_names[ix]))
                    else:
                        parts.append(str(ix + 1))
                return " vs ".join(parts)
            except Exception:
                return str(k)

        try:
            return str(int(k)) if type(k).__name__.startswith("int") else str(k)
        except Exception:
            return str(k)

    def fmt_value(v: Any) -> Any:
        if isinstance(v, dict):
            return {fmt_key(kk): fmt_value(vv) for kk, vv in v.items()}
        return v

    return {fmt_key(k): fmt_value(v) for k, v in (stats or {}).items()}

def render_survival_prediction_html(
    *,
    out_dir: Path,
    model_name: str,
    n_samples: int,
    conf_hist_svg: str,
    conf_violin_svg: str,
    km_svg: Optional[str],
    cox_forest_svg: Optional[str],
    ctx: Optional[Any] = None,
    stats: Optional[Dict[str, Any]] = None,
    threshold_km_svg: Optional[str] = None,
    threshold_cox_forest_svg: Optional[str] = None,
    threshold_stats: Optional[Dict[str, Any]] = None,) -> Path:
    """
    Render a standalone HTML report for the prediction + survival analysis phase,
    using shared report components + assets (portable single-file HTML).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{model_name}_survival_prediction.html"

    stats = _clean_stats_for_kv(stats or {})
    global_p = stats.get("global_p", None)

    # ---------- header metrics ----------
    top_metrics: List[Tuple[str, Any]] = [("N predictions", n_samples)]
    if global_p is not None:
        # keep formatting but avoid exceptions
        try:
            top_metrics.append(("Global log-rank p", f"{float(global_p):.3e}"))
        except Exception:
            top_metrics.append(("Global log-rank p", str(global_p)))

    # ---------- body ----------
    parts: List[str] = []

    # Summary card (small metadata)
    parts.append(
        card(
            "Summary",
            kv_table(
                {
                    "model": model_name.replace("_", " "),
                    "n_samples": n_samples,
                    "global_p": global_p,
                    **{k: v for k, v in stats.items() if k not in {"global_p"}},

                },
                order=["model", "n_samples", "global_p"],
                empty_value="—",
                max_depth=2,
            ),
            subtitle="Prediction-phase survival analysis",
        )
    )

    # Confidence plots side-by-side
    parts.append(
        split_card_1_1(
            title="Confidence diagnostics",
            subtitle="Model confidence distribution over predictions",
            left_title="Confidence histogram",
            right_title="Confidence violin per class",
            left_html=svg_panel(
                title="Confidence histogram",
                svg=conf_hist_svg,
                collapsible=True,
                download_name="confidence_histogram.svg",
            ),
            right_html=svg_panel(
                title="Confidence violin per class",
                svg=conf_violin_svg,
                collapsible=True,
                download_name="confidence_violin.svg",
            ),
        )
    )

    # Kaplan–Meier (optional)
    if km_svg:
        cap = ""
        if global_p is not None:
            try:
                cap = f"Global log-rank p = {float(global_p):.3e}"
            except Exception:
                cap = f"Global log-rank p = {global_p}"
        parts.append(
            card(
                "Kaplan–Meier curves",
                svg_panel(
                    title="Kaplan–Meier curves",
                    svg=km_svg,
                    collapsible=True,
                    open_by_default=False,
                    caption=cap,
                    download_name="km_curves.svg",
                ),
                subtitle="Survival stratified by predicted class / risk group",
            )
        )

    # Cox forest (optional)
    if cox_forest_svg:
        parts.append(
            card(
                "Cox model forest plot",
                svg_panel(
                    title="Cox forest plot",
                    svg=cox_forest_svg,
                    collapsible=True,
                    open_by_default=False,
                    download_name="cox_forest.svg",
                ),
                subtitle="Hazard ratios with confidence intervals",
            )
        )
    # Threshold-optimized survival plots (optional)
    class_names = None
    if hasattr(ctx, "cache") and "labels_for_plot" in ctx.cache:
        class_names = list(ctx.cache["labels_for_plot"]) 
    threshold_stats = _format_threshold_stats(_clean_stats_for_kv(threshold_stats or {}), class_names=class_names)
    threshold_global_p = threshold_stats.get("global_p", None)

    threshold_parts: List[str] = []

    if threshold_km_svg:
        cap = "Predictions generated using F1-optimized thresholds learned from OOF predictions."
        if threshold_global_p is not None:
            try:
                cap += f" Global log-rank p = {float(threshold_global_p):.3e}"
            except Exception:
                cap += f" Global log-rank p = {threshold_global_p}"

        threshold_parts.append(
            svg_panel(
                title="Threshold-optimized Kaplan–Meier curves",
                svg=threshold_km_svg,
                collapsible=True,
                open_by_default=False,
                caption=cap,
                download_name="threshold_optimized_km_curves.svg",
            )
        )

    if threshold_cox_forest_svg:
        threshold_parts.append(
            svg_panel(
                title="Threshold-optimized Cox forest plot",
                svg=threshold_cox_forest_svg,
                collapsible=True,
                open_by_default=False,
                caption="Cox model using threshold-optimized predicted classes.",
                download_name="threshold_optimized_cox_forest.svg",
            )
        )

    if threshold_stats:
        threshold_parts.append(
            panel(
                "Threshold-optimized survival statistics",
                kv_table(
                    threshold_stats,
                    empty_value="—",
                    max_depth=2,
                ),
                open_by_default=False,
            )
        )

    if threshold_parts:
        parts.append(
            panel(
                "Threshold Optimization",
                "\n".join(threshold_parts),
                open_by_default=False,
            )
        )
    

    body_html = "\n".join(parts)
    # sidebar
    base_dir = out_dir.parent
    tree, _ = build_sidebar_from_run_dir(base_dir)
    current_page = Path(html_path).relative_to(base_dir).as_posix() 
    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)

    full_html = wrap_report_html(
        title=f"Prediction & Survival – {model_name}",
        subtitle=f"N predictions: {n_samples}",
        body_html=body_html,
        sidebar_html=sidebar_html,
        assets=get_inline_assets(),
    )

    html_path.write_text(full_html, encoding="utf-8")
    return html_path
