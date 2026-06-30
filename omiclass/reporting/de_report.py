# generic-classifier.reporting.de_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Union, Optional, List, Tuple
import datetime
import pandas as pd
from omiclass.reporting.assets import wrap_report_html
from omiclass.reporting.components import card, kv_table, metric_grid, chips_panel, esc, vertical_metric_grid, split_card_3_1, svg_panel
from omiclass.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html
from omiclass.utils.gene_symbols import get_gene_symbols
from omiclass.utils.toolbox import get_class_display_name


def write_de_geneset_report_html(
    *,
    output_dir: Path,
    de_results_csv: Path,
    volcano_svgs: Optional[List[Tuple[str, str]]] = None,
    design_col: str,
    n_classes: int,
    K: int,
    fdr_thresh: float,
    logfc_thresh: float,
    min_count: float,
    min_samples_frac: float,
    geneset_summary: Union[Dict[str, Any], None],
    combined_genes: Union[Sequence[str], None],
    covariates: Union[Sequence[str], None],
    html_name: str = "de_report.html",
    cfg: Optional[Dict[str, Any]] = None,
    analysis_label: str = "Differential Expression",
    feature_label_singular: str = "gene",
    feature_label_plural: str = "genes",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tree, _ = build_sidebar_from_run_dir(output_dir)

    report_dir = output_dir / "de_results"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / html_name
    current_page = out_path.relative_to(output_dir).as_posix()

    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)


    # Normalize inputs
    volcano_svgs = volcano_svgs or []
    combined_genes = list(combined_genes or [])
    covariates = list(covariates or [])

    def gene_chip_items(gene_ids: Sequence[str]) -> List[Dict[str, str]]:
        gene_ids = list(gene_ids or [])
        symbols = list(get_gene_symbols(gene_ids, cfg=cfg) or [])

        return [
            {
                "id": str(gid),
                "symbol": str(sym) if sym else "",
            }
            for gid, sym in zip(gene_ids, symbols)
        ]

    # Load DE counts (avoid heavy tables)
    df = pd.read_csv(de_results_csv, index_col=0)
    # Build config metadata
    meta = {
        "Timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "Design column": design_col or "—",
        "Covariates": ", ".join(covariates) if covariates else "—",
        "Number of classes": int(n_classes),
        f"Selected {feature_label_plural} (K)": int(K),
        "FDR threshold": fdr_thresh,
        "log2FC threshold": logfc_thresh,
        "Min count": min_count,
        "Min samples fraction": min_samples_frac,
        # "DE genes detected": int(len(df)),
        # "Volcano plots": int(len(volcano_svgs)),
    }

    # Summary metrics
    n_one_vs_rest = len((geneset_summary or {}).get("one_vs_rest_genes", []) or []) if geneset_summary else 0
    n_pairwise = len((geneset_summary or {}).get("pairwise_genes", []) or []) if geneset_summary else 0
    selection_summary = (geneset_summary or {}).get("selection_summary", []) or []
    n_filtered_contrasts = sum(1 for row in selection_summary if str(row.get("selection_pool", "")) == "filtered")
    n_fallback_contrasts = sum(1 for row in selection_summary if str(row.get("selection_pool", "")) == "fallback")
    n_final = len(combined_genes)
    summary_html = vertical_metric_grid(
        [
            (f"DE {feature_label_plural}", f"{len(df):,}"),
            ("One-vs-rest markers", f"{n_one_vs_rest:,}"),
            ("Close-pair markers", f"{n_pairwise:,}"),
            ("Filtered contrasts", f"{n_filtered_contrasts:,}"),
            ("Fallback contrasts", f"{n_fallback_contrasts:,}"),
            (f"Final {feature_label_singular}-set", f"{n_final:,}"),
            ("Volcano plots", f"{len(volcano_svgs):,}"),
        ]
    )

    # Gene panels
    genes_html_parts: List[str] = []

    if n_classes <= 2:
        genes_html_parts.append(
            chips_panel(
                panel_id="genes_selected",
                title=f"Selected {feature_label_plural}",
                items=gene_chip_items(combined_genes),
                
                open_by_default=False,
                searchable=True,
                copyable=True,
            )
        )
    else:
        if geneset_summary:
            for c in (geneset_summary.get("contrast_order", []) or []):
                items = (geneset_summary.get("by_contrast", {}) or {}).get(c, []) or []
                class_name = get_class_display_name(c, cfg=cfg, class_values=(geneset_summary.get("contrast_order", []) or []))
                pid = f"genes_{str(c).replace(' ', '_').replace('/', '_')}"
                genes_html_parts.append(
                    chips_panel(
                        panel_id=pid,
                        title=f"Class-specific: {class_name}",
                        items=gene_chip_items(items),
                        open_by_default=False,
                        searchable=True,
                        copyable=True,
                    )
                )
            pairwise_order = geneset_summary.get("pairwise_contrast_order", []) or []
            pairwise_by_contrast = geneset_summary.get("pairwise_by_contrast", {}) or {}
            class_values = geneset_summary.get("contrast_order", []) or []
            for i, contrast in enumerate(pairwise_order):
                display_name = get_class_display_name(
                    contrast,
                    cfg=cfg,
                    class_values=class_values,
                )
                genes_html_parts.append(
                    chips_panel(
                        panel_id=f"genes_pairwise_close_class_{i}",
                        title=f"Close-class pairwise: {display_name}",
                        items=gene_chip_items(pairwise_by_contrast.get(str(contrast), []) or []),
                        open_by_default=False,
                        searchable=True,
                        copyable=True,
                    )
                )

        genes_html_parts.append(
            chips_panel(
                panel_id="genes_final",
                title=f"Final combined {feature_label_singular}-set",
                items=gene_chip_items(combined_genes),
                open_by_default=False,
                searchable=True,
                copyable=True,
            )
        )

    genes_html = "".join(genes_html_parts) if genes_html_parts else f"<div class='muted'>No {feature_label_plural} provided</div>"

    figs = ""
    for i, (caption, svg) in enumerate(volcano_svgs, start=1):
        safe_name = f"volcano_{i}.svg"
        figs += svg_panel(
            title=caption,
            svg=svg,
            caption="",
            open_by_default=False,
            collapsible=True,
            download_name=f"volcano_{i}.svg",
        )

    volcano_html = figs if figs else "<div class='muted'>No plots found</div>"

    # Assemble report
    side_by_side = split_card_3_1(
    title="Configuration Summary",
    subtitle=f"Parameters and settings used for {analysis_label.lower()} analysis.",
    left_title="Settings",
    right_title="Results",
    left_html=kv_table(meta),
    right_html=summary_html,
)
    


    body_html = (
        side_by_side
        + card(f"Selected {feature_label_plural.title()}", genes_html)
        + card("Volcano Plots", volcano_html)
    )


    # Render full HTML
    final_html = wrap_report_html(
        title=f"{analysis_label} Report",
        subtitle=f"{analysis_label} and {feature_label_singular}-set selection summary.",
        body_html=body_html,
        sidebar_html=sidebar_html,
    )


    out_path.write_text(final_html, encoding="utf-8")
    return out_path
