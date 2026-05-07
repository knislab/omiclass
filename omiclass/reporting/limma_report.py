# generic_classifier/reporting/limma_report.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime
import json
import re

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from generic_classifier.reporting.assets import wrap_report_html, get_inline_assets
from generic_classifier.reporting.components import card, kv_table, metric_grid, panel, esc, svg_panel, split_card_1_1, nav_buttons
from generic_classifier.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html


# ----------------
# DATACLASSES
# ----------------

@dataclass
class LimmaRun:
    path: str
    timestamp: str


@dataclass
class LimmaData:
    expr_shape: List[int]
    n_genes: int
    n_samples: int
    meta_cols: List[str]


@dataclass
class LimmaConfig:
    state: int
    batch_mode: str  # "batch", "covariates", "pcs"
    batch_col: Optional[str]
    design_cols: Optional[List[str]]
    n_pc: int
    na_fill: Optional[float]


@dataclass
class LimmaEval:
    labels: List[str]
    silhouette_scores: Dict[str, float]
    silhouette_scores_orig: Dict[str, float]
    gene_level_orig: Optional[Dict[str, Any]] = None
    gene_level: Optional[Dict[str, Any]] = None


@dataclass
class LimmaFigures:
    heatmap: Optional[str] = None
    pca: Optional[Dict[str, Dict[str, str]]] = None
    boxplots: Optional[Dict[str, str]] = None
    dendrogram: Optional[str] = None
    top_batch_gene_boxplot: Optional[str] = None


@dataclass
class LimmaReport:
    run: LimmaRun
    data: LimmaData
    config: LimmaConfig
    eval: LimmaEval
    figures_corrected: LimmaFigures
    figures_original: LimmaFigures


# ----------------
# PATHS
# ----------------

def limma_report_json_path(out_dir: Path) -> Path:
    return out_dir / "limma.report.json"


def limma_index_html_path(out_dir: Path) -> Path:
    return out_dir / "index.html"


def limma_pca_comparison_html_path(out_dir: Path) -> Path:
    return out_dir / "pca_comparison.html"


def limma_diagnostic_html_path(out_dir: Path) -> Path:
    return out_dir / "diagnostic_plots.html"


# ----------------
# JSON helpers
# ----------------

def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    try:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass

    if pd is not None:
        try:
            if isinstance(o, (pd.Timestamp,)):
                return o.isoformat()
            if isinstance(o, (pd.Series, pd.Index)):
                return o.tolist()
            if isinstance(o, (pd.DataFrame,)):
                return o.to_dict(orient="records")
        except Exception:
            pass
    return str(o)


def _gene_level_summary(gl: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not gl or not isinstance(gl, dict):
        return None
    return {
        "batch_like_name": gl.get("label_name"),
        "fdr": gl.get("fdr"),
        "n_genes_tested": gl.get("n_genes_tested"),
        "n_significant_fdr": gl.get("n_significant_fdr"),
    }


# ----------------
# Shared UI helpers (components-based)
# ----------------

def _extend_assets_css(extra_css: str):
    a = get_inline_assets()
    return a.__class__(css_text=a.css_text + ("\n" + extra_css if extra_css else ""), js_text=a.js_text)

def _svg_block(
    *,
    site_dir: Path,
    title: str,
    src: Optional[str],
    open_by_default: bool = False,
) -> str:
    svg = _read_svg_text(site_dir=site_dir, src=src or "")
    if not svg:
        return "<div class='muted'>Missing figure</div>"

    # Render zoomable SVG (but NOT collapsible here)
    inner = svg_panel(
        title="",               
        svg=svg,
        caption="",
        open_by_default=True,     
        collapsible=False,
        download_name=f"{title.replace(' ', '_').lower()}.svg",
    )

    # Wrap with collapsible details 
    return panel(
        summary=title,
        body_html=inner,
        open_by_default=open_by_default,
    )


def _header_card(*, run: Dict[str, Any], active: str, title: str, subtitle: str = "") -> str:
    run_path = run.get("path", "")
    ts = run.get("timestamp", "")
    sub = f"<div class='subtle'>{esc(subtitle)}</div>" if subtitle else ""
    nav = nav_buttons([
        ("Overview & Stats", "index.html", active == "index"),
        ("PCA Comparison", "pca_comparison.html", active == "pca"),
        ("Diagnostics", "diagnostic_plots.html", active == "diag"),
    ])

    return card(
        title,
        body_html=f"""
    {sub}
    <div class="muted">Run: <code>{esc(run_path)}</code> • {esc(ts)}</div>
    {nav}
    """.strip(),
    )


_SVG_MARKUP_RE = re.compile(r"^\s*(<\?xml\b[^>]*\?>\s*)?<svg\b", re.IGNORECASE | re.DOTALL)


def _looks_like_svg_markup(s: str) -> bool:
    if not s:
        return False
    # quick cheap checks
    if "<svg" in s[:500].lower():
        return True
    if s.lstrip().startswith("<?xml"):
        return True
    return bool(_SVG_MARKUP_RE.search(s[:2000]))


def _read_svg_text(site_dir: Path, src: str) -> Optional[str]:
    """
    Read SVG markup from either:
      - an inline SVG markup string (src contains <svg ...>)
      - a filesystem path (relative to site_dir, or absolute)
    """
    if not src:
        return None

    # 1) INLINE SVG MARKUP CASE (THIS IS YOUR CURRENT LIMMA ISSUE)
    if _looks_like_svg_markup(src):
        return src

    # 2) PATH CASE
    try:
        p = Path(src)
        if not p.is_absolute():
            p = (site_dir / src)
        p = p.resolve()
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

def _extra_css_for_svgs() -> str:
    # CSS only (wrap_report_html already wraps assets.css_text in <style>).
    return """
/* Ensure huge SVGs aren't clipped */
.panel-body { overflow: visible; }

/* Optional: ensure cards never clip, too */
.card { overflow: visible; }
""".strip()


# ----------------
# PAGE RENDERERS
# ----------------

def _render_index_html(d: Dict[str, Any], site_dir: Path) -> str:
    run = d["run"]
    data = d["data"]
    cfg = d["config"]
    ev = d["eval"]

    labels = sorted(ev.get("labels") or [])

    metric_items = []
    for name in labels:
        corr = ev["silhouette_scores"].get(name)
        orig = ev["silhouette_scores_orig"].get(name)

        corr_txt = "—"
        sub = ""
        if corr is not None and not np.isnan(float(corr)):
            corr_txt = f"{float(corr):.4f}"

        if orig is not None and corr is not None and (not np.isnan(float(orig))) and (not np.isnan(float(corr))):
            diff = float(corr) - float(orig)
            sub = f"orig {float(orig):.4f} ({'+' if diff >= 0 else ''}{diff:.4f})"
        elif orig is not None and not np.isnan(float(orig)):
            sub = f"orig {float(orig):.4f}"

        label_txt = f"{name}" + (f" • {sub}" if sub else "")
        metric_items.append((label_txt, corr_txt))

    sil_metrics = metric_grid(metric_items, columns=4) if metric_items else "<div class='muted'>No silhouette scores.</div>"

    rows = []
    for name in labels:
        orig_val = ev["silhouette_scores_orig"].get(name)
        corr_val = ev["silhouette_scores"].get(name)

        def fmt(v):
            return f"{float(v):.4f}" if v is not None and not np.isnan(float(v)) else "—"

        rows.append(
            f"<tr><th>{esc(name)}</th>"
            f"<td>Original: {esc(fmt(orig_val))} &nbsp;•&nbsp; Corrected: {esc(fmt(corr_val))}</td></tr>"
        )
    sil_table_html = f"<table class='kv'>{''.join(rows)}</table>"

    gl_orig = ev.get("gene_level_orig") or {}
    gl_corr = ev.get("gene_level") or {}
    gene_card_html = ""
    if gl_orig or gl_corr:
        name = gl_corr.get("batch_like_name") or gl_orig.get("batch_like_name") or "batch"
        fdr = gl_corr.get("fdr") or gl_orig.get("fdr")
        fdr_txt = f"{float(fdr):.3f}" if fdr is not None else "—"

        n0 = gl_orig.get("n_significant_fdr")
        n1 = gl_corr.get("n_significant_fdr")
        t0 = gl_orig.get("n_genes_tested")
        t1 = gl_corr.get("n_genes_tested")

        gene_metrics = metric_grid(
            [
                ("Original significant", f"{int(n0):,}" if n0 is not None else "—"),
                ("Corrected significant", f"{int(n1):,}" if n1 is not None else "—"),
                ("Tested (orig)", f"{int(t0):,}" if t0 is not None else "—"),
                ("Tested (corr)", f"{int(t1):,}" if t1 is not None else "—"),
            ],
            columns=4,
        )
        gene_card_html = card(
            "Gene-level batch association",
            f"""
<div class="muted">Label: <code>{esc(name)}</code> • FDR: {esc(fdr_txt)}</div>
<div style="margin-top:10px;">{gene_metrics}</div>
""".strip(),
        )

    data_tbl = kv_table(
        {
            "Expression shape": tuple(data["expr_shape"]),
            "Genes": data["n_genes"],
            "Samples": data["n_samples"],
            "Metadata columns": ", ".join(data.get("meta_cols") or []),
        }
    )
    cfg_tbl = kv_table(
        {
            "State": cfg["state"],
            "Batch mode": cfg["batch_mode"],
            "Batch column": cfg["batch_col"] or "—",
            "Design cols": ", ".join(cfg.get("design_cols") or []) or "—",
            "n_pc (pcs mode)": cfg["n_pc"],
            "na_fill": cfg["na_fill"],
        }
    )

    body = (
        _header_card(
            run=run,
            active="index",
            title="Limma Batch-Correction Evaluation",
            subtitle="Overview & summary statistics.",
        )
        + f"""
<div class="grid grid-2">
  {card("Data", data_tbl)}
  {card("Configuration", cfg_tbl)}
</div>
"""
        + card(
            "Silhouette scores (PC space)",
            f"{sil_metrics}<div style='margin-top:12px;'>{sil_table_html}</div>",
        )
        + gene_card_html
    )
    out_html = site_dir / "index.html"
    base_dir = site_dir.parent
    tree, _ = build_sidebar_from_run_dir(base_dir)
    current_page = Path(out_html).relative_to(base_dir).as_posix() 
    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)

    return wrap_report_html(
        title="Limma Batch-Correction Evaluation — Overview",
        subtitle="Overview & stats",
        body_html=body,
        assets=_extend_assets_css(_extra_css_for_svgs()),
        sidebar_html=sidebar_html,
    )



def _render_pca_comparison_html(d: Dict[str, Any], site_dir: Path) -> str:
    run = d["run"]
    orig = d["figures_original"]
    corr = d["figures_corrected"]

    blocks: List[str] = []

    blocks.append(
        split_card_1_1(
            title="Hierarchical clustering dendrogram",
            subtitle="Original vs corrected.",
            left_title="Original",
            right_title="Corrected",
            left_html=_svg_block(site_dir=site_dir, title="Hierarchical clustering dendrogram", src=orig.get("dendrogram"), open_by_default=False),
            right_html=_svg_block(site_dir=site_dir, title="Hierarchical clustering dendrogram", src=corr.get("dendrogram"), open_by_default=False),
        )
    )

    pca_o = orig.get("pca") or {}
    pca_c = corr.get("pca") or {}
    labels = sorted(set(pca_o.keys()) | set(pca_c.keys()))

    for lab in labels:
        o_paths = pca_o.get(lab) or {}
        c_paths = pca_c.get(lab) or {}

        blocks.append(
            split_card_1_1(
                title=f"PCA PC1–2 by {lab}",
                subtitle="Original vs corrected.",
                left_title="Original",
                right_title="Corrected",
                left_html=_svg_block(site_dir=site_dir, title=f"PC1–2 by {lab}", src=o_paths.get("pc1_pc2"), open_by_default=False),
                right_html=_svg_block(site_dir=site_dir, title=f"PC1–2 by {lab}", src=c_paths.get("pc1_pc2"), open_by_default=False),
            )
        )

        blocks.append(
            split_card_1_1(
                title=f"PCA PC3–4 by {lab}",
                subtitle="Original vs corrected.",
                left_title="Original",
                right_title="Corrected",
                left_html=_svg_block(site_dir=site_dir, title=f"PC3–4 by {lab}", src=o_paths.get("pc3_pc4"), open_by_default=False),
                right_html=_svg_block(site_dir=site_dir, title=f"PC3–4 by {lab}", src=c_paths.get("pc3_pc4"), open_by_default=False),
            )
        )

    body = (
        _header_card(
            run=run,
            active="pca",
            title="Limma Batch-Correction Evaluation",
            subtitle="PCA comparison (Original vs corrected).",
        )
        + ("".join(blocks) if blocks else card("PCA comparison", "<div class='muted'>No PCA/dendrogram figures recorded.</div>"))
    )
    out_html = site_dir / "pca_comparison.html"
    base_dir = site_dir.parent
    tree, _ = build_sidebar_from_run_dir(base_dir)
    current_page = Path(out_html).relative_to(base_dir).as_posix() 
    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)

    return wrap_report_html(
        title="Limma Batch-Correction Evaluation — PCA Comparison",
        subtitle="Original vs corrected",
        body_html=body,
        sidebar_html=sidebar_html,
        assets=get_inline_assets(),
    )



def _render_diagnostic_html(d: Dict[str, Any], site_dir: Path) -> str:
    run = d["run"]
    orig = d["figures_original"]
    corr = d["figures_corrected"]

    blocks: List[str] = []

    blocks.append(
        split_card_1_1(
            title="PC vs variables (−log10 p) heatmap",
            subtitle="Original vs corrected.",
            left_title="Original",
            right_title="Corrected",
            left_html=_svg_block(site_dir=site_dir, title="PC vs variables Heatmap", src=orig.get("heatmap"), open_by_default=False),
            right_html=_svg_block(site_dir=site_dir, title="PC vs variables Heatmap", src=corr.get("heatmap"), open_by_default=False),
        )
    )

    box_o = orig.get("boxplots") or {}
    box_c = corr.get("boxplots") or {}
    labels = sorted(set(box_o.keys()) | set(box_c.keys()))
    for lab in labels:
        blocks.append(
            split_card_1_1(
                title=f"PC1–4 boxplots by {lab}",
                subtitle="Original vs corrected.",
                left_title="Original",
                right_title="Corrected",
                left_html=_svg_block(site_dir=site_dir, title=f"PC1–4 boxplots by {lab}", src=box_o.get(lab), open_by_default=False),
                right_html=_svg_block(site_dir=site_dir, title=f"PC1–4 boxplots by {lab}", src=box_c.get(lab), open_by_default=False),
            )
        )

    blocks.append(
        split_card_1_1(
            title="Top batch-associated genes (feature-level)",
            subtitle="Original vs corrected.",
            left_title="Original",
            right_title="Corrected",
            left_html=_svg_block(site_dir=site_dir, title="Top batch-associated genes", src=orig.get("top_batch_gene_boxplot"), open_by_default=False),
            right_html=_svg_block(site_dir=site_dir, title="Top batch-associated genes", src=corr.get("top_batch_gene_boxplot"), open_by_default=False),
        )
    )

    body = (
        _header_card(
            run=run,
            active="diag",
            title="Limma Batch-Correction Evaluation",
            subtitle="Diagnostics (Original vs corrected).",
        )
        + ("".join(blocks) if blocks else card("Diagnostics", "<div class='muted'>No diagnostic figures recorded.</div>"))
    )
    out_html = site_dir / "diagnostic_plots.html"
    base_dir = site_dir.parent
    tree, _ = build_sidebar_from_run_dir(base_dir)
    current_page = Path(out_html).relative_to(base_dir).as_posix() 
    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)

    return wrap_report_html(
        title="Limma Batch-Correction Evaluation — Diagnostics",
        subtitle="Deep diagnostics",
        body_html=body,
        sidebar_html=sidebar_html,
        assets=get_inline_assets(),
    )



# ----------------
# MAIN WRITER
# ----------------

def write_limma_report(
    out_dir: Path,
    *,
    expr_shape: tuple[int, int],
    meta_cols: List[str],
    batch_col: Optional[str],
    design_cols: Optional[List[str]],
    n_pc: int,
    na_fill: Optional[float],
    limma_result: Dict[str, Any],
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rel_plot(p: Optional[str]) -> Optional[str]:
        # keep as-is (may be relative path OR inline svg string)
        return p

    state = int(limma_result.get("state", 0))
    if state in (1, 2):
        batch_mode = "batch"
    elif state in (3, 4):
        batch_mode = "covariates"
    elif state in (5, 6):
        batch_mode = "pcs"
    else:
        batch_mode = "unknown"

    sil = limma_result.get("silhouette_scores") or {}
    plots = limma_result.get("plots") or {}
    gene_level = limma_result.get("gene_level") or None

    sil_orig = limma_result.get("silhouette_scores_orig") or {}
    plots_orig = limma_result.get("plots_orig") or {}
    gene_level_orig = limma_result.get("gene_level_orig") or None

    def _process_figs(raw_plots: Dict[str, Any]) -> LimmaFigures:
        raw_heat = raw_plots.get("heatmap")
        heat_rel = _rel_plot(raw_heat) if isinstance(raw_heat, str) else None

        raw_pca = raw_plots.get("pca") or {}
        pca_rel: Dict[str, Dict[str, str]] = {}
        for label, paths in raw_pca.items():
            if not isinstance(paths, dict):
                continue
            pca_rel[label] = {}
            for k, v in paths.items():
                if isinstance(v, str) and v:
                    pca_rel[label][k] = _rel_plot(v)

        raw_box = raw_plots.get("boxplots") or {}
        box_rel: Dict[str, str] = {
            label: _rel_plot(path)
            for label, path in raw_box.items()
            if isinstance(path, str) and path
        }

        dendrogram = raw_plots.get("dendrogram")
        dendrogram_rel = _rel_plot(dendrogram) if isinstance(dendrogram, str) else None

        top_box = raw_plots.get("top_batch_gene_boxplot") or raw_plots.get("top_batch_gene_boxplots")
        top_box_rel = _rel_plot(top_box) if isinstance(top_box, str) else None

        return LimmaFigures(
            heatmap=heat_rel,
            pca=pca_rel or None,
            boxplots=box_rel or None,
            dendrogram=dendrogram_rel,
            top_batch_gene_boxplot=top_box_rel,
        )

    figs_corrected = _process_figs(plots)
    figs_original = _process_figs(plots_orig)

    doc = LimmaReport(
        run=LimmaRun(path=str(out_dir), timestamp=datetime.datetime.now().isoformat(timespec="seconds")),
        data=LimmaData(
            expr_shape=[int(expr_shape[0]), int(expr_shape[1])],
            n_genes=int(expr_shape[0]),
            n_samples=int(expr_shape[1]),
            meta_cols=list(meta_cols),
        ),
        config=LimmaConfig(
            state=state,
            batch_mode=batch_mode,
            batch_col=batch_col,
            design_cols=design_cols or [],
            n_pc=int(n_pc),
            na_fill=float(na_fill) if na_fill is not None else None,
        ),
        eval=LimmaEval(
            labels=list(sil.keys()),
            silhouette_scores={str(k): float(v) for k, v in sil.items()},
            silhouette_scores_orig={str(k): float(v) for k, v in sil_orig.items()},
            gene_level_orig=_gene_level_summary(gene_level_orig),
            gene_level=_gene_level_summary(gene_level),
        ),
        figures_corrected=figs_corrected,
        figures_original=figs_original,
    )

    # JSON
    jpath = limma_report_json_path(out_dir)
    jpath.write_text(json.dumps(asdict(doc), indent=2, default=_json_default), encoding="utf-8")

    dd = asdict(doc)

    # HTMLs
    index_path = limma_index_html_path(out_dir)
    pca_path = limma_pca_comparison_html_path(out_dir)
    diag_path = limma_diagnostic_html_path(out_dir)

    index_path.write_text(_render_index_html(dd, out_dir), encoding="utf-8")
    pca_path.write_text(_render_pca_comparison_html(dd, out_dir), encoding="utf-8")
    diag_path.write_text(_render_diagnostic_html(dd, out_dir), encoding="utf-8")

    return index_path
