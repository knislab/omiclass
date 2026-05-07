# generic_classifier/reporting/per_model_report.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

from generic_classifier.reporting.assets import wrap_report_html
from generic_classifier.reporting.components import card, kv_table, metric_grid, panel, chips_panel, esc, svg_panel

from generic_classifier.plotting.geneset_plotter import plot_geneset_top_frequency_bar
from generic_classifier.reporting.nav import build_sidebar_from_run_dir, render_sidebar_html
from generic_classifier.utils.gene_symbols import get_gene_symbols



# -----------------------------
# Datamodel (kept local)
# -----------------------------

@dataclass
class RunInfo:
    path: str
    timestamp: str

@dataclass
class DataSnapshot:
    x_shape: List[int]
    y_len: int
    class_counts: Dict[str, int]
    class_name_map: Optional[Dict[str, str]] = None

@dataclass
class Provenance:
    git: Optional[str] = None
    python: Optional[str] = None
    env: Optional[str] = None
    seed: Optional[int] = None
    cfg_short: Optional[Dict[str, Any]] = None

@dataclass
class CVSummary:
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    per_fold: List[Dict[str, Any]]

@dataclass
class TuningInfo:
    final_params_full_data: Dict[str, Any]
    cv_inner_hint: Optional[List[Dict[str, Any]]] = None

@dataclass
class TestResult:
    metrics: Dict[str, float]
    per_class: List[Dict[str, Any]]

@dataclass
class Downloads:
    model: Optional[str] = None
    predictions: Optional[str] = None
    cv_results: Optional[str] = None
    importances: Optional[str] = None
    shap: Optional[str] = None
    other: Optional[Dict[str, str]] = None

@dataclass
class ReportDoc:
    run: RunInfo
    model: Dict[str, Any]
    data: DataSnapshot
    provenance: Provenance
    cv: CVSummary
    tuning: TuningInfo
    geneset: Optional[Dict[str, Any]] = None
    test: Optional[TestResult] = None
    downloads: Optional[Downloads] = None


# -----------------------------
# Paths / JSON IO
# -----------------------------

def _reports_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "reports"

def report_json_path(run_dir: Path, model_name: str) -> Path:
    return _reports_dir(run_dir) / f"{model_name}.report.json"

def report_html_path(run_dir: Path, model_name: str) -> Path:
    return _reports_dir(run_dir) / f"{model_name}.html"

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
    try:
        import pandas as _pd
        if isinstance(o, (_pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, (_pd.Series, _pd.Index)):
            return o.tolist()
        if isinstance(o, (_pd.DataFrame,)):
            return o.to_dict(orient="records")
    except Exception:
        pass
    return str(o)

def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# -----------------------------
# Gene-set block (kept local)
# -----------------------------

def _build_geneset_block(
    *,
    cvres: Any,
    per_fold: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Pure per-model CV reporting: build a small dict that the HTML can render.
    No dependency on other files.
    """
    consensus_genes = getattr(cvres, "geneset_consensus_genes", None)
    fs_full_info = getattr(cvres, "geneset_full_info", None)
    fs_cfg = getattr(cvres, "geneset_cfg_short", {}) or {}

    if not consensus_genes:
        return None

    consensus_genes = list(map(str, consensus_genes))
    consensus_min_genes = int(fs_cfg.get("consensus_min_genes", 20) or 20)

    per_fold_gs: List[Dict[str, Any]] = []
    for i, m in enumerate(per_fold):
        genes = m.get("selected_features") or []
        k = m.get("n_features")
        if genes:
            per_fold_gs.append({"fold": i, "k": k, "genes": list(map(str, genes))})

    top_plot_svg: Optional[str] = None
    top_by_frequency: List[List[Any]] = []

    gene_df = None
    if isinstance(fs_full_info, pd.DataFrame) and not fs_full_info.empty and "section" in fs_full_info.columns:
        gene_df = fs_full_info[fs_full_info["section"] == "gene"].copy()

    if gene_df is not None and not gene_df.empty:
        gene_df["count_folds"] = pd.to_numeric(gene_df.get("count_folds"), errors="coerce").fillna(0).astype(int)
        gene_df["gene"] = gene_df.get("gene").astype(str)
        gene_df = gene_df.sort_values(["count_folds", "gene"], ascending=[False, True])

        topN = gene_df.head(consensus_min_genes)
        top_by_frequency = [[str(r["gene"]), int(r["count_folds"])] for _, r in topN.iterrows()]

        # optional svg plot (stays inline svg string)
        top_plot_svg = plot_geneset_top_frequency_bar(
            gene_df=gene_df,
            top_n=consensus_min_genes,
            title=f"Top {consensus_min_genes} genes by fold-frequency",
        )

    return {
        "enabled": True,
        "consensus_genes": consensus_genes,
        "consensus_size": int(len(consensus_genes)),
        "params": dict(fs_cfg),
        "per_fold": per_fold_gs,
        "top_by_frequency": top_by_frequency,
        "top_freq_svg": top_plot_svg, 
    }

def clear_name(name):
    if name.endswith("NoEarlyStop"):
        return name[:-len("NoEarlyStop")]
    return name


# -----------------------------
# Public API
# -----------------------------

def report_cv(
    *,
    cvres: Any,
    artifacts_dir: Path,
    model_name: str,
    estimator_cls_name: str,
    X_shape: tuple[int, int],
    y_series: pd.Series,
    seed: int,
    cfg_short: Optional[Dict[str, Any]] = None,
    final_params_full_data: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[List[str]] = None,
    class_name_map: Optional[Dict[int, str]] = None,
) -> Path:
    """
    CV stage writer: creates reports/{model}.report.json with CV + tuning info.
    """
    per_fold = [dict(m) for m in (getattr(cvres, "per_fold_metrics", None) or [])]
    df = pd.DataFrame(per_fold) if per_fold else pd.DataFrame()

    base_cols = ["accuracy", "f1_macro", "auroc_macro", "auprc_macro", "logloss"]
    if extra_metrics:
        base_cols += list(extra_metrics)
    present = [c for c in base_cols if c in df.columns]

    cv_mean = {m: float(df[m].mean()) for m in present} if not df.empty else {}
    cv_std  = {m: float(df[m].std(ddof=0)) for m in present} if not df.empty else {}

    run_dir = Path(artifacts_dir).parent
    _reports_dir(run_dir).mkdir(parents=True, exist_ok=True)

    prov = Provenance(
        git=os.environ.get("GIT_COMMIT", None),
        python=sys.version.split()[0],
        env=os.environ.get("CONDA_DEFAULT_ENV", None),
        seed=int(seed),
        cfg_short=cfg_short or {},
    )

    class_counts = y_series.value_counts().to_dict()
    geneset_block = _build_geneset_block(cvres=cvres, per_fold=per_fold)
    label_to_name = {int(k): str(v) for k, v in (class_name_map or {}).items()}
    class_counts_named = {
        str(label_to_name.get(int(label), label)): int(count)
        for label, count in class_counts.items()
    }


    doc = ReportDoc(
        run=RunInfo(path=str(run_dir), timestamp=_now_iso()),
        model={"name": model_name, "estimator": estimator_cls_name},
        data=DataSnapshot(
            x_shape=[int(X_shape[0]), int(X_shape[1])],
            y_len=int(len(y_series)),
            class_counts=class_counts_named,
            class_name_map={str(k): str(v) for k, v in (class_name_map or {}).items()} if class_name_map else None,
        ),
        provenance=prov,
        cv=CVSummary(metrics_mean=cv_mean, metrics_std=cv_std, per_fold=per_fold),
        tuning=TuningInfo(final_params_full_data=final_params_full_data or {}, cv_inner_hint=None),
        geneset=geneset_block,
        downloads=Downloads(model=str(Path(run_dir) / "train_artifacts" / f"{model_name}.joblib")),
    )

    p = report_json_path(run_dir, model_name)
    p.write_text(json.dumps(asdict(doc), indent=2, default=_json_default), encoding="utf-8")
    return p


def finalize_report_for_model(
    run_dir: Path,
    model_name: str,
    *,
    test_metrics_csv: Path,
    per_class_csv: Optional[Path],
    model_rel_path: str,
    cfg: Optional[Dict[str, Any]] = None,
    threshold_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Test stage writer: reads existing JSON, adds regular holdout metrics,
    optionally adds threshold-optimized holdout metrics, then renders one HTML.
    """
    run_dir = Path(run_dir)
    p = report_json_path(run_dir, model_name)
    data = json.loads(p.read_text(encoding="utf-8"))

    # regular holdout
    tm = pd.read_csv(test_metrics_csv).to_dict(orient="records")[0]

    per_class: List[Dict[str, Any]] = []
    if per_class_csv and Path(per_class_csv).exists():
        per_class = pd.read_csv(per_class_csv).to_dict(orient="records")

    data["test"] = {
        "metrics": tm,
        "per_class": per_class,
    }

    # optional threshold-optimized holdout
    if threshold_metrics:
        threshold_info = dict(threshold_metrics.get("thresholds") or {})
        threshold_per_class = _per_class_rows_from_report(threshold_metrics.get("report", {}))

        threshold_tm = {
            "n_test": threshold_metrics.get("n_test"),
            "accuracy": threshold_metrics.get("accuracy"),
            "f1_macro": threshold_metrics.get("f1_macro"),
            "auroc_ovr_macro": threshold_metrics.get("auroc_ovr_macro"),
            "logloss": threshold_metrics.get("logloss"),
        }

        data["threshold_optimized_holdout"] = {
            "metrics": threshold_tm,
            "per_class": threshold_per_class,
            "thresholds": threshold_info,
            "note": "Thresholds were learned from OOF training predictions and applied only to the holdout test set.",
        }

    # keep model path up to date
    data.setdefault("downloads", {})
    data["downloads"]["model"] = str(model_rel_path)

    p.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")

    hp = report_html_path(run_dir, model_name)
    html = _render_per_model_html(data, hp, run_dir, cfg=cfg)
    hp.write_text(html, encoding="utf-8")
    return hp


# -----------------------------
# HTML rendering (per-model only)
# -----------------------------

def _fmt(x: Any) -> str:
    if x is None or x == "":
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.4f}"
    return str(x)

def _fmt_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"

def _render_per_fold_table(per_fold: List[Dict[str, Any]]) -> str:
    if not per_fold:
        return "<div class='muted'>No per-fold metrics.</div>"

    cols = ["fold", "accuracy", "f1_macro"]
    head = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"
    rows = []
    for r in per_fold:
        cells = []
        for c in cols:
            v = r.get(c)
            if c == "fold":
                cells.append(f"<td>{esc(v)}</td>")
            else:
                cells.append(f"<td>{esc(_fmt(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table class='kv'>{head}{''.join(rows)}</table>"

def _render_per_class_table(per_class: List[Dict[str, Any]], class_name_map: Dict[str, str]) -> str:
    if not per_class:
        return "<div class='muted'>No per-class metrics.</div>"

    cols = ["class", "precision", "recall", "f1", "support"]
    head = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"

    rows = []
    for r in per_class:
        cls_raw = r.get("class")
        cls_key = str(cls_raw)
        cls_label = class_name_map.get(cls_key, str(cls_raw))
        rows.append(
            "<tr>"
            f"<td>{esc(cls_label)}</td>"
            f"<td>{esc(_fmt(r.get('precision')))}</td>"
            f"<td>{esc(_fmt(r.get('recall')))}</td>"
            f"<td>{esc(_fmt(r.get('f1')))}</td>"
            f"<td>{esc(_fmt(r.get('support')))}</td>"
            "</tr>"
        )
    return f"<table class='kv'>{head}{''.join(rows)}</table>"

def _per_class_rows_from_report(report_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cls, stats in (report_dict or {}).items():
        if cls in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(stats, dict):
            continue
        rows.append({
            "class": cls,
            "precision": float(stats.get("precision", np.nan)),
            "recall": float(stats.get("recall", np.nan)),
            "f1": float(stats.get("f1-score", np.nan)),
            "support": int(stats.get("support", 0)),
        })
    return rows

def _render_thresholds_table(threshold_info: Dict[str, Any], class_name_map: Dict[str, str]) -> str:
    thresholds = threshold_info.get("thresholds") or {}
    if not thresholds:
        return "<div class='muted'>No learned thresholds.</div>"

    rows = []
    for cls, value in thresholds.items():
        cls_label = class_name_map.get(str(cls), str(cls))
        rows.append(
            "<tr>"
            f"<td>{esc(cls_label)}</td>"
            f"<td>{esc(_fmt(value))}</td>"
            "</tr>"
        )
    return "<table class='kv'><tr><th>Class</th><th>Threshold</th></tr>" + "".join(rows) + "</table>"

def gene_chip_items(gene_ids: List[str], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    gene_ids = list(gene_ids or [])

    try:
        symbols = get_gene_symbols(gene_ids, cfg)
    except KeyError:
        symbols = gene_ids  # fallback if some IDs are missing

    return [
        {
            "id": str(gid),
            "symbol": str(sym) if sym else str(gid),
        }
        for gid, sym in zip(gene_ids, symbols)
    ]

def _render_geneset(gs: Optional[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None) -> str:
    if not gs or not gs.get("enabled"):
        return ""

    params = gs.get("params") or {}
    consensus = gs.get("consensus_genes") or []
    per_fold = gs.get("per_fold") or []
    top_svg = gs.get("top_freq_svg")

    parts: List[str] = []
    parts.append(
        card(
            "Gene-set optimization",
            kv_table({
                "Consensus size": gs.get("consensus_size", 0),
                **{f"param:{k}": v for k, v in params.items()},
            })
        )
    )

    if top_svg:
        parts.append(
            svg_panel(
                panel_id="geneset_top_freq",
                title="Top genes by fold-frequency",
                svg=top_svg,                 # inline SVG markup
                caption="Frequency across CV folds",
                open_by_default=False,
                collapsible=True,
                download_name="geneset_top_frequency.svg",
            )
        )

    if consensus:
        parts.append(chips_panel(
        panel_id="geneset_consensus",
        title="Consensus genes",
        items=gene_chip_items(consensus, cfg),
        open_by_default=False,
        searchable=True,
        copyable=True,
    ))

    if per_fold:
        fold_panels = []
        for row in per_fold:
            fold = row.get("fold")
            genes = row.get("genes") or []
            k = row.get("k")
            fold_panels.append(chips_panel(
                panel_id=f"geneset_fold_{fold}",
                title=f"Fold {fold} (k={k})",
                items=gene_chip_items(genes, cfg),
                open_by_default=False,
                searchable=True,
                copyable=True,
            ))

        parts.append(card("Per-fold selected genes", "".join(fold_panels)))

    return "".join(parts)

def _render_downloads(dl: Optional[Dict[str, Any]]) -> str:
    dl = dl or {}
    items = []
    for k, v in dl.items():
        if not v:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                items.append(f"<li><code>{esc(kk)}</code>: {esc(vv)}</li>")
        else:
            items.append(f"<li><code>{esc(k)}</code>: {esc(v)}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else "<div class='muted'>No artifacts recorded.</div>"

def _render_threshold_optimization_card(
    opt: Optional[Dict[str, Any]],
    class_name_map: Dict[str, str],
) -> str:
    if not opt:
        return ""

    metrics = opt.get("metrics") or {}
    threshold_info = opt.get("thresholds") or {}

    threshold_meta = {
        "Source": threshold_info.get("source", "OOF training predictions"),
        "Method": threshold_info.get("method"),
        "Optimized metric": threshold_info.get("optimized_metric"),
        "OOF baseline F1": threshold_info.get("baseline_oof_f1"),
        "OOF optimized F1": threshold_info.get("optimized_oof_f1"),
    }

    inner_html = (
        card(
            "Holdout Metrics",
            metric_grid([
                ("Test Macro-F1", _fmt(metrics.get("f1_macro"))),
                ("Test AUROC (macro)", _fmt(metrics.get("auroc_ovr_macro"))),
                ("Test Accuracy", _fmt(metrics.get("accuracy"))),
                ("Test LogLoss", _fmt(metrics.get("logloss"))),
            ], columns=4),
        )
        + card(
            "Learned Thresholds",
            kv_table(threshold_meta)
            + "<div style='height:10px;'></div>"
            + _render_thresholds_table(threshold_info, class_name_map),
        )
        + card(
            "Per-Class Holdout Metrics",
            _render_per_class_table(opt.get("per_class") or [], class_name_map),
        )
    )

    return panel(
        "Threshold Optimization",
        inner_html,
        open_by_default=True,
    )

def _render_per_model_html(d: Dict[str, Any], out_html: Path, site_dir: Path, cfg: Optional[Dict[str, Any]] = None) -> str:
    run = d.get("run") or {}
    model = d.get("model") or {}
    data = d.get("data") or {}
    prov = d.get("provenance") or {}
    cv = d.get("cv") or {}
    tuning = d.get("tuning") or {}
    test = d.get("test") or None
    threshold_opt = d.get("threshold_optimized_holdout") or None
    geneset = d.get("geneset") or None
    downloads = d.get("downloads") or None

    title = str(model.get("name") or "model").replace("_", " ").title()
    subtitle = str(model.get("estimator") or "")
    subtitle = clear_name(subtitle)

    header = card(
        title,
        f"<div class='muted'>{esc(subtitle)}</div>"
        f"<div class='muted'>Run: <code>{esc(run.get('path',''))}</code> • {esc(run.get('timestamp',''))}</div>",
    )

    data_card = card(
        "Data & Provenance",
        kv_table({
            "X shape": str(tuple(data.get("x_shape", []))),
            "y length": data.get("y_len"),
            "class counts": data.get("class_counts", {}),
        })
        + "<div style='height:10px;'></div>"
        + kv_table({k: v for k, v in prov.items() if k and k != "cfg_short"})
        + (panel("Full config snapshot", kv_table(prov.get("cfg_short") or {}), open_by_default=False) if prov.get("cfg_short") else "")
        # + (panel("Full config snapshot", json_pretty(prov.get("cfg_short") or {}), open_by_default=False) if prov.get("cfg_short") else "")
    )

    cv_mean = (cv.get("metrics_mean") or {})
    cv_std = (cv.get("metrics_std") or {})
    scorecards = card(
        "Scorecards",
        metric_grid([
            ("CV Macro-F1", _fmt_mean_std(cv_mean.get("f1_macro"), cv_std.get("f1_macro"))),
            ("CV AUROC (macro)", _fmt_mean_std(cv_mean.get("auroc_macro"), cv_std.get("auroc_macro"))),
            ("CV Accuracy", _fmt_mean_std(cv_mean.get("accuracy"), cv_std.get("accuracy"))),
            ("CV LogLoss", _fmt_mean_std(cv_mean.get("logloss"), cv_std.get("logloss"))),
        ], columns=4)
        + (
            "<div style='height:10px;'></div>"
            + metric_grid([
                ("Test Macro-F1", _fmt((test.get("metrics") or {}).get("f1_macro"))),
                ("Test AUROC (macro)", _fmt((test.get("metrics") or {}).get("auroc_ovr_macro"))),
                ("Test Accuracy", _fmt((test.get("metrics") or {}).get("accuracy"))),
                ("Test LogLoss", _fmt((test.get("metrics") or {}).get("logloss"))),
            ], columns=4)
            if test else ""
        )
    )

    train_card = card(
        "Train (Cross-Validation)",
        kv_table({k: _fmt_mean_std(cv_mean.get(k), cv_std.get(k)) for k in cv_mean.keys()})
        + panel("Per-fold metrics", _render_per_fold_table(cv.get("per_fold") or []), open_by_default=False)
        + _render_geneset(geneset, cfg=cfg)
    )

    class_name_map = data.get("class_name_map") or {}
    threshold_card = _render_threshold_optimization_card(
        threshold_opt,
        class_name_map,
    )
    test_card = card(
        "Test (Holdout)",
        _render_per_class_table((test or {}).get("per_class") or [], class_name_map) if test else "<div class='muted'>No holdout results.</div>"
    )

    tuning_card = card(
        "Hyperparameters (final, from full-data tune)",
        kv_table(tuning.get("final_params_full_data") or {})
    )

    downloads_card = card("Artifacts", _render_downloads(downloads))

    body_html = header + data_card + scorecards + train_card + test_card  + tuning_card + threshold_card + downloads_card

    out_html = site_dir / "diagnostic_plots.html"
    base_dir = site_dir.parent
    tree, _ = build_sidebar_from_run_dir(base_dir)
    current_page = Path(out_html).relative_to(base_dir).as_posix() 
    sidebar_html = render_sidebar_html(tree=tree, current_page=current_page)

    return wrap_report_html(
        title=f"{title} — Report",
        subtitle=subtitle,
        body_html=body_html,
        sidebar_html=sidebar_html,
    )