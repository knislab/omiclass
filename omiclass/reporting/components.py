# omiclass/reporting/components.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import html
import uuid
import json
import math

# --- key prettyfier ---
_KEY_LABEL_OVERRIDES = {
    "f1_macro": "F1 (macro)",
    "auroc_macro": "AUROC (macro)",
    "auprc_macro": "AUPRC (macro)",
    "logloss": "LogLoss",
    "brier": "Brier",
    "cv_folds": "CV folds",
    "tune_inner_cv": "Inner CV",
    "bayes_iter": "Bayes iterations",
    "auroc_ovr_macro": "AUROC (OvR macro)",
}
def _pretty_key(k: Any) -> str:
    s = str(k)
    if s in _KEY_LABEL_OVERRIDES:
        return _KEY_LABEL_OVERRIDES[s]
    s2 = s.replace("_", " ").replace("-", " ").strip()
    # keep common acronyms readable
    for token in ["AUROC", "AUPRC", "FDR", "PCA", "PC", "TPM", "RNA", "DNA", "CV"]:
        s2 = s2.replace(token.title(), token)
    # Title-case but don't destroy ALLCAPS tokens we injected above
    return s2[:1].upper() + s2[1:]

def _fmt_scalar(v: Any, empty_value: str) -> str:
    if v is None or v == "":
        return empty_value
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, (float,)):
        if math.isnan(v):
            return empty_value
        return f"{v:.4f}"
    return str(v)

def esc(x: Any) -> str:
    """HTML-escape anything safely."""
    return html.escape("" if x is None else str(x))


def card(title: str, body_html: str, *, subtitle: str = "", id: Optional[str] = None) -> str:
    sid = f' id="{esc(id)}"' if id else ""
    sub = f'<div class="subtle">{esc(subtitle)}</div>' if subtitle else ""
    return f"""
<section class="card"{sid}>
  <h2>{esc(title)}</h2>
  {sub}
  {body_html}
</section>
""".strip()


def kv_table(
    d: Dict[str, Any],
    *,
    empty_value: str = "—",
    order: Optional[Sequence[str]] = None,
    class_name: str = "kv",max_depth: int = 2,
    max_list_items: int = 30,
) -> str:
    """
    Recursive kv table:
    - dict -> nested table (until max_depth)
    - list/tuple/set -> short list preview (until max_list_items)
    - scalars -> formatted
    """
    if not d:
        return f"<div class='muted'>{html.escape(empty_value)}</div>"

    keys = list(order) if order else list(d.keys())
    rows: List[str] = []

    def render_value(v: Any, depth: int) -> str:
        # dict -> nested table (depth-limited)
        if isinstance(v, dict):
            if depth >= max_depth:
                txt = json.dumps(v, indent=2, ensure_ascii=False, default=str)
                return f"<pre class='code' style='margin:0;'><code>{html.escape(txt)}</code></pre>"
            return kv_table(
                v,
                empty_value=empty_value,
                class_name=class_name,
                max_depth=max_depth,
                max_list_items=max_list_items,
            )

        # lists -> bullets / preview
        if isinstance(v, (list, tuple, set)):
            seq = list(v)
            if len(seq) == 0:
                return f"<span class='muted'>{html.escape(empty_value)}</span>"
            if all(not isinstance(x, (dict, list, tuple, set)) for x in seq):
                shown = seq[:max_list_items]
                more = len(seq) - len(shown)
                items = "".join(f"<li>{html.escape(_fmt_scalar(x, empty_value))}</li>" for x in shown)
                tail = f"<li class='muted'>… +{more} more</li>" if more > 0 else ""
                return f"<ul style='margin:0; padding-left:18px;'>{items}{tail}</ul>"
            # complex list -> fallback JSON
            txt = json.dumps(seq[:max_list_items], indent=2, ensure_ascii=False, default=str)
            suffix = "\n… (truncated)" if len(seq) > max_list_items else ""
            return f"<pre class='code' style='margin:0;'><code>{html.escape(txt + suffix)}</code></pre>"

        # scalar
        return html.escape(_fmt_scalar(v, empty_value))

    for k in keys:
        if k not in d:
            continue
        v = d[k]
        k_label = _pretty_key(k)
        cell = render_value(v, depth=0)
        rows.append(f"<tr><th>{html.escape(k_label)}</th><td>{cell}</td></tr>")

    return f"<table class='{html.escape(class_name)}'>{''.join(rows)}</table>"

def metric_grid(items: Sequence[Tuple[str, Any]], *, columns: int = 4) -> str:
    """
    Render metrics using shared .metric styles inside .grid grid-N.
    items: list of (label, value)
    """
    if not items:
        return "<div class='muted'>—</div>"

    columns = int(columns)
    if columns < 1:
        columns = 1
    if columns > 4:
        columns = 4

    grid_cls = f"grid grid-{columns}"

    blocks: List[str] = []
    for k, v in items:
        blocks.append(
            f"<div class='metric'>"
            f"<div class='metric-k'>{esc(k)}</div>"
            f"<div class='metric-v'>{esc(v)}</div>"
            f"</div>"
        )
    return f"<div class='{grid_cls}'>" + "".join(blocks) + "</div>"


def vertical_metric_grid(items: Sequence[Tuple[str, Any]], *, gap: int = 10) -> str:
    """
    Render metrics as a vertical stack (no outer card; meant to be embedded inside a card).
    """
    if not items:
        return "<div class='muted'>—</div>"

    blocks: List[str] = []
    for k, v in items:
        blocks.append(
            f"<div class='metric'>"
            f"<div class='metric-k'>{esc(k)}</div>"
            f"<div class='metric-v'>{esc(v)}</div>"
            f"</div>"
        )
    return f"<div class='metrics-vertical' style='gap:{int(gap)}px;'>" + "".join(blocks) + "</div>"


def panel(
    summary: str,
    body_html: str,
    *,
    open_by_default: bool = False,
    badge: Optional[str] = None,
    id: Optional[str] = None,
) -> str:
    """
    Collapsible <details> block using shared .panel styles.
    """
    open_attr = " open" if open_by_default else ""
    sid = f' id="{esc(id)}"' if id else ""
    badge_html = f"<span class='badge'>{esc(badge)}</span>" if badge is not None else ""
    return f"""
<details class="panel"{open_attr}{sid}>
  <summary><span class="summary-title">{esc(summary)}</span>{badge_html}</summary>
  <div class="panel-body">{body_html}</div>
</details>
""".strip()


def chips_grid(items: Sequence[Any], *, code: bool = True, empty_value: str = "None") -> str:
    """
    Chip grid (no search/copy).
    """
    if not items:
        return f"<div class='muted'>{esc(empty_value)}</div>"

    chips: List[str] = []
    for x in items:
        content = f"<code>{esc(x)}</code>" if code else esc(x)
        chips.append(f"<span class='chip'>{content}</span>")
    return "<div class='chips'>" + "".join(chips) + "</div>"

def svg_panel(
    *,
    panel_id: Optional[str] = None,
    title: str,
    svg: str,
    caption: str = "",
    open_by_default: bool = False,
    collapsible: bool = True,
    download_name: Optional[str] = None,
) -> str:
    """
    Inline SVG panel that uses the global zoom/pan behavior from OMICLASS_CSS/JS.

    Contract:
      - wrapper .svg-frame
      - inner .svg-viewport has id=<fig_id> and data-scale="1"
      - toolbar buttons have data-zoom-*(fig_id)
    """
    fig_id = panel_id or f"fig_{uuid.uuid4().hex[:10]}"

    cap = f"<div class='muted small' style='margin:6px 0 0 0;'>{esc(caption)}</div>" if caption else ""

    wrapped_svg = f"""
<div class="svg-frame" data-figure="{esc(fig_id)}">
  <div class="svg-viewport" id="{esc(fig_id)}" data-scale="1">
    {svg}
  </div>
</div>
""".strip()

    dl = ""
    if download_name:
        png_name = (
            download_name[:-4] + ".png"
            if download_name.lower().endswith(".svg")
            else download_name + ".png"
        )
        dl = f"""
    <button class="btn btn-sm"
            type="button"
            data-download-svg="{esc(fig_id)}"
            data-download-name="{esc(download_name)}">
    Download SVG
    </button>
    <button class="btn btn-sm"
            type="button"
            data-download-png="{esc(fig_id)}"
            data-download-name="{esc(png_name)}">
    Download PNG
    </button>
    """.strip()
        
    toolbar = f"""
<div class="figure-toolbar">
  <button class="btn btn-sm" type="button" data-zoom-out="{esc(fig_id)}">−</button>
  <button class="btn btn-sm" type="button" data-zoom-in="{esc(fig_id)}">+</button>
  <button class="btn btn-sm" type="button" data-zoom-reset="{esc(fig_id)}">Reset</button>
  {dl}
  <span class="muted small">Wheel to zoom, drag to pan</span>
</div>
""".strip()

    body_html = toolbar + wrapped_svg + cap

    if not collapsible:
        return card(title, body_html)

    return panel(title, body_html, open_by_default=open_by_default, badge=None, id=None)

def chips_panel(
    *,
    panel_id: str,
    title: str,
    items: Sequence[Any],
    open_by_default: bool = False,
    searchable: bool = True,
    copyable: bool = True,
    empty_value: str = "None",
    code: bool = True,
) -> str:
    """
    Collapsible chip panel with optional search + copy.

    Supports:
      - plain items: ["ENSG...", ...]
      - gene items: [{"id": "ENSG...", "symbol": "TP53"}, ...]
    """
    items = list(items or [])
    n = len(items)

    if n == 0:
        return panel(title, f"<div class='muted'>{esc(empty_value)}</div>", open_by_default=open_by_default, badge="0")

    has_gene_symbols = any(isinstance(x, dict) and "id" in x and "symbol" in x for x in items)

    tools: List[str] = []
    if searchable:
        tools.append(f'<input class="input" type="search" placeholder="Search…" data-target="{esc(panel_id)}" />')

    if has_gene_symbols:
        tools.append(
            f'<button class="btn" type="button" data-chip-toggle="{esc(panel_id)}" data-mode="id">'
            f'Show symbols</button>'
        )

    if copyable:
        tools.append(f'<button class="btn" data-copy="{esc(panel_id)}">Copy list</button>')

    tools.append(f'<span class="muted small">{n} items</span>')
    toolbar_html = f'<div class="toolbar">{"".join(tools)}</div>'

    copy_lines: List[str] = []
    chips: List[str] = []

    for x in items:
        if isinstance(x, dict):
            gene_id = str(x.get("id", ""))
            symbol = str(x.get("symbol", ""))

            visible = gene_id
            search_text = f"{gene_id} {symbol}".strip()
            copy_lines.append(f"{gene_id}\t{symbol}".rstrip())

            content = f"<code>{esc(visible)}</code>" if code else esc(visible)

            chips.append(
                f"<span class='chip gene-chip' "
                f"data-id='{esc(gene_id)}' "
                f"data-symbol='{esc(symbol)}' "
                f"data-search='{esc(search_text)}'>"
                f"{content}</span>"
            )
        else:
            value = str(x)
            copy_lines.append(value)
            content = f"<code>{esc(value)}</code>" if code else esc(value)
            chips.append(
                f"<span class='chip' data-search='{esc(value)}'>{content}</span>"
            )

    joined = "\n".join(copy_lines)
    data_html = f'<textarea class="hidden" id="{esc(panel_id)}__data">{esc(joined)}</textarea>'

    chips_html = f"<div class='chips' id='{esc(panel_id)}'>" + "".join(chips) + "</div>"

    return panel(
        title,
        toolbar_html + data_html + chips_html,
        open_by_default=open_by_default,
        badge=str(n),
    )


def json_pretty(obj: Any, *, max_chars: int = 250_000) -> str:
    """
    Pretty JSON in a code block (fallback). Keep it for debugging sections.
    """
    try:
        s = json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        s = str(obj)

    if len(s) > max_chars:
        s = s[:max_chars] + "\n… (truncated)"
    return f"<pre><code>{esc(s)}</code></pre>"


def json_kv(obj: Any, *, empty_value: str = "—") -> str:
    """
    If obj is a dict, show as kv_table; otherwise fallback to json_pretty.
    """
    if obj is None:
        return f"<div class='muted'>{esc(empty_value)}</div>"
    if isinstance(obj, dict):
        return kv_table({str(k): v for k, v in obj.items()}, empty_value=empty_value)
    return json_pretty(obj)

def lazy_svg_panel(
    *,
    panel_id: str,
    title: str,
    svg_path: str,
    open_by_default: bool = False,
    caption: str = "",
) -> str:
    """
    Lazy-load an external SVG path into a slot when <details> opens.
    JS contract:
      - details[data-lazy-svg-src] will fetch the SVG and inject into .lazy-slot
    """
    cap = f"<div class='muted small'>{esc(caption)}</div>" if caption else ""
    body = f"""
{cap}
<div class="lazy-slot" id="{esc(panel_id)}__slot" data-state="empty"></div>
<div class="muted small" style="margin-top:8px;">
  <a href="{esc(svg_path)}" download>Download SVG</a>
</div>
""".strip()

    return f"""
<details class="panel" {'open' if open_by_default else ''} data-lazy-svg-src="{esc(svg_path)}" id="{esc(panel_id)}">
  <summary><span class="summary-title">{esc(title)}</span></summary>
  <div class="panel-body">{body}</div>
</details>
""".strip()


def vcard(title: str, body_html: str, *, subtitle: str = "", id: Optional[str] = None) -> str:
    """Same as card(), but forces vertical layout via an extra CSS class."""
    sid = f' id="{esc(id)}"' if id else ""
    sub = f'<div class="subtle">{esc(subtitle)}</div>' if subtitle else ""
    return f"""
<section class="card card-vertical"{sid}>
  <h2>{esc(title)}</h2>
  {sub}
  {body_html}
</section>
""".strip()

def split_card_3_1(
    *,
    title: str,
    subtitle: str = "",
    left_title: str = "",
    right_title: str = "",
    left_html: str,
    right_html: str,
) -> str:
    sub = f'<div class="subtle" style="margin-bottom:10px;">{esc(subtitle)}</div>' if subtitle else ""
    lt = f'<div class="muted small" style="margin-bottom:6px;">{esc(left_title)}</div>' if left_title else ""
    rt = f'<div class="muted small" style="margin-bottom:6px;">{esc(right_title)}</div>' if right_title else ""
    body = f"""
{sub}
<div class="split-3-1">
  <div>
    {lt}
    {left_html}
  </div>
  <div>
    {rt}
    {right_html}
  </div>
</div>
""".strip()
    return card(title, body)


def split_card_1_1(
    *,
    title: str,
    subtitle: str = "",
    left_title: str = "",
    right_title: str = "",
    left_html: str,
    right_html: str,
) -> str:
    sub = f'<div class="subtle" style="margin-bottom:10px;">{esc(subtitle)}</div>' if subtitle else ""
    lt = f'<div class="muted small" style="margin-bottom:6px;">{esc(left_title)}</div>' if left_title else ""
    rt = f'<div class="muted small" style="margin-bottom:6px;">{esc(right_title)}</div>' if right_title else ""
    body = f"""
{sub}
<div class="grid grid-2">
  <div>
    {lt}
    {left_html}
  </div>
  <div>
    {rt}
    {right_html}
  </div>
</div>
""".strip()
    return card(title, body)

def nav_buttons(items: Sequence[Tuple[str, str, bool]]) -> str:
    # items: (label, href, is_active)
    btns = []
    for label, href, active in items:
        style = "font-weight:700;" if active else ""
        btns.append(f'<a class="btn" href="{esc(href)}" style="{style}">{esc(label)}</a>')
    return f'<div class="toolbar" style="margin-top:10px;">{"".join(btns)}</div>'
