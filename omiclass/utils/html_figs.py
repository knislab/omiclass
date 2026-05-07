# omiclass/utils/html_figs.py
from __future__ import annotations
import base64
import html as _html
from typing import Optional
from pathlib import Path
import re
import uuid
import html

_SVG_META_RE = re.compile(r"<metadata\b.*?</metadata>", re.IGNORECASE | re.DOTALL)
_SVG_DESC_RE = re.compile(r"<desc\b.*?</desc>", re.IGNORECASE | re.DOTALL)
_SVG_COMM_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SVG_WS_RE = re.compile(r">\s+<")
_SVG_FLOAT_RE = re.compile(r"(?<![\w-])(-?\d+\.\d+)(?![\w-])")

def _compact_svg(svg: str, *, max_decimals: int = 3) -> str:
    """Reduce SVG text size without changing the format (still SVG)."""
    if not svg:
        return svg
    s = svg.strip()
    s = _SVG_META_RE.sub("", s)
    s = _SVG_DESC_RE.sub("", s)
    s = _SVG_COMM_RE.sub("", s)
    s = _SVG_WS_RE.sub("><", s)

    def _trim(m: re.Match) -> str:
        val = m.group(1)
        try:
            f = float(val)
        except Exception:
            return val
        out = f"{f:.{max_decimals}f}".rstrip("0").rstrip(".")
        return out

    s = _SVG_FLOAT_RE.sub(_trim, s)
    return s

def _load_svg(svg_or_path: str) -> str:
    """Accepts SVG markup or path to SVG file; returns SVG markup string."""
    if not svg_or_path:
        return ""
    if "<svg" in svg_or_path[:300]:
        return svg_or_path
    p = Path(svg_or_path)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    return svg_or_path

def _lazy_inline_svg(*, label: str, src: Optional[str], open_by_default: bool = False) -> str:
    """
    Inline SVG but do NOT parse it until expanded.
    Keeps everything self-contained inside HTML (no saving files).
    """
    if not src:
        return ""
    svg_markup = _load_svg(src)
    if not svg_markup:
        return ""

    # Reduce size
    svg_markup = _compact_svg(svg_markup, max_decimals=3)

    safe_label = html.escape(label)
    uid = f"lazy_{uuid.uuid4().hex}"

    if open_by_default:
        # Render immediately (parsed)
        return f"""
<figure class="lazy-fig" id="{uid}">
  <figcaption>{safe_label}</figcaption>
  {svg_markup}
</figure>
""".strip()

    # Store as text; parse only on <details> open
    svg_text = html.escape(svg_markup)
    return f"""
<figure class="lazy-fig" id="{uid}">
  <details>
    <summary>{safe_label}</summary>
    <div class="lazy-slot" data-state="empty" style="margin-top:10px;">
      <div class="muted">Click to load SVG…</div>
    </div>
    <script type="text/plain" class="lazy-svg">{svg_text}</script>
  </details>
</figure>
""".strip()

def _inject_svg_style(svg: str, *, max_height_px: Optional[int]) -> str:
    """
    Make inline SVG responsive:
      - width: 100%, height: auto
      - optional max-height with contained scaling
    """
    style_bits = [
        "display:block",
        "width:100%",
        "height:auto",
    ]
    if max_height_px is not None:
        style_bits += [
            f"max-height:{int(max_height_px)}px",
            "object-fit:contain",
        ]
    style_add = ";".join(style_bits)

    # If svg already has style="", append; otherwise add style=""
    if re.search(r"<svg[^>]*\sstyle=", svg, flags=re.IGNORECASE):
        return re.sub(
            r"(<svg[^>]*\sstyle=\")",
            r"\1" + style_add + ";",
            svg,
            count=1,
            flags=re.IGNORECASE,
        )
    return re.sub(
        r"<svg\b",
        f"<svg style=\"{style_add}\"",
        svg,
        count=1,
        flags=re.IGNORECASE,
    )

def svg_str_to_data_uri(svg: str) -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"

def render_svg_figure(
    *,
    svg: str,
    caption: str,
    download_name: Optional[str] = None,
    collapsible: bool = True,
    open_by_default: bool = False,
    max_height_px: Optional[int] = None,
) -> str:
    cap = _html.escape(caption)

    fig_id = f"fig_{uuid.uuid4().hex[:10]}"

    dl = ""
    if download_name:
        uri = svg_str_to_data_uri(svg)
        dl = (
            "<div style='margin-top:6px;'>"
            f"<a download='{_html.escape(download_name)}' href='{uri}'>Download SVG</a>"
            "</div>"
        )

    svg = _inject_svg_style(svg, max_height_px=max_height_px)

    # zoom toolbar + scrollable frame + viewport (for transform)
    tool = f"""
<div class="figure-toolbar">
  <button class="btn btn-sm" type="button" data-zoom-out="{fig_id}">−</button>
  <button class="btn btn-sm" type="button" data-zoom-in="{fig_id}">+</button>
  <button class="btn btn-sm" type="button" data-zoom-reset="{fig_id}">Reset</button>
  <span class="muted small">Wheel to zoom, drag to pan</span>
</div>
""".strip()

    wrapped_svg = f"""
<div class="svg-frame" data-figure="{fig_id}">
  <div class="svg-viewport" id="{fig_id}" data-scale="1">
    {svg}
  </div>
</div>
""".strip()

    fig = (
        "<figure style='margin:16px 0;'>"
        f"{tool}"
        f"{wrapped_svg}"
        f"<figcaption style='color:#666;font-size:0.9rem;margin-top:6px;'>{cap}</figcaption>"
        f"{dl}"
        "</figure>"
    )

    if not collapsible:
        return fig

    open_attr = " open" if open_by_default else ""
    return (
        f"<details{open_attr}>"
        f"<summary style='cursor:pointer;font-weight:600;margin:6px 0;'>{cap}</summary>"
        f"{fig}</details>"
    )

def maybe_svg(
    *,
    label: str,
    svg: Optional[str],
    download_name: Optional[str] = None,
    collapsible: bool = True,
    open_by_default: bool = False,
    max_height_px: Optional[int] = None,   # <<< add this
) -> str:
    if not svg:
        return ""
    return render_svg_figure(
        svg=svg,
        caption=label,
        download_name=download_name,
        collapsible=collapsible,
        open_by_default=open_by_default,
        max_height_px=max_height_px,
    )

def embed_svg_file_then_delete(svg_path: Path, caption: str) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    html_block = render_svg_figure(svg=svg, caption=caption, download_name=svg_path.name, collapsible=True)
    svg_path.unlink(missing_ok=True)
    return html_block