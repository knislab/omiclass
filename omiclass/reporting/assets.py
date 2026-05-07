# omiclass.reporting.assets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import html


# =========================
# Public API
# =========================

@dataclass(frozen=True)
class ReportAssets:
    css_text: str
    js_text: str


def get_inline_assets() -> ReportAssets:
    """Return inline CSS/JS for single-file portable reports."""
    return ReportAssets(css_text=OMICLASS_CSS, js_text=OMICLASS_JS)


def wrap_report_html(
    *,
    title: str,
    body_html: str,
    subtitle: str = "",
    sidebar_html: str = "", 
    extra_head_html: str = "",
    assets: Optional[ReportAssets] = None,
) -> str:
    """
    Standard report shell (INLINE MODE ONLY).

    body_html must contain ONLY inner content (cards/sections),
    not a full HTML document.
    """
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle) if subtitle else ""

    assets = assets or get_inline_assets()
    subtitle_html = f'<div class="subtle">{safe_subtitle}</div>' if subtitle else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>

  <style>
{assets.css_text}
  </style>

  {extra_head_html}

  <script>
{assets.js_text}
  </script>
</head>
<body>
  <div class="container">
    <div class="app-shell">

      <aside class="sidebar">
        {sidebar_html}
      </aside>
      <main class="main">
        <header class="page-header">
          <div class="page-title">
            <h1>{safe_title}</h1>
            {subtitle_html}
          </div>
        </header>

        {body_html}
      </main>

    </div>
  </div>
</body>
</html>
"""


# =========================
# Shared CSS
# =========================

OMICLASS_CSS = r"""
:root{
  --bg:#ffffff;
  --fg:#0f172a;
  --muted:#64748b;
  --border:#e5e7eb;
  --card:#ffffff;
  --chip:#f8fafc;
  --shadow:0 1px 2px rgba(0,0,0,.05);
  --radius:14px;
  --radius-sm:10px;
  --maxw:1150px;
}

*{ box-sizing:border-box; }
html,body{ height:100%; }

body{
  margin:0;
  background:var(--bg);
  color:var(--fg);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  line-height:1.45;
}

.container{
  max-width:var(--maxw);
  margin:0 auto;
  padding:28px 22px 60px;
}

/* ---- App shell (sidebar + content) ---- */
.app-shell{
  display:grid;
  grid-template-columns: 290px 1fr;
  gap: 14px;
  align-items:start;
}

.sidebar{
  position: sticky;
  top: 16px;
  align-self: start;
}

.main{
  min-width: 0; /* prevents overflow issues in grids */
}

@media (max-width: 900px){
  .app-shell{ grid-template-columns: 1fr; }
  .sidebar{ position: static; }
}

.page-header{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:16px;
  margin-bottom:14px;
}

h1{
  font-size:34px;
  letter-spacing:-0.02em;
  margin:0 0 6px 0;
}

h2{
  font-size:18px;
  margin:0 0 10px 0;
}

.subtle{ color:var(--muted); font-size:13px; }
.small{ font-size:12px; }
.muted{ color:var(--muted); }

.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  box-shadow:var(--shadow);
  padding:16px;
  margin-top:14px;
}

.grid{
  display:grid;
  gap:12px;
}

.grid-2{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
.grid-3{ grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-4{ grid-template-columns: repeat(4, minmax(0, 1fr)); }

@media (max-width:900px){
  .grid-4{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .grid-3{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width:560px){
  .grid-2, .grid-3, .grid-4{ grid-template-columns: 1fr; }
}

.split-3-1{
  display: grid;
  grid-template-columns: 3fr 1fr;
  gap: 12px;
  align-items: start;
}

@media (max-width: 900px){
  .split-3-1{
    grid-template-columns: 1fr;
  }
}

/* Vertical metrics stack */
.metrics-vertical{
  display: flex;
  flex-direction: column;
}

.metric{
  border:1px solid var(--border);
  border-radius:12px;
  padding:10px 12px;
  background:#fff;
}

.metric-k{
  color:var(--muted);
  font-size:12px;
}

.metric-v{
  font-size:20px;
  font-weight:700;
  margin-top:2px;
}

.kv{
  width:100%;
  border-collapse:collapse;
}

.kv th{
  text-align:left;
  color:var(--muted);
  font-weight:600;
  padding:10px 12px;
  width:230px;
  border-bottom:1px solid var(--border);
  background:#fafafa;
}

.kv td{
  padding:10px 12px;
  border-bottom:1px solid var(--border);
}

.kv tr:last-child th,
.kv tr:last-child td{
  border-bottom:0;
}

.btn{
  padding:8px 10px;
  border-radius:10px;
  border:1px solid var(--border);
  background:#fff;
  cursor:pointer;
  font-weight:600;
}

.btn:hover{ background:#fafafa; }

.input{
  padding:8px 10px;
  border-radius:10px;
  border:1px solid var(--border);
  min-width:220px;
}

.panel{
  border:1px solid var(--border);
  border-radius:12px;
  background:#fff;
  margin:10px 0;
  overflow:hidden;
}

.panel summary{
  list-style:none;
  cursor:pointer;
  padding:10px 12px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  font-weight:700;
}

.panel summary::-webkit-details-marker{ display:none; }

.panel summary::after {
  content: "▸";
  transition: transform 0.2s ease;
}

.panel[open] summary::after {
  transform: rotate(90deg);
}

.panel .panel-body{
  padding:12px;
  border-top:1px solid var(--border);
}

.badge{
  font-size:12px;
  font-weight:700;
  color:var(--muted);
  border:1px solid var(--border);
  padding:2px 8px;
  border-radius:999px;
}

.toolbar{
  display:flex;
  gap:10px;
  align-items:center;
  flex-wrap:wrap;
  margin-bottom:10px;
}

.chips{
  display:grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap:8px;
}

.chip{
  background:var(--chip);
  border:1px solid var(--border);
  border-radius:10px;
  padding:8px 10px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}

code{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size:12.5px;
}

.hidden{
  position:absolute;
  left:-9999px;
  top:-9999px;
}
/* ---- Sidebar navigation ---- */
.sidebar-nav { display:flex; flex-direction:column; gap:10px; }

.nav-group { border:1px solid var(--border); border-radius:12px; background:var(--card); padding:8px; }

.nav-group-title { cursor:pointer; font-weight:600; padding:6px 8px; }

.nav-group-items { display:flex; flex-direction:column; gap:2px; padding:4px 2px 6px; }

.nav-item { display:block; padding:7px 10px; border-radius:10px; text-decoration:none; color:var(--fg); }

.nav-item:hover { background: rgba(0,0,0,0.04); }

.nav-item.active { background: rgba(0,0,0,0.08);}

.nav-item.disabled { opacity:0.6; padding:7px 10px; }

/* ---- SVG figure UX: frame + zoom/pan ---- */
.figure-toolbar{
  display:flex;
  gap:8px;
  align-items:center;
  flex-wrap:wrap;
  margin: 0 0 8px 0;
}

.btn-sm{
  padding:6px 8px;
  border-radius:10px;
  font-weight:700;
  line-height:1;
}

/* keeps SVG inside the card and enables scrolling when needed */
.svg-frame{
  width: 100%;
  max-width: 100%;
  max-height: min(70vh, 900px);
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #fff;
  padding: 8px;
  cursor: grab;
}

.svg-frame:active{
  cursor: grabbing;
}

/* this is what we scale */
.svg-viewport{
  display: block;
}

/* make sure inline svgs behave */
.svg-frame svg{
  display:block;
  max-width: none;   /* important: allow scroll when scaled */
  width: auto;
  height: auto;
}

"""


# =========================
# Shared JS 
# =========================

OMICLASS_JS = r"""
(function(){
  function norm(s){
    return (s || "").toLowerCase();
  }

  /* -----------------------------
   * Chip search filtering
   * ----------------------------- */
  document.addEventListener("input", (ev) => {
    const el = ev.target;
    if(!(el instanceof HTMLInputElement)) return;
    if(!el.matches("input[data-target]")) return;

    const targetId = el.getAttribute("data-target");
    const q = norm(el.value);
    const grid = targetId ? document.getElementById(targetId) : null;
    if(!grid) return;

    grid.querySelectorAll(".chip").forEach(ch => {
      const txt = norm(ch.getAttribute("data-search") || ch.textContent);
      ch.style.display = txt.includes(q) ? "" : "none";
    });
  });
  document.addEventListener("click", (ev) => {
  const btn = ev.target instanceof HTMLElement
    ? ev.target.closest("button[data-chip-toggle]")
    : null;

  if (!btn) return;

  const panelId = btn.getAttribute("data-chip-toggle");
  const panel = panelId ? document.getElementById(panelId) : null;
  if (!panel) return;

  const currentMode = btn.getAttribute("data-mode") || "id";
  const nextMode = currentMode === "id" ? "symbol" : "id";

  btn.setAttribute("data-mode", nextMode);
  btn.textContent = nextMode === "symbol" ? "Show IDs" : "Show symbols";

  panel.querySelectorAll(".gene-chip").forEach(chip => {
    const value = chip.getAttribute(`data-${nextMode}`) || chip.getAttribute("data-id") || "";
    chip.innerHTML = "";

    const code = document.createElement("code");
    code.textContent = value;
    chip.appendChild(code);
  });
});

  /* -----------------------------
  * SVG download
  * ----------------------------- */
  document.addEventListener("click", (ev) => {
    const btn = ev.target instanceof HTMLElement
      ? ev.target.closest("button[data-download-svg]")
      : null;
    if (!btn) return;

    const figId = btn.getAttribute("data-download-svg");
    const filename = btn.getAttribute("data-download-name") || "figure.svg";
    if (!figId) return;

    const vp = document.getElementById(figId);
    if (!vp) return;

    const svg = vp.querySelector("svg");
    if (!svg) return;

    // clone to avoid mutating live DOM
    const clone = svg.cloneNode(true);

    // ensure standalone SVG validity
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

    const text = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([text], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();

    setTimeout(() => URL.revokeObjectURL(url), 2000);
  });

  /* -----------------------------
   * Copy-to-clipboard buttons
   * ----------------------------- */
  document.addEventListener("click", async (ev) => {
    const btn = ev.target instanceof HTMLElement
      ? ev.target.closest("button[data-copy]")
      : null;
    if(!btn) return;

    const id = btn.getAttribute("data-copy");
    if(!id) return;

    const ta = document.getElementById(id + "__data");
    if(!(ta instanceof HTMLTextAreaElement)) return;

    try{
      await navigator.clipboard.writeText(ta.value);
    }catch(e){
      ta.focus();
      ta.select();
      document.execCommand("copy");
    }

    const old = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => {
      btn.textContent = old || "Copy";
    }, 900);
  });

    /* -----------------------------
    * SVG zoom/pan helpers
    * ----------------------------- */
    function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }

    function parseLen(v){
      if(v == null) return NaN;
      const n = parseFloat(String(v));
      return isFinite(n) ? n : NaN;
    }

    function getViewport(id){
      const vp = document.getElementById(id);
      return (vp instanceof HTMLElement) ? vp : null;
    }

    function getScale(vp){
      const s = parseFloat(vp.dataset.scale || "1");
      return isFinite(s) ? s : 1;
    }

    function getNaturalSize(vp){
      const svg = vp.querySelector("svg");
      if(!(svg instanceof SVGElement)) return null;

      const cachedW = parseFloat(vp.dataset.naturalW || "");
      const cachedH = parseFloat(vp.dataset.naturalH || "");
      if(isFinite(cachedW) && isFinite(cachedH) && cachedW > 0 && cachedH > 0){
        return { svg, w: cachedW, h: cachedH };
      }

      let w = NaN;
      let h = NaN;

      const vb = svg.viewBox && svg.viewBox.baseVal;
      if(vb && vb.width > 0 && vb.height > 0){
        w = vb.width;
        h = vb.height;
      }

      if(!(w > 0 && h > 0)){
        const aw = parseLen(svg.getAttribute("width"));
        const ah = parseLen(svg.getAttribute("height"));
        if(aw > 0 && ah > 0){
          w = aw;
          h = ah;
        }
      }

      if(!(w > 0 && h > 0)){
        const rect = svg.getBoundingClientRect();
        if(rect.width > 0 && rect.height > 0){
          w = rect.width;
          h = rect.height;
        }
      }

      if(!(w > 0 && h > 0)) return null;

      vp.dataset.naturalW = String(w);
      vp.dataset.naturalH = String(h);
      return { svg, w, h };
    }

    function setScale(vp, s){
      const natural = getNaturalSize(vp);
      if(!natural) return;

      vp.dataset.scale = String(s);

      const sw = natural.w * s;
      const sh = natural.h * s;

      vp.style.width = `${sw}px`;
      vp.style.height = `${sh}px`;
      natural.svg.style.width = `${sw}px`;
      natural.svg.style.height = `${sh}px`;
    }

    function fitAndCenter(vp){
      const frame = vp.closest(".svg-frame");
      if(!frame) return false;

      const natural = getNaturalSize(vp);
      if(!natural) return false;

      // Reset transform before measuring natural size
      vp.dataset.scale = "1";
      vp.style.width = `${natural.w}px`;
      vp.style.height = `${natural.h}px`;
      natural.svg.style.width = `${natural.w}px`;
      natural.svg.style.height = `${natural.h}px`;

      const frameStyle = getComputedStyle(frame);
      const padX = (parseFloat(frameStyle.paddingLeft) || 0) + (parseFloat(frameStyle.paddingRight) || 0);
      const padY = (parseFloat(frameStyle.paddingTop) || 0) + (parseFloat(frameStyle.paddingBottom) || 0);

      const naturalW = natural.w;
      const naturalH = natural.h;
      if(!naturalW || !naturalH) return false;

      const availW = Math.max(1, frame.clientWidth - padX);
      const availH = Math.max(1, Math.min(frame.clientHeight, window.innerHeight * 0.7, 900) - padY);

      const scale = clamp(Math.min(availW / naturalW, availH / naturalH, 1), 0.1, 6);
      setScale(vp, scale);

      requestAnimationFrame(() => {
        const scaledW = naturalW * scale;
        const scaledH = naturalH * scale;

        frame.scrollLeft = Math.max(0, (scaledW - availW) / 2);
        frame.scrollTop  = Math.max(0, (scaledH - availH) / 2);
      });
      return true;
    }

    function fitOnOpen(details, vp){
      const frame = details.querySelector(".svg-frame");
      if(!(frame instanceof HTMLElement)){
        fitAndCenter(vp);
        return;
      }

      let lastW = -1;
      let lastH = -1;
      let stableCount = 0;
      let rafCount = 0;
      const maxFrames = 30;

      function step(){
        fitAndCenter(vp);

        const w = frame.clientWidth;
        const h = frame.clientHeight;

        if (w === lastW && h === lastH) {
          stableCount += 1;
        } else {
          stableCount = 0;
          lastW = w;
          lastH = h;
        }

        rafCount += 1;

        if (stableCount >= 3 || rafCount >= maxFrames) {
          fitAndCenter(vp); // one final pass, same as reset target state
          return;
        }

        requestAnimationFrame(step);
      }

  requestAnimationFrame(step);
}

    document.addEventListener("click", (ev) => {
      const el = ev.target instanceof HTMLElement ? ev.target : null;
      if(!el) return;

      const zin = el.closest("button[data-zoom-in]");
      const zout = el.closest("button[data-zoom-out]");
      const zreset = el.closest("button[data-zoom-reset]");

      if(zin){
        const id = zin.getAttribute("data-zoom-in");
        const vp = id ? getViewport(id) : null;
        if(!vp) return;
        setScale(vp, clamp(getScale(vp) * 1.15, 0.25, 6));
        return;
      }

      if(zout){
        const id = zout.getAttribute("data-zoom-out");
        const vp = id ? getViewport(id) : null;
        if(!vp) return;
        setScale(vp, clamp(getScale(vp) / 1.15, 0.25, 6));
        return;
      }

      if(zreset){
        const id = zreset.getAttribute("data-zoom-reset");
        const vp = id ? getViewport(id) : null;
        if(!vp) return;
        fitAndCenter(vp);
        return;
      }
    });

    // Wheel zoom when hovering the SVG frame
    document.addEventListener("wheel", (ev) => {
      const target = ev.target instanceof HTMLElement ? ev.target : null;
      if(!target) return;

      const frame = target.closest(".svg-frame");
      if(!frame) return;

      ev.preventDefault();

      const vp = frame.querySelector(".svg-viewport");
      if(!(vp instanceof HTMLElement)) return;

      const s0 = getScale(vp);
      const factor = (ev.deltaY < 0) ? 1.10 : (1/1.10);
      const s1 = clamp(s0 * factor, 0.25, 6);

      // zoom around cursor position
      const rect = frame.getBoundingClientRect();
      const cx = ev.clientX - rect.left + frame.scrollLeft;
      const cy = ev.clientY - rect.top + frame.scrollTop;

      setScale(vp, s1);

      frame.scrollLeft = cx * (s1 / s0) - (ev.clientX - rect.left);
      frame.scrollTop  = cy * (s1 / s0) - (ev.clientY - rect.top);
    }, { passive: false });

    // Drag-to-pan (scroll)
    let drag = null;
    document.addEventListener("mousedown", (ev) => {
      const target = ev.target instanceof HTMLElement ? ev.target : null;
      if(!target) return;

      const frame = target.closest(".svg-frame");
      if(!frame) return;

      drag = {
        frame,
        x: ev.clientX,
        y: ev.clientY,
        sl: frame.scrollLeft,
        st: frame.scrollTop
      };
    });

    document.addEventListener("mousemove", (ev) => {
      if(!drag) return;
      const dx = ev.clientX - drag.x;
      const dy = ev.clientY - drag.y;
      drag.frame.scrollLeft = drag.sl - dx;
      drag.frame.scrollTop  = drag.st - dy;
    });

    document.addEventListener("mouseup", () => {
      drag = null;
    });

    document.addEventListener("toggle", (ev) => {
      const details = ev.target;
      if(!(details instanceof HTMLDetailsElement)) return;
      
      // Only trigger if we are opening
      if(!details.open) return;

      const vp = details.querySelector(".svg-viewport");
      if(!(vp instanceof HTMLElement)) return;

      // Force a small delay to allow the browser to calculate layout
      // and then run the stable retry loop
      requestAnimationFrame(() => {
        fitOnOpen(details, vp);
      });
    }, true); // Use capture phase to ensure we catch it early

    window.addEventListener("load", () => {
      document.querySelectorAll(".svg-viewport").forEach(vp => {
        if(!(vp instanceof HTMLElement)) return;

        const details = vp.closest("details");
        if(details instanceof HTMLDetailsElement){
          if(!details.open) return;
          fitOnOpen(details, vp);
        } else {
          fitAndCenter(vp);
        }
      });
    });

  })();
  """
