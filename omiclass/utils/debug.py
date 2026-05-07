from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd


@dataclass
class VolcanoDebugResult:
    ok: bool
    label: str
    csv_path: str
    n_rows: int
    columns: Sequence[str]
    required_cols_ok: bool
    usable_rows: int
    would_pass_thresholds: int
    svg_is_none: bool
    svg_len: int
    svg_saved_path: Optional[str]
    notes: Sequence[str]

@dataclass
class PredSurvDebugResult:
    ok: bool
    n_samples_x: int
    n_samples_proba: int
    n_samples_meta: int
    n_samples_overlap: int
    n_classes: int
    classes: List[str]
    notes: List[str]
    paths: Dict[str, str]


def debug_volcano_plot_from_limma(
    *,
    de_results: Union[str, Path, pd.DataFrame],
    out_dir: Union[str, Path],
    label: str = "volcano",
    # thresholds (must match your plotting call)
    fdr_thresh: float = 0.05,
    logfc_thresh: float = 1.0,
    # column names (match limma output; override if your R writes different names)
    logfc_col: str = "logFC",
    fdr_col_candidates: Sequence[str] = ("adj.P.Val", "FDR"),
    pval_col_candidates: Sequence[str] = ("P.Value",),
    contrast_col: str = "Contrast",
    # plot options
    title: Optional[str] = None,
    markers: Optional[Sequence[str]] = None,
    # behavior
    write_debug_csv: bool = True,
    write_svg: bool = True,
    raise_on_error: bool = False,
) -> VolcanoDebugResult:
    """
    Debug EVERYTHING around volcano plotting:
    - Reads DE results (path or df)
    - Validates required columns for volcano (logFC + FDR/adj.P.Val)
    - Computes how many rows are usable and how many would pass thresholds
    - Optionally splits by Contrast (if present) by using 'label' only (caller controls grouping)
    - Calls plot_volcano_from_limma and saves SVG
    - Writes debug artifacts (CSV head, stats json-ish txt) into out_dir

    Returns a structured VolcanoDebugResult, so you can log it / attach to HTML.
    """
    # local import so this function is standalone + doesn't import plotting unless called
    from generic_classifier.plotting.volcano_plot import plot_volcano_from_limma

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []

    def _note(msg: str) -> None:
        notes.append(msg)

    def _safe_name(s: str) -> str:
        s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))
        return s[:180] if len(s) > 180 else s

    safe_label = _safe_name(label)
    dump_prefix = out_dir / f"debug_{safe_label}"

    # ---------- load df ----------
    df: pd.DataFrame
    csv_path: Optional[Path] = None

    try:
        if isinstance(de_results, pd.DataFrame):
            df = de_results.copy()
            _note("Input provided as DataFrame.")
        else:
            csv_path = Path(de_results)
            if not csv_path.exists():
                msg = f"DE results file not found: {csv_path}"
                _note(msg)
                if raise_on_error:
                    raise FileNotFoundError(msg)
                return VolcanoDebugResult(
                    ok=False,
                    label=label,
                    csv_path=str(csv_path),
                    n_rows=0,
                    columns=[],
                    required_cols_ok=False,
                    usable_rows=0,
                    would_pass_thresholds=0,
                    svg_is_none=True,
                    svg_len=0,
                    svg_saved_path=None,
                    notes=notes,
                )
            if csv_path.stat().st_size == 0:
                msg = f"DE results file exists but is empty: {csv_path}"
                _note(msg)
                if raise_on_error:
                    raise ValueError(msg)
                return VolcanoDebugResult(
                    ok=False,
                    label=label,
                    csv_path=str(csv_path),
                    n_rows=0,
                    columns=[],
                    required_cols_ok=False,
                    usable_rows=0,
                    would_pass_thresholds=0,
                    svg_is_none=True,
                    svg_len=0,
                    svg_saved_path=None,
                    notes=notes,
                )

            # try common patterns: index_col=0 if gene IDs are rownames
            df = pd.read_csv(csv_path, index_col=0)
            _note(f"Loaded CSV with index_col=0: shape={df.shape}")
    except Exception as e:
        _note(f"Failed loading DE results: {type(e).__name__}: {e}")
        if raise_on_error:
            raise
        return VolcanoDebugResult(
            ok=False,
            label=label,
            csv_path=str(csv_path) if csv_path else "<dataframe>",
            n_rows=0,
            columns=[],
            required_cols_ok=False,
            usable_rows=0,
            would_pass_thresholds=0,
            svg_is_none=True,
            svg_len=0,
            svg_saved_path=None,
            notes=notes,
        )

    # ---------- basic sanity ----------
    if df.empty:
        _note("DE DataFrame is empty after loading.")
    cols = [str(c) for c in df.columns.tolist()]

    # Determine FDR column
    fdr_col: Optional[str] = None
    for c in fdr_col_candidates:
        if c in df.columns:
            fdr_col = c
            break

    # Determine p-value column (optional for thresholds, but useful)
    pval_col: Optional[str] = None
    for c in pval_col_candidates:
        if c in df.columns:
            pval_col = c
            break

    if logfc_col not in df.columns:
        _note(f"Missing logFC column: expected '{logfc_col}'. Available: {cols}")
    if fdr_col is None:
        _note(f"Missing FDR/adj column: tried {list(fdr_col_candidates)}. Available: {cols}")
    if pval_col is None:
        _note(f"P-value column not found (optional): tried {list(pval_col_candidates)}. Available: {cols}")

    required_cols_ok = (logfc_col in df.columns) and (fdr_col is not None)

    # ---------- numeric coercion + stats ----------
    usable_rows = 0
    would_pass = 0

    if required_cols_ok and not df.empty:
        logfc = pd.to_numeric(df[logfc_col], errors="coerce")
        fdr = pd.to_numeric(df[fdr_col], errors="coerce")  # type: ignore[index]
        usable_mask = logfc.notna() & fdr.notna()
        usable_rows = int(usable_mask.sum())

        if usable_rows == 0:
            _note("0 usable rows after coercion (logFC and FDR are all NaN or missing).")
        else:
            pass_mask = (fdr <= float(fdr_thresh)) & (np.abs(logfc) >= float(logfc_thresh))
            would_pass = int((pass_mask & usable_mask).sum())

            _note(f"usable_rows={usable_rows} / n={len(df)}")
            _note(f"would_pass_thresholds={would_pass} with fdr<={fdr_thresh}, |logFC|>={logfc_thresh}")

            # range summaries
            _note(f"logFC: min={np.nanmin(logfc.values):.4g}, median={np.nanmedian(logfc.values):.4g}, max={np.nanmax(logfc.values):.4g}")
            _note(f"{fdr_col}: min={np.nanmin(fdr.values):.4g}, median={np.nanmedian(fdr.values):.4g}, max={np.nanmax(fdr.values):.4g}")  # type: ignore[arg-type]

            if pval_col and pval_col in df.columns:
                pval = pd.to_numeric(df[pval_col], errors="coerce")
                if int(pval.notna().sum()) > 0:
                    _note(f"{pval_col}: min={np.nanmin(pval.values):.4g}, median={np.nanmedian(pval.values):.4g}, max={np.nanmax(pval.values):.4g}")

    # ---------- write debug artifacts ----------
    # always write a "head" snapshot for inspection
    try:
        head_path = dump_prefix.with_suffix(".head.csv")
        df.head(200).to_csv(head_path)
        _note(f"Wrote head snapshot: {head_path}")
    except Exception as e:
        _note(f"Failed writing head snapshot: {type(e).__name__}: {e}")

    # optionally write full debug csv (helpful if you passed a DataFrame, or want to freeze the exact input)
    debug_csv_path: Optional[Path] = None
    if write_debug_csv:
        try:
            debug_csv_path = dump_prefix.with_suffix(".full.csv")
            df.to_csv(debug_csv_path)
            _note(f"Wrote full debug CSV: {debug_csv_path}")
        except Exception as e:
            _note(f"Failed writing full debug CSV: {type(e).__name__}: {e}")

    # write a text report
    try:
        report_path = dump_prefix.with_suffix(".report.txt")
        report_lines = [
            f"label: {label}",
            f"source_csv: {csv_path if csv_path else '<dataframe>'}",
            f"shape: {df.shape}",
            f"columns: {cols}",
            f"contrast_present: {contrast_col in df.columns}",
            f"required_cols_ok: {required_cols_ok}",
            f"logfc_col: {logfc_col}",
            f"fdr_col: {fdr_col}",
            f"pval_col: {pval_col}",
            f"usable_rows: {usable_rows}",
            f"would_pass_thresholds: {would_pass}",
            "",
            "notes:",
            *[f"- {n}" for n in notes],
        ]
        report_path.write_text("\n".join(report_lines))
        _note(f"Wrote report: {report_path}")
    except Exception as e:
        _note(f"Failed writing report: {type(e).__name__}: {e}")

    # ---------- call the plotter ----------
    svg: Optional[str] = None
    svg_saved_path: Optional[str] = None

    try:
        # plot_volcano_from_limma expects a PATH (in your code), so ensure we pass a file
        if isinstance(de_results, pd.DataFrame) or csv_path is None:
            # ensure we have a file path for plotter
            plot_input = debug_csv_path if debug_csv_path else dump_prefix.with_suffix(".plot_input.csv")
            if plot_input and not plot_input.exists():
                df.to_csv(plot_input)
            plot_input_path = plot_input
        else:
            plot_input_path = csv_path

        svg = plot_volcano_from_limma(
            de_results=plot_input_path,
            fdr_thresh=float(fdr_thresh),
            logfc_thresh=float(logfc_thresh),
            title=title if title is not None else f"Volcano: {label}",
            markers=list(markers) if markers is not None else [],
        )

        if svg is None:
            _note("plot_volcano_from_limma returned None.")
        else:
            _note(f"plot_volcano_from_limma returned SVG length={len(svg)}.")

        if write_svg and svg:
            svg_path = dump_prefix.with_suffix(".svg")
            svg_path.write_text(svg)
            svg_saved_path = str(svg_path)
            _note(f"Wrote SVG: {svg_path}")

    except Exception as e:
        _note(f"Plot call failed: {type(e).__name__}: {e}")
        if raise_on_error:
            raise

    ok = (svg is not None) and (isinstance(svg, str)) and (len(svg) > 50)

    # Helpful extra note: if plot is "empty", this often explains why
    if required_cols_ok and usable_rows > 0 and would_pass == 0:
        _note("Likely reason for 'empty': no points pass thresholds. Try lowering logfc_thresh/fdr_thresh.")
    if not required_cols_ok:
        _note("Likely reason for 'empty': column mismatch. Confirm what plot_volcano_from_limma expects vs your R output.")

    return VolcanoDebugResult(
        ok=ok,
        label=label,
        csv_path=str(csv_path) if csv_path else "<dataframe>",
        n_rows=int(len(df)),
        columns=cols,
        required_cols_ok=bool(required_cols_ok),
        usable_rows=int(usable_rows),
        would_pass_thresholds=int(would_pass),
        svg_is_none=(svg is None),
        svg_len=(len(svg) if isinstance(svg, str) else 0),
        svg_saved_path=svg_saved_path,
        notes=notes,
    )


def debug_prediction_and_survival_inputs(
    *,
    # Required
    y_proba_df: pd.DataFrame,                 # index = samples, columns = class names, values = probabilities
    meta_df: Union[pd.DataFrame, str, Path],  # clinical/survival metadata (index = samples)
    out_dir: Union[str, Path],

    # Optional alignment reference (recommended if you have it)
    X_index: Optional[Sequence[str]] = None,  # e.g., X_aligned.index; used only for sanity checks

    # Survival columns (you must tell the function what you use)
    time_col: str = "os_days",                # e.g. "TTFT_days" or "os_days"
    event_col: str = "os",                    # 1=event, 0=censored
    # or if your pipeline uses years columns like T5/T6, you can pass those separately:
    time_years_col: Optional[str] = None,     # e.g. "T6"
    years_to_days: float = 365.25,

    # Prediction handling
    expected_n_classes: Optional[int] = 4,
    allow_unity_tol: float = 1e-3,            # tolerance for proba row sum ~ 1
    min_group_size_for_km: int = 5,
    min_events_for_km: int = 3,

    # Diagnostics output options
    dump_csv: bool = True,
    max_rows_dump: int = 5000,
) -> PredSurvDebugResult:
    """
    Debugs everything that commonly makes prediction+survival plots "look wrong" or fail silently:
    - index alignment between X / y_proba / meta
    - probability sanity (NaNs, negatives, row sums, argmax distribution)
    - survival sanity (missing/negative times, all-censored, event coding)
    - per-class sample counts and per-class event counts (KM needs events!)
    - writes a text report + optional CSV snapshots to out_dir
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    notes: List[str] = []

    def _note(msg: str) -> None:
        notes.append(msg)

    # ---- load meta ----
    if isinstance(meta_df, (str, Path)):
        meta_path = Path(meta_df)
        if not meta_path.exists():
            return PredSurvDebugResult(
                ok=False,
                n_samples_x=len(X_index) if X_index is not None else 0,
                n_samples_proba=len(y_proba_df),
                n_samples_meta=0,
                n_samples_overlap=0,
                n_classes=int(y_proba_df.shape[1]),
                classes=[str(c) for c in y_proba_df.columns.tolist()],
                notes=[f"meta_df path not found: {meta_path}"],
                paths={},
            )
        mdf = pd.read_csv(meta_path, index_col=0)
        paths["meta_loaded_from"] = str(meta_path)
        _note(f"Loaded meta_df from {meta_path} with shape={mdf.shape}")
    else:
        mdf = meta_df.copy()
        _note(f"meta_df provided as DataFrame with shape={mdf.shape}")

    # ---- ensure string indices ----
    y_proba = y_proba_df.copy()
    y_proba.index = y_proba.index.astype(str)
    mdf.index = mdf.index.astype(str)

    X_idx = None
    if X_index is not None:
        X_idx = pd.Index([str(x) for x in X_index])
        _note(f"X_index provided: n={len(X_idx)}")

    # ---- index overlap ----
    overlap = y_proba.index.intersection(mdf.index)
    _note(f"Index overlap(y_proba, meta) = {len(overlap)} / proba={len(y_proba)} meta={len(mdf)}")

    if X_idx is not None:
        overlap_x = pd.Index(y_proba.index).intersection(X_idx)
        _note(f"Index overlap(y_proba, X_index) = {len(overlap_x)} / proba={len(y_proba)} X={len(X_idx)}")

    if len(overlap) == 0:
        # hard fail: plots will be nonsense
        report_path = out_dir / "pred_surv_debug.report.txt"
        report_path.write_text("\n".join(["ERROR: no overlap between y_proba_df and meta_df indices."] + notes))
        paths["report"] = str(report_path)
        return PredSurvDebugResult(
            ok=False,
            n_samples_x=len(X_idx) if X_idx is not None else 0,
            n_samples_proba=len(y_proba),
            n_samples_meta=len(mdf),
            n_samples_overlap=0,
            n_classes=int(y_proba.shape[1]),
            classes=[str(c) for c in y_proba.columns.tolist()],
            notes=notes,
            paths=paths,
        )

    # align to overlap
    y_proba = y_proba.loc[overlap]
    mdf = mdf.loc[overlap]

    # ---- proba sanity ----
    classes = [str(c) for c in y_proba.columns.tolist()]
    n_classes = len(classes)
    if expected_n_classes is not None and n_classes != int(expected_n_classes):
        _note(f"WARNING: expected_n_classes={expected_n_classes} but y_proba has {n_classes} columns.")

    # numeric coercion
    y_num = y_proba.apply(pd.to_numeric, errors="coerce")
    n_nan = int(y_num.isna().any(axis=1).sum())
    n_inf = int(np.isinf(y_num.to_numpy()).any(axis=1).sum())
    if n_nan:
        _note(f"WARNING: {n_nan} rows in y_proba have NaN(s).")
    if n_inf:
        _note(f"WARNING: {n_inf} rows in y_proba have inf(s).")

    neg_count = int((y_num < 0).any(axis=1).sum())
    if neg_count:
        _note(f"WARNING: {neg_count} rows have negative probabilities.")

    row_sums = y_num.sum(axis=1)
    bad_sum = int(((row_sums - 1.0).abs() > float(allow_unity_tol)).sum())
    _note(f"Proba row-sum check: bad={bad_sum}/{len(row_sums)} (tol={allow_unity_tol})")
    if bad_sum:
        _note(f"Row-sum stats: min={row_sums.min():.4g} median={row_sums.median():.4g} max={row_sums.max():.4g}")

    # predicted class + confidence
    pred = y_num.idxmax(axis=1).astype(str)
    conf = y_num.max(axis=1)

    pred_counts = pred.value_counts(dropna=False).to_dict()
    _note(f"Predicted class distribution: {pred_counts}")
    _note(f"Confidence: min={conf.min():.4g} median={conf.median():.4g} max={conf.max():.4g}")

    # ---- survival sanity ----
    # derive time if given as years
    if time_years_col is not None and time_col not in mdf.columns:
        if time_years_col in mdf.columns:
            t_years = pd.to_numeric(mdf[time_years_col], errors="coerce")
            mdf[time_col] = t_years * float(years_to_days)
            _note(f"Derived {time_col} from {time_years_col} * {years_to_days}")
        else:
            _note(f"ERROR: time_col '{time_col}' missing and time_years_col '{time_years_col}' missing too.")

    missing_time = time_col not in mdf.columns
    missing_event = event_col not in mdf.columns
    if missing_time:
        _note(f"ERROR: time_col '{time_col}' not found in meta_df.")
    if missing_event:
        _note(f"ERROR: event_col '{event_col}' not found in meta_df.")

    # If time/event columns missing, still dump prediction sanity report (but survival plots will be broken)
    if missing_time or missing_event:
        report_path = out_dir / "pred_surv_debug.report.txt"
        report_path.write_text("\n".join(["ERROR: missing survival columns."] + notes))
        paths["report"] = str(report_path)

        if dump_csv:
            snap = pd.DataFrame({"pred": pred, "conf": conf})
            p = out_dir / "pred_surv_debug.pred_snapshot.csv"
            snap.head(max_rows_dump).to_csv(p)
            paths["pred_snapshot_csv"] = str(p)

        return PredSurvDebugResult(
            ok=False,
            n_samples_x=len(X_idx) if X_idx is not None else 0,
            n_samples_proba=len(y_proba_df),
            n_samples_meta=len(meta_df) if isinstance(meta_df, pd.DataFrame) else len(pd.read_csv(Path(meta_df), index_col=0)),
            n_samples_overlap=len(overlap),
            n_classes=n_classes,
            classes=classes,
            notes=notes,
            paths=paths,
        )

    time = pd.to_numeric(mdf[time_col], errors="coerce")
    event = pd.to_numeric(mdf[event_col], errors="coerce")

    # event normalization: allow bool / 0/1 / True/False
    # (don’t change your real data, just diagnose)
    event_bool = event.fillna(0)
    # if someone encoded events as {1,2} or {0,1,2} etc, flag it:
    uniq_event = sorted(pd.unique(event_bool.dropna()).tolist())
    _note(f"Event unique values (after NaN->0 for inspection): {uniq_event}")

    n_time_nan = int(time.isna().sum())
    n_time_neg = int((time < 0).sum(skipna=True))
    n_time_zero = int((time == 0).sum(skipna=True))
    _note(f"Time '{time_col}': NaN={n_time_nan} neg={n_time_neg} zero={n_time_zero} "
          f"min={time.min(skipna=True):.4g} median={time.median(skipna=True):.4g} max={time.max(skipna=True):.4g}")

    n_event_nan = int(event.isna().sum())
    _note(f"Event '{event_col}': NaN={n_event_nan} "
          f"mean={event.mean(skipna=True):.4g} sum={event.sum(skipna=True):.4g}")

    # global event rate sanity
    # if it's all censored, KM curves can look "flat"/identical
    try:
        total_events = int((event.fillna(0) > 0).sum())
        _note(f"Total events (event>0): {total_events}/{len(event)}")
    except Exception:
        total_events = 0
        _note("Could not compute total_events (event>0).")

    # per predicted class survival readiness
    surv_ready = pd.DataFrame(
        {
            "pred": pred,
            "conf": conf,
            "time": time,
            "event": event,
        },
        index=mdf.index,
    )

    per_class = []
    for c in classes:
        sub = surv_ready[surv_ready["pred"] == c]
        n = len(sub)
        n_valid_time = int(sub["time"].notna().sum())
        n_events = int((sub["event"].fillna(0) > 0).sum())
        per_class.append((c, n, n_valid_time, n_events))

    _note("Per-predicted-class (class, n, valid_time, events): " + str(per_class))

    # flags that usually explain “weird KM”
    too_small = [c for (c, n, _, _) in per_class if n < min_group_size_for_km]
    too_few_events = [c for (c, _, _, e) in per_class if e < min_events_for_km]
    if too_small:
        _note(f"WARNING: small predicted groups (n<{min_group_size_for_km}): {too_small}")
    if too_few_events:
        _note(f"WARNING: groups with few events (events<{min_events_for_km}): {too_few_events}")
    if total_events < max(min_events_for_km, 3):
        _note("WARNING: overall event count is very low — KM curves will be unstable / uninformative.")

    # ---- write artifacts ----
    if dump_csv:
        # compact snapshot used for debugging survival plots
        snap = surv_ready.copy()
        snap["meta_has_time"] = snap["time"].notna()
        snap_path = out_dir / "pred_surv_debug.snapshot.csv"
        snap.head(max_rows_dump).to_csv(snap_path)
        paths["snapshot_csv"] = str(snap_path)

        # save full proba maybe
        proba_path = out_dir / "pred_surv_debug.y_proba_aligned.csv"
        y_proba.head(max_rows_dump).to_csv(proba_path)
        paths["proba_aligned_csv"] = str(proba_path)

        meta_path = out_dir / "pred_surv_debug.meta_aligned.csv"
        mdf.head(max_rows_dump).to_csv(meta_path)
        paths["meta_aligned_csv"] = str(meta_path)

    report_path = out_dir / "pred_surv_debug.report.txt"
    report_path.write_text("\n".join(notes))
    paths["report"] = str(report_path)

    ok = True
    # treat as not-ok if survival columns exist but are unusable
    if len(overlap) == 0 or missing_time or missing_event:
        ok = False
    if n_time_nan == len(time):
        ok = False
    if (event.fillna(0) > 0).sum() == 0:
        ok = False

    return PredSurvDebugResult(
        ok=ok,
        n_samples_x=len(X_idx) if X_idx is not None else 0,
        n_samples_proba=int(len(y_proba_df)),
        n_samples_meta=int(len(meta_df) if isinstance(meta_df, pd.DataFrame) else len(pd.read_csv(Path(meta_df), index_col=0))),
        n_samples_overlap=int(len(overlap)),
        n_classes=int(n_classes),
        classes=classes,
        notes=notes,
        paths=paths,
    )