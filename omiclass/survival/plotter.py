# omiclass/survival/plotter.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence, Union
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests


from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test, logrank_test
from omiclass.utils.io import fig_to_svg_str
from omiclass.utils.toolbox import get_class_display_name




# ----------------- confidence plots ----------------- #

def plot_confidence_histogram(
    y_proba_df: pd.DataFrame,
    model_name: str,
) -> str:
    confidence_scores = np.max(y_proba_df.values, axis=1)

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(confidence_scores, bins=20, kde=True, ax=ax)
    ax.set_xlabel("Prediction confidence (max probability)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    svg = fig_to_svg_str(fig, pdf_name="survival_confidence_histogram")
    plt.close(fig)
    return svg


def plot_confidence_violin(
    y_proba_df: pd.DataFrame,
    model_name: str,
    class_names: Optional[Sequence[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a violin plot of confidence per predicted class to output_path.
    If class_names is None, use y_proba_df.columns as labels.
    """
    proba_values = y_proba_df.values
    confidence_scores = np.max(proba_values, axis=1)
    pred_classes = np.argmax(proba_values, axis=1)

    class_names = [get_class_display_name(i, cfg) for i in range(y_proba_df.shape[1])]

    pred_class_labels = [class_names[i] for i in pred_classes]

    df = pd.DataFrame({"Confidence": confidence_scores, "Predicted Class": pred_class_labels})

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(x="Predicted Class", y="Confidence", data=df, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Confidence score")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    svg = fig_to_svg_str(fig, pdf_name="survival_confidence_violin")
    plt.close(fig)
    return svg


# ----------------- KM + Cox forest ----------------- #

def _to_binary_event(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x != 0)
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    s = str(x).strip().lower()
    if s in {"true", "t", "yes", "y"}:
        return 1
    if s in {"false", "f", "no", "n"}:
        return 0
    try:
        return int(float(s) != 0.0)
    except Exception:
        return np.nan


def plot_km_and_forest(
    df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    group_col: str,
    time_unit_label: str = "years",
    km_output_path: Optional[Path] = None,
    forest_output_path: Optional[Path] = None,
    min_group_n: int = 3,
    adjust_method: str = "bonferroni",
    ref_class: Optional[Union[int, str]] = None,
    title: str = "Kaplan–Meier by groups",
    cfg: Any = None,
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """
    Plot Kaplan–Meier curves + separate Cox forest plot.
    Returns (km_svg, forest_svg, stats_dict).
    """

    df = df[[time_col, event_col, group_col]].copy()

    # ---------- Clean ----------
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = df[event_col].apply(_to_binary_event).astype("Float64")
    df = df.dropna(subset=[time_col, event_col, group_col])
    df = df[df[time_col] >= 0]

    # ---------- Filter small groups ----------
    valid_groups = df[group_col].value_counts()[lambda s: s >= min_group_n].index
    df = df[df[group_col].isin(valid_groups)]

    stats: Dict[str, Any] = {}

    if df.empty:
        return None, None, stats
    
    class_values = sorted(df[group_col].dropna().unique())

    # ---------- KM plot ----------
    fitters, labels = [], []

    fig_km, ax_km = plt.subplots(figsize=(7, 6))

    # label mapping is left generic
    for g, sub in df.groupby(group_col):
        label = get_class_display_name(g, cfg=cfg, class_values=class_values)
        kmf_g = KaplanMeierFitter()
        kmf_g.fit(sub[time_col], event_observed=sub[event_col], label=label)
        kmf_g.plot_survival_function(ax=ax_km, ci_show=False)
        fitters.append(kmf_g)
        labels.append(label)

    ax_km.set_xlabel(f"Time ({time_unit_label})")
    ax_km.set_ylabel("Survival probability")
    ax_km.legend(title=str(group_col))
    add_at_risk_counts(*fitters, labels=labels, ax=ax_km)
    fig_km.tight_layout()

    # ---------- Global log-rank ----------
    global_res = multivariate_logrank_test(
        event_durations=df[time_col],
        groups=df[group_col].astype(str),
        event_observed=df[event_col].astype(int),
    )
    global_p = float(global_res.p_value)
    ax_km.text(
        0.02,
        0.02,
        f"Global log-rank p = {global_p:.3e}",
        transform=ax_km.transAxes,
        fontsize=10,
        verticalalignment="bottom",
    )

    stats["global_p"] = global_p

    # Save KM
    km_svg = fig_to_svg_str(fig_km, pdf_name="kaplan_meier")
    plt.close(fig_km)

    # ---------- Pairwise log-rank ----------
    groups = sorted(df[group_col].unique())
    raw_p: Dict[tuple, float] = {}
    for g1, g2 in combinations(groups, 2):
        A = df[df[group_col] == g1]
        B = df[df[group_col] == g2]
        res = logrank_test(
            A[time_col],
            B[time_col],
            event_observed_A=A[event_col],
            event_observed_B=B[event_col],
        )
        raw_p[(g1, g2)] = float(res.p_value)

    m = len(raw_p)
    if adjust_method.lower().startswith("bonf"):
        adj_p = {k: min(v * m, 1.0) for k, v in raw_p.items()}
    elif adjust_method.lower().startswith("fdr"):
        _, qvals, _, _ = multipletests(list(raw_p.values()), method="fdr_bh")
        adj_p = dict(zip(raw_p.keys(), qvals))
    else:
        adj_p = raw_p

    stats["pairwise_p_raw"] = raw_p
    stats["pairwise_p_adj"] = adj_p

    # ---------- Cox model ----------
    forest_svg = None
    try:
        df_c = df[[time_col, event_col, group_col]].copy()

        # choose reference class
        uniq = sorted(df_c[group_col].unique())
        if ref_class is None:
            # default: first group in sorted order as reference
            ref_class = uniq[0]
        order = sorted(uniq, key=lambda x: x != ref_class)

        df_c[group_col] = pd.Categorical(df_c[group_col], categories=order, ordered=True)
        cph = CoxPHFitter()
        cph.fit(df_c, duration_col=time_col, event_col=event_col, formula=f"C({group_col})")
        cox_df = cph.summary.reset_index().rename(
            columns={
                "index": "covariate",
                "exp(coef)": "HR",
                "exp(coef) lower 95%": "HR_lower",
                "exp(coef) upper 95%": "HR_upper",
            }
        )

        stats["cox_summary"] = cph.summary

        # ---------- Forest plot ----------
        cox_df = cox_df.sort_values("HR", ascending=True)
        y_pos = np.arange(len(cox_df))

        fig_forest, ax_forest = plt.subplots(figsize=(5, 6))
        ax_forest.errorbar(
            cox_df["HR"],
            y_pos,
            xerr=[cox_df["HR"] - cox_df["HR_lower"], cox_df["HR_upper"] - cox_df["HR"]],
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=3,
        )
        ax_forest.axvline(1.0, color="red", linestyle="--", lw=1)
        ax_forest.set_yticks(y_pos)
        ax_forest.set_yticklabels(cox_df["covariate"])
        ax_forest.set_xlabel("Hazard ratio (log scale)")
        ax_forest.set_xscale("log")
        ax_forest.set_title("Cox model hazard ratios")
        ax_forest.grid(True, axis="x", linestyle="--", alpha=0.5)
        fig_forest.tight_layout()

        forest_svg = fig_to_svg_str(fig_forest, pdf_name="cox_forest")
        plt.close(fig_forest)
    except Exception as e:
        print(f"[survival] Cox fit failed: {type(e).__name__}: {e}")
    forest_svg = None

    return km_svg, forest_svg, stats
