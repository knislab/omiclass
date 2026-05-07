import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional
import matplotlib.pyplot as plt
from omiclass.utils.io import load_matrix, fig_to_svg_str


def plot_volcano_from_limma(
    de_results: Union[pd.DataFrame, str, Path],
    *,
    markers: Optional[List[str]],
    logfc_thresh: float = 1.0,
    fdr_thresh: float = 0.05,
    max_labels: int = 20,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:

    if isinstance(de_results, (str, Path)):
        de_df = load_matrix(de_results)
    else:
        de_df = de_results.copy()

    df = de_df[["logFC", "P.Value", "adj.P.Val"]].copy()
    df["neg_log10_p"] = -np.log10(df["P.Value"].clip(lower=1e-300))
    df["gene"] = df.index.astype(str)

    sig = (df["adj.P.Val"] <= fdr_thresh) & (df["logFC"].abs() >= logfc_thresh)
    df["is_signif"] = sig
    df["is_marker"] = df["gene"].isin(markers)

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        created = True

    nonsig = df[~df["is_signif"]]
    ax.scatter(nonsig["logFC"], nonsig["neg_log10_p"], s=10, alpha=0.3)

    sig_df = df[df["is_signif"] & ~df["is_marker"]]
    ax.scatter(sig_df["logFC"], sig_df["neg_log10_p"], s=12, alpha=0.6)

    markers = markers or []
    mark_df = df[df["is_marker"]]
    if len(mark_df) > 0:
        ax.scatter(
            mark_df["logFC"],
            mark_df["neg_log10_p"],
            s=35,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.7,
        )
        for _, row in mark_df.nlargest(max_labels, "neg_log10_p").iterrows():
            ax.text(row["logFC"], row["neg_log10_p"], row["gene"],
                    fontsize=8, ha="right", va="bottom")

    ax.axhline(-np.log10(fdr_thresh), linestyle="--", linewidth=1)
    ax.axvline(logfc_thresh, linestyle="--", linewidth=1)
    ax.axvline(-logfc_thresh, linestyle="--", linewidth=1)
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel("-log10(p-value)")

    if created:
        ax.figure.tight_layout()

    svg = fig_to_svg_str(ax.figure)
    if created:
        plt.close(ax.figure)

    return svg
