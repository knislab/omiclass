from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from omiclass.utils.io import fig_to_svg_str 


def plot_geneset_top_frequency_bar(
    *,
    gene_df: pd.DataFrame,
    top_n: int,
    title: str,
) -> Optional[str]:
    if gene_df is None or gene_df.empty:
        return None
    if "gene" not in gene_df.columns or "count_folds" not in gene_df.columns:
        return None

    df = gene_df.copy()
    df["count_folds"] = pd.to_numeric(df["count_folds"], errors="coerce").fillna(0).astype(int)
    df["gene"] = df["gene"].astype(str)

    df = df.sort_values(["count_folds", "gene"], ascending=[False, True]).head(int(top_n))
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(df)), 4))
    ax.bar(df["gene"].tolist(), df["count_folds"].tolist())
    ax.set_xticklabels(df["gene"].tolist(), rotation=90)
    ax.set_ylabel("Fold frequency")
    ax.set_title(title)
    fig.tight_layout()
    svg = fig_to_svg_str(fig, pdf_name="geneset_top_frequency_bar")
    plt.close(fig)
    return svg
