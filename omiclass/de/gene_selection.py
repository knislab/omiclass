from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from omiclass.utils.surface_markers import apply_surface_marker_filter_with_min


@dataclass
class GeneSetSelectionSummary:
    K: int
    n_classes: int
    fdr_thresh: float
    logfc_thresh: float
    k_pan: int
    k_per_contrast: int
    pan_genes: List[str]
    by_contrast: Dict[str, List[str]]
    combined: List[str]
    contrast_order: List[str]


def select_marker_genes(
    de_results: pd.DataFrame,
    K: int = 100,
    logfc_thresh: float = 1.0,
    fdr_thresh: float = 0.05,
    n_classes: int = 2,
    surface_markers: bool = False,
    min_surface_markers: int = 20,
) -> GeneSetSelectionSummary:

    if de_results.empty or K <= 0:
        return GeneSetSelectionSummary(
            K=K,
            n_classes=n_classes,
            fdr_thresh=fdr_thresh,
            logfc_thresh=logfc_thresh,
            k_pan=0,
            k_per_contrast=0,
            pan_genes=[],
            by_contrast={},
            combined=[],
            contrast_order=[],
        )

    df = de_results.copy()

    if surface_markers:
        df = apply_surface_marker_filter_with_min(df, min_features=min_surface_markers)

    df["P.Value"] = df["P.Value"].clip(lower=1e-300)
    df["Score"] = df["logFC"].abs() * (-np.log10(df["P.Value"]))

    sig = (df["adj.P.Val"] <= fdr_thresh) & (df["logFC"].abs() >= logfc_thresh)
    working = df[sig] if sig.sum() >= K else df

    per_gene_score = working.groupby(working.index)["Score"].max()
    pan_sorted = per_gene_score.sort_values(ascending=False).index.astype(str).tolist()

    if n_classes <= 2:
        genes = pan_sorted[:K]
        return GeneSetSelectionSummary(
            K=K,
            n_classes=n_classes,
            fdr_thresh=fdr_thresh,
            logfc_thresh=logfc_thresh,
            k_pan=K,
            k_per_contrast=0,
            pan_genes=genes,
            by_contrast={},
            combined=genes,
            contrast_order=[],
        )

    k_pan = K // 2
    pan_genes = pan_sorted[:k_pan]
    k_remaining = max(K - k_pan, 0)

    if "Contrast" not in working.columns:
        combined = (pan_genes + pan_sorted)[0:K]
        return GeneSetSelectionSummary(
            K=K,
            n_classes=n_classes,
            fdr_thresh=fdr_thresh,
            logfc_thresh=logfc_thresh,
            k_pan=k_pan,
            k_per_contrast=0,
            pan_genes=pan_genes,
            by_contrast={},
            combined=combined,
            contrast_order=[],
        )

    contrast_order = (
        working["Contrast"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    contrast_counts = working.groupby(working.index)["Contrast"].nunique()
    unique_genes = contrast_counts[contrast_counts == 1].index.astype(str)
    unique_df = working.loc[unique_genes]

    k_per_contrast = max(k_remaining // max(len(contrast_order), 1), 1)

    by_contrast: Dict[str, List[str]] = {}
    selected_cs: List[str] = []

    for c in contrast_order:
        sub = unique_df[unique_df["Contrast"].astype(str) == c]
        scores = sub.groupby(sub.index)["Score"].max()
        ordered = scores.sort_values(ascending=False).index.astype(str).tolist()

        picked: List[str] = []
        for g in ordered:
            if g in pan_genes or g in selected_cs:
                continue
            picked.append(g)
            selected_cs.append(g)
            if len(picked) >= k_per_contrast:
                break

        by_contrast[c] = picked

    combined: List[str] = []
    for g in pan_genes + selected_cs:
        if g not in combined:
            combined.append(g)

    if len(combined) < K:
        for g in pan_sorted:
            if g not in combined:
                combined.append(g)
            if len(combined) >= K:
                break

    return GeneSetSelectionSummary(
        K=K,
        n_classes=n_classes,
        fdr_thresh=fdr_thresh,
        logfc_thresh=logfc_thresh,
        k_pan=k_pan,
        k_per_contrast=k_per_contrast,
        pan_genes=pan_genes,
        by_contrast=by_contrast,
        combined=combined[:K],
        contrast_order=contrast_order,
    )
