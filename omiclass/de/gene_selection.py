from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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
    target_k: int = 0
    one_vs_rest_budget: int = 0
    close_pair_budget: int = 0
    one_vs_rest_genes: List[str] = field(default_factory=list)
    pairwise_genes: List[str] = field(default_factory=list)
    close_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    selection_summary: List[Dict[str, object]] = field(default_factory=list)


def compute_target_gene_count(
    *,
    n_classes: int,
    min_genes: int = 200,
    max_genes_per_class: int = 75,
    max_genes: int = 600,
    genes_per_class: Optional[int] = None,
) -> int:
    per_class = int(genes_per_class) if genes_per_class is not None else int(max_genes_per_class)
    target_k = max(int(min_genes), per_class * int(n_classes))
    return min(target_k, int(max_genes))


def _prepare_de_frame(
    de_results: Optional[pd.DataFrame],
    *,
    source: str,
    fdr_thresh: float,
    logfc_thresh: float,
    surface_markers: bool,
    min_surface_markers: int,
) -> pd.DataFrame:
    if de_results is None or de_results.empty:
        return pd.DataFrame()

    df = de_results.copy()
    if surface_markers:
        from omiclass.utils.surface_markers import apply_surface_marker_filter_with_min

        df = apply_surface_marker_filter_with_min(df, min_features=min_surface_markers)
    if df.empty:
        return df

    if "gene_id" in df.columns:
        df = df.set_index("gene_id", drop=True)
    df.index = df.index.astype(str)
    df["P.Value"] = pd.to_numeric(df["P.Value"], errors="coerce").clip(lower=1e-300)
    df["adj.P.Val"] = pd.to_numeric(df["adj.P.Val"], errors="coerce")
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df = df.dropna(subset=["P.Value", "adj.P.Val", "logFC"])
    df["score"] = df["logFC"] * (-np.log10(df["P.Value"]))
    df["source"] = source
    if "Contrast" not in df.columns:
        df["Contrast"] = source
    df["Contrast"] = df["Contrast"].astype(str)
    df["passes_threshold"] = (
        (df["adj.P.Val"] <= fdr_thresh)
        & (df["logFC"].abs() >= logfc_thresh)
    )
    return df


def _row_to_record(gene: str, row: pd.Series, direction: str, selection_pool: str) -> Dict[str, object]:
    return {
        "gene": str(gene),
        "source": str(row.get("source", "")),
        "contrast": str(row.get("Contrast", "")),
        "direction": direction,
        "passed_threshold": bool(row.get("passes_threshold", False)),
        "selection_pool": selection_pool,
        "score": float(row.get("score", np.nan)),
        "logFC": float(row.get("logFC", np.nan)),
        "P.Value": float(row.get("P.Value", np.nan)),
        "adj.P.Val": float(row.get("adj.P.Val", np.nan)),
    }


def _ranked_directional_records(
    df: pd.DataFrame,
    *,
    min_genes_per_class: int,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, object]]], List[Dict[str, object]]]:
    ranked: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    summaries: List[Dict[str, object]] = []
    if df.empty:
        return ranked, summaries

    groups = (
        df[["source", "Contrast"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for source, contrast in groups:
        sub = df[
            (df["source"].astype(str) == source)
            & (df["Contrast"].astype(str) == contrast)
        ]
        n_total = int(len(sub))
        n_passed = int(sub["passes_threshold"].sum())
        selection_pool = "filtered" if n_passed >= int(min_genes_per_class) else "fallback"
        pool_df = sub[sub["passes_threshold"]] if selection_pool == "filtered" else sub
        summaries.append(
            {
                "contrast": contrast,
                "source": source,
                "n_total_genes": n_total,
                "n_passed_threshold": n_passed,
                "selection_pool": selection_pool,
                "n_selected": 0,
                "n_selected_positive": 0,
                "n_selected_negative": 0,
                "min_genes_per_class": int(min_genes_per_class),
            }
        )
        directions = [
            ("positive", pool_df[pool_df["logFC"] > 0], False),
            ("negative", pool_df[pool_df["logFC"] < 0], True),
        ]
        for direction, dsub, ascending in directions:
            if dsub.empty:
                ranked[(source, contrast, direction)] = []
                continue
            ordered = dsub.sort_values("score", ascending=ascending)
            seen: set[str] = set()
            records: List[Dict[str, object]] = []
            for gene, row in ordered.iterrows():
                gene = str(gene)
                if gene in seen:
                    continue
                seen.add(gene)
                records.append(_row_to_record(gene, row, direction, selection_pool))
            ranked[(source, contrast, direction)] = records
    return ranked, summaries


def _select_balanced(
    ranked: Dict[Tuple[str, str, str], List[Dict[str, object]]],
    *,
    budget: int,
    existing_genes: Optional[set[str]] = None,
    existing_records: Optional[Sequence[Dict[str, object]]] = None,
    max_genes_per_class: Optional[int] = None,
) -> List[Dict[str, object]]:
    if budget <= 0 or not ranked:
        return []

    keys = [key for key, records in ranked.items() if records]
    if not keys:
        return []

    existing = set(existing_genes or set())
    selected: List[Dict[str, object]] = []
    selected_genes: set[str] = set()
    max_per_contrast = int(max_genes_per_class) if max_genes_per_class is not None else None
    group_counts: Dict[Tuple[str, str], int] = {}
    for record in existing_records or []:
        group = (str(record.get("source", "")), str(record.get("contrast", "")))
        group_counts[group] = group_counts.get(group, 0) + 1
    pointers = {key: 0 for key in keys}
    quotas = {key: budget // len(keys) for key in keys}
    for key in keys[: budget % len(keys)]:
        quotas[key] += 1

    def take_one(key: Tuple[str, str, str]) -> Optional[Dict[str, object]]:
        records = ranked[key]
        pos = pointers[key]
        while pos < len(records):
            record = records[pos]
            pos += 1
            gene = str(record["gene"])
            if gene in existing or gene in selected_genes:
                continue
            group = (str(record.get("source", "")), str(record.get("contrast", "")))
            if max_per_contrast is not None and group_counts.get(group, 0) >= max_per_contrast:
                continue
            pointers[key] = pos
            selected_genes.add(gene)
            group_counts[group] = group_counts.get(group, 0) + 1
            return record
        pointers[key] = pos
        return None

    for key in keys:
        while len(selected) < budget and sum(1 for r in selected if (r["source"], r["contrast"], r["direction"]) == key) < quotas[key]:
            record = take_one(key)
            if record is None:
                break
            selected.append(record)

    made_progress = True
    while len(selected) < budget and made_progress:
        made_progress = False
        for key in keys:
            if len(selected) >= budget:
                break
            record = take_one(key)
            if record is not None:
                selected.append(record)
                made_progress = True

    return selected


def _dedupe_records(records: Sequence[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        gene = str(record["gene"])
        if gene in seen:
            continue
        seen.add(gene)
        out.append(record)
        if len(out) >= limit:
            break
    return out


def _records_by_contrast(records: Sequence[Dict[str, object]]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for record in records:
        contrast = str(record.get("contrast", ""))
        grouped.setdefault(contrast, [])
        gene = str(record.get("gene", ""))
        if gene and gene not in grouped[contrast]:
            grouped[contrast].append(gene)
    return grouped


def _finalize_selection_summary(
    summaries: Sequence[Dict[str, object]],
    final_records: Sequence[Dict[str, object]],
    *,
    max_genes_per_class: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for summary in summaries:
        source = str(summary.get("source", ""))
        contrast = str(summary.get("contrast", ""))
        selected = [
            record
            for record in final_records
            if str(record.get("source", "")) == source
            and str(record.get("contrast", "")) == contrast
        ]
        row = dict(summary)
        row["n_selected"] = len(selected)
        row["n_selected_positive"] = sum(1 for record in selected if str(record.get("direction", "")) == "positive")
        row["n_selected_negative"] = sum(1 for record in selected if str(record.get("direction", "")) == "negative")
        row["max_genes_per_class"] = int(max_genes_per_class)
        out.append(row)
    return out


def _write_selection_artifacts(
    *,
    artifacts_dir: Optional[Path],
    final_records: Sequence[Dict[str, object]],
    one_vs_rest_genes: Sequence[str],
    pairwise_genes: Sequence[str],
    selection_summary: Sequence[Dict[str, object]],
) -> None:
    if artifacts_dir is None:
        return

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    final_genes = [str(record["gene"]) for record in final_records]
    (artifacts_dir / "selected_genes_final.txt").write_text("\n".join(final_genes))
    (artifacts_dir / "selected_genes_one_vs_rest.txt").write_text("\n".join(map(str, one_vs_rest_genes)))
    (artifacts_dir / "selected_genes_pairwise_close_class.txt").write_text("\n".join(map(str, pairwise_genes)))

    source_df = pd.DataFrame(list(final_records))
    if source_df.empty:
        source_df = pd.DataFrame(
            columns=[
                "gene",
                "source",
                "contrast",
                "direction",
                "passed_threshold",
                "selection_pool",
                "score",
                "logFC",
                "P.Value",
                "adj.P.Val",
            ]
        )
    source_df.to_csv(artifacts_dir / "selected_gene_sources.csv", index=False)

    summary_df = pd.DataFrame(list(selection_summary))
    if summary_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "contrast",
                "source",
                "n_total_genes",
                "n_passed_threshold",
                "selection_pool",
                "n_selected",
                "n_selected_positive",
                "n_selected_negative",
                "min_genes_per_class",
                "max_genes_per_class",
            ]
        )
    summary_df.to_csv(artifacts_dir / "gene_selection_summary.csv", index=False)


def select_marker_genes(
    de_results: pd.DataFrame,
    K: int = 100,
    logfc_thresh: float = 1.0,
    fdr_thresh: float = 0.05,
    n_classes: int = 2,
    surface_markers: bool = False,
    min_surface_markers: int = 20,
    pairwise_de_results: Optional[pd.DataFrame] = None,
    one_vs_rest_fraction: float = 0.5,
    close_pair_fraction: float = 0.5,
    min_genes_per_class: int = 20,
    max_genes_per_class: int = 75,
    artifacts_dir: Optional[Path] = None,
    close_pairs: Optional[Sequence[Tuple[str, str, float]]] = None,
) -> GeneSetSelectionSummary:
    if K <= 0:
        _write_selection_artifacts(
            artifacts_dir=artifacts_dir,
            final_records=[],
            one_vs_rest_genes=[],
            pairwise_genes=[],
            selection_summary=[],
        )
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
            target_k=K,
            selection_summary=[],
        )

    ovr_df = _prepare_de_frame(
        de_results,
        source="one_vs_rest",
        fdr_thresh=fdr_thresh,
        logfc_thresh=logfc_thresh,
        surface_markers=surface_markers,
        min_surface_markers=min_surface_markers,
    )
    pair_df = _prepare_de_frame(
        pairwise_de_results,
        source="pairwise_close_class",
        fdr_thresh=fdr_thresh,
        logfc_thresh=logfc_thresh,
        surface_markers=surface_markers,
        min_surface_markers=min_surface_markers,
    )

    frac_total = float(one_vs_rest_fraction) + float(close_pair_fraction)
    if frac_total <= 0:
        one_budget = K
    else:
        one_budget = int(round(K * (float(one_vs_rest_fraction) / frac_total)))
    pair_budget = max(K - one_budget, 0)
    if pair_df.empty:
        one_budget = K
        pair_budget = 0

    ovr_ranked, ovr_summaries = _ranked_directional_records(
        ovr_df,
        min_genes_per_class=min_genes_per_class,
    )
    pair_ranked, pair_summaries = _ranked_directional_records(
        pair_df,
        min_genes_per_class=min_genes_per_class,
    )

    one_records = _select_balanced(
        ovr_ranked,
        budget=one_budget,
        max_genes_per_class=max_genes_per_class,
    )
    pair_records = _select_balanced(
        pair_ranked,
        budget=pair_budget,
        existing_genes={str(r["gene"]) for r in one_records},
        existing_records=one_records,
        max_genes_per_class=max_genes_per_class,
    )

    final_records = _dedupe_records([*one_records, *pair_records], K)
    if len(final_records) < K:
        fill_records = _select_balanced(
            ovr_ranked,
            budget=K - len(final_records),
            existing_genes={str(r["gene"]) for r in final_records},
            existing_records=final_records,
            max_genes_per_class=max_genes_per_class,
        )
        final_records = _dedupe_records([*final_records, *fill_records], K)
    if len(final_records) < K and pair_ranked:
        fill_records = _select_balanced(
            pair_ranked,
            budget=K - len(final_records),
            existing_genes={str(r["gene"]) for r in final_records},
            existing_records=final_records,
            max_genes_per_class=max_genes_per_class,
        )
        final_records = _dedupe_records([*final_records, *fill_records], K)

    combined = [str(record["gene"]) for record in final_records]
    one_vs_rest_genes = [
        str(record["gene"])
        for record in final_records
        if str(record.get("source", "")) == "one_vs_rest"
    ]
    pairwise_genes = [
        str(record["gene"])
        for record in final_records
        if str(record.get("source", "")) == "pairwise_close_class"
    ]
    selection_summary = _finalize_selection_summary(
        [*ovr_summaries, *pair_summaries],
        final_records,
        max_genes_per_class=max_genes_per_class,
    )

    _write_selection_artifacts(
        artifacts_dir=artifacts_dir,
        final_records=final_records,
        one_vs_rest_genes=one_vs_rest_genes,
        pairwise_genes=pairwise_genes,
        selection_summary=selection_summary,
    )

    contrast_order = ovr_df["Contrast"].dropna().astype(str).drop_duplicates().tolist() if not ovr_df.empty else []
    by_contrast = _records_by_contrast([r for r in final_records if r.get("source") == "one_vs_rest"])

    return GeneSetSelectionSummary(
        K=K,
        n_classes=n_classes,
        fdr_thresh=fdr_thresh,
        logfc_thresh=logfc_thresh,
        k_pan=0,
        k_per_contrast=0,
        pan_genes=[],
        by_contrast=by_contrast,
        combined=combined,
        contrast_order=contrast_order,
        target_k=K,
        one_vs_rest_budget=one_budget,
        close_pair_budget=pair_budget,
        one_vs_rest_genes=one_vs_rest_genes,
        pairwise_genes=pairwise_genes,
        close_pairs=list(close_pairs or []),
        selection_summary=selection_summary,
    )
