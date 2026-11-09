from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from omiclass.config.schema import Config
from omiclass.utils.io import load_matrix, load_labels, load_list, load_meta_matrix
from omiclass.utils.checks import check_and_prepare_xy, ensure_dense_finite
from omiclass.utils.surface_markers import fillter_surface_markers
from omiclass.errors.errors import DataError, DataShapeError


@dataclass
class ExpressionBundle:
    """Expression + gene list before / after batch-correction."""
    X: pd.DataFrame  
    genes: List[str]


class DataProcessor:
    """
    Centralized data loading + validation.

    Responsibilities:
    - Load expression matrix and gene list robustly.
    - Optionally filter genes (e.g., surface markers).
    - Validate and load batch metadata (meta_df, cov_matrix, batch hold-out IDs).
    - Turn a limma-corrected matrix + gene list into (X, y) ready for the ML pipeline.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def clean_meta_from_column(
        self,
        meta_df: pd.DataFrame,
        drop_columns: List[str],
    ) -> pd.DataFrame:
        """
        Clean meta_df by dropping specified columns.
        """
        cleaned_meta = meta_df.drop(columns=drop_columns, errors='ignore')
        cleaned_meta.attrs.update(meta_df.attrs)
        return cleaned_meta

    # 1. Expression + gene list
    def load_expression_and_genes(
        self,
        expression_path: Path,
        gene_path: Optional[Path] = None,
        *,
        surface_markers: bool = False,
    ) -> ExpressionBundle:
        """
        Load expression matrix and gene list, sanitize dtype/NaNs/inf,
        and optionally filter genes to surface markers.
        """
        expression_path = Path(expression_path)
        if not expression_path.exists():
            raise DataError(f"Expression file not found: {expression_path}")

        X = load_matrix(expression_path)
        if X.empty:
            raise DataError(f"Expression matrix is empty: {expression_path}")

        # Keep only numeric, cast to float32, clean infinities/NaNs
        X = X.select_dtypes(include=[np.number]).astype("float32")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Gene list: either from file or from index
        genes: List[str]
        if gene_path is not None:
            gene_path = Path(gene_path)
            if not gene_path.exists():
                raise DataError(f"Gene list file not found: {gene_path}")
            genes = load_list(gene_path)
        else:
            # assume genes are the matrix index (standard for your pipeline)
            genes = list(X.index)

        if not genes:
            raise DataError("Gene list is empty after loading.")

        if surface_markers:
            genes = fillter_surface_markers(genes)
            if not genes:
                raise DataError(
                    "After filtering for surface markers, gene list became empty."
                )

        return ExpressionBundle(X=X, genes=genes)

    def load_modality_input(
        self,
        expression_path: Path,
        gene_path: Optional[Path] = None,
        *,
        surface_markers: bool = False,
    ) -> ExpressionBundle:
        """
        Modality-aware input loader.
        - rnaseq: preserve current behavior exactly.
        - cptac_ms: load protein abundance matrix (features x samples) with generic feature IDs.
        """
        modality = getattr(getattr(self.cfg, "project", None), "modality", "rnaseq")

        if modality == "rnaseq":
            return self.load_expression_and_genes(
                expression_path=expression_path,
                gene_path=gene_path,
                surface_markers=surface_markers,
            )

        if modality == "cptac_ms":
            expression_path = Path(expression_path)
            if not expression_path.exists():
                raise DataError(f"Expression file not found: {expression_path}")
            X = load_matrix(expression_path)
            if X.empty:
                raise DataError(f"Expression matrix is empty: {expression_path}")
            X = X.select_dtypes(include=[np.number]).astype("float32")
            X = X.replace([np.inf, -np.inf], np.nan)

            genes: List[str]
            if gene_path is not None:
                gene_path = Path(gene_path)
                if not gene_path.exists():
                    raise DataError(f"Gene list file not found: {gene_path}")
                genes = load_list(gene_path)
            else:
                genes = list(X.index)
            if not genes:
                raise DataError("Feature list is empty after loading CPTAC matrix.")

            bundle = ExpressionBundle(X=X, genes=genes)
            # Keep identifiers fully generic for proteomics workflows.
            bundle.X.index = bundle.X.index.map(str)
            bundle.genes = [str(g) for g in bundle.genes]
            return bundle

        raise DataError(
            f"Unsupported project.modality '{modality}'. Supported modalities: rnaseq, cptac_ms."
        )

    def preprocess_cptac_ms(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        CPTAC proteomics preprocessing (features x samples):
        missingness filtering -> optional log2 transform -> optional imputation.
        """
        cfg = getattr(self.cfg, "cptac_ms_preprocessing", None)
        if cfg is None:
            return X, {"modality": "cptac_ms", "preprocessing": "none"}

        out = X.copy()
        summary = {
            "modality": "cptac_ms",
            "input_shape": list(out.shape),
            "max_missing_fraction": cfg.max_missing_fraction,
            "log2_transform": bool(cfg.log2_transform),
            "imputation": str(cfg.imputation),
        }

        if cfg.max_missing_fraction is not None:
            row_missing_frac = out.isna().mean(axis=1)
            keep = row_missing_frac <= float(cfg.max_missing_fraction)
            removed = int((~keep).sum())
            out = out.loc[keep]
            summary["removed_features_by_missingness"] = removed

        if bool(cfg.log2_transform):
            valid = out.to_numpy(dtype=float)
            finite = valid[np.isfinite(valid)]
            if finite.size > 0:
                min_val = float(np.nanmin(finite))
                if min_val <= 0:
                    offset = abs(min_val) + float(cfg.pseudocount)
                else:
                    offset = 0.0
                out = np.log2(out + offset)
                summary["log2_offset_added"] = float(offset)
            else:
                summary["log2_offset_added"] = 0.0

        if str(cfg.imputation) == "median_by_feature":
            med = out.median(axis=1, skipna=True)
            out = out.T.fillna(med).T
            summary["imputed_missing_values"] = True
        else:
            summary["imputed_missing_values"] = False

        summary["output_shape"] = list(out.shape)
        return out, summary
    
    def load_batch_metadata_prediction(self,
        X_expr: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[list[str]]]:
        bc = self.cfg.batch_correction
        meta_path_cfg = self.cfg.project
        cluster_column = self.cfg.project.cluster_column
        survival_cfg = getattr(self.cfg.runtime, "survival_prediction", None)
        has_time_col = (
            survival_cfg is not None
            and hasattr(survival_cfg, "time_col")
            and survival_cfg.time_col
        )
        has_event_col = (
            survival_cfg is not None
            and hasattr(survival_cfg, "event_col")
            and survival_cfg.event_col
        )

        # Metadata
        meta_df: Optional[pd.DataFrame] = None
        if getattr(meta_path_cfg, "meta_path", None):
            meta_path = Path(meta_path_cfg.meta_path)
            if not meta_path.exists():
                raise DataError(f"batch_correction.meta_path does not exist: {meta_path}")
            meta_df, category_maps = load_meta_matrix(meta_path, return_category_maps=True)
            meta_df.attrs["category_maps"] = category_maps
            if meta_df.empty:
                raise DataError(f"Meta file is empty: {meta_path}")

            # Check intersection with expression samples
            overlap = set(X_expr.columns) & set(meta_df.index)
            if not overlap:
                raise DataError(
                    "No overlapping sample IDs between expression matrix and meta_df "
                    f"(meta_path={meta_path})."
                )

        # Covariate matrix
        cov_matrix: Optional[pd.DataFrame] = None
        if getattr(bc, "covariates_path", None):
            cov_path = Path(bc.covariates_path)
            if not cov_path.exists():
                raise DataError(
                    f"batch_correction.covariates_path does not exist: {cov_path}"
                )
            cov_matrix = load_matrix(cov_path)
            if cov_matrix.empty:
                raise DataError(f"Covariate matrix is empty: {cov_path}")

            # Intersection with expression samples
            overlap = set(X_expr.columns) & set(cov_matrix.index)
            if not overlap:
                raise DataError(
                    "No overlapping sample IDs between expression matrix and cov_matrix "
                    f"(covariates_path={cov_path})."
                )
            
        labels = meta_df[cluster_column]

        drop_columns = [cluster_column] + ([survival_cfg.time_col] if has_time_col else []) + ([survival_cfg.event_col] if has_event_col else [])
        meta_df = self.clean_meta_from_column(
            meta_df,
            drop_columns=drop_columns
        ) if meta_df is not None else None

        return meta_df, cov_matrix, labels

    # 2. Batch metadata / covariates / batch hold-out IDs
    def load_batch_metadata(
        self,
        X_expr: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[list[str]]]:
        """
        Load and minimally validate batch-related metadata:

        - meta_df (if cfg.batch_correction.meta_path is set)
        - cov_matrix (if cfg.batch_correction.covariates_path is set)
        - batch_hold_out_ids (if cfg.batch_correction.hold_out_path exists)

        Does *not* run limma; just ensures files exist and sample IDs intersect.
        """
        bc = self.cfg.batch_correction
        meta_path_cfg = self.cfg.project
        cluster_column = self.cfg.project.cluster_column
        survival_cfg = getattr(self.cfg.runtime, "survival_prediction", None)
        has_time_col = (
            survival_cfg is not None
            and hasattr(survival_cfg, "time_col")
            and survival_cfg.time_col
        )
        has_event_col = (
            survival_cfg is not None
            and hasattr(survival_cfg, "event_col")
            and survival_cfg.event_col
        )

        # Metadata
        meta_df: Optional[pd.DataFrame] = None
        if getattr(meta_path_cfg, "meta_path", None):
            meta_path = Path(meta_path_cfg.meta_path)
            if not meta_path.exists():
                raise DataError(f"batch_correction.meta_path does not exist: {meta_path}")
            meta_df, category_maps = load_meta_matrix(meta_path, return_category_maps=True)
            meta_df.attrs["category_maps"] = category_maps
            if meta_df.empty:
                raise DataError(f"Meta file is empty: {meta_path}")

            # Check intersection with expression samples
            overlap = set(X_expr.columns) & set(meta_df.index)
            if not overlap:
                raise DataError(
                    "No overlapping sample IDs between expression matrix and meta_df "
                    f"(meta_path={meta_path})."
                )

        # Covariate matrix
        cov_matrix: Optional[pd.DataFrame] = None
        if getattr(bc, "covariates_path", None):
            cov_path = Path(bc.covariates_path)
            if not cov_path.exists():
                raise DataError(
                    f"batch_correction.covariates_path does not exist: {cov_path}"
                )
            cov_matrix = load_matrix(cov_path)
            if cov_matrix.empty:
                raise DataError(f"Covariate matrix is empty: {cov_path}")

            # Check overlap with expression samples
            overlap = set(X_expr.columns) & set(cov_matrix.index)
            if not overlap:
                raise DataError(
                    "No overlapping sample IDs between expression matrix and cov_matrix "
                    f"(covariates_path={cov_path})."
                )

        drop_columns = [cluster_column] + ([survival_cfg.time_col] if has_time_col else []) + ([survival_cfg.event_col] if has_event_col else [])
        meta_df = self.clean_meta_from_column(
            meta_df,
            drop_columns=drop_columns
        ) if meta_df is not None else None

        return meta_df, cov_matrix

    # 3. After limma: turn corrected matrix + gene list into (X, y)
    def build_xy_from_corrected(
        self,
        X_full_raw: pd.DataFrame,
        gene_list: list[str],
        drop_constant_cols: bool = True,
        clean_finite: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Given a limma-corrected expression matrix (genes x samples) and a gene list,
        build X (samples x selected_genes) and y aligned labels, then run
        check_and_prepare_xy + ensure_dense_finite.
        """
        if X_full_raw.empty:
            raise DataError("Limma-corrected expression matrix is empty.")

        if not gene_list:
            raise DataError("Gene list is empty in build_xy_from_corrected().")

        missing_genes = [g for g in gene_list if g not in X_full_raw.index]
        if missing_genes:
            print(
                f"[WARN] {len(missing_genes)} genes from gene_list not found in "
                f"limma output; first few: {missing_genes[:5]}"
            )

        selected_genes = [g for g in gene_list if g in X_full_raw.index]
        if not selected_genes:
            raise DataError(
                "No genes from gene_list found in limma-corrected matrix."
            )

        # X_full_raw: genes x samples -> X_raw: samples x genes
        X_raw = X_full_raw.loc[selected_genes].T

        # Load labels aligned to X_raw.index

        meta_path = self.cfg.project.meta_path
        cluster_column = self.cfg.project.cluster_column
        y_raw, match_x_index = load_labels(meta_path, cluster_column, X_raw.index, cfg=self.cfg)
        X_raw = X_raw.loc[match_x_index]

        if len(X_raw) != len(y_raw):
            raise DataShapeError(
                f"X and y length mismatch after label loading: X={len(X_raw)}, y={len(y_raw)}"
            )

        # Use existing helpers to clean/prepare
        X_clean, y_clean = check_and_prepare_xy(
            X_raw,
            y_raw,
            na_y="drop",
            coerce_object=True,
            drop_all_nan_rows=False,
            drop_constant_cols=drop_constant_cols,
        )
        if clean_finite:
            X_clean = ensure_dense_finite(X_clean)

        if len(X_clean) == 0:
            raise DataShapeError("After cleaning, no samples remain (X is empty).")

        return X_clean, y_clean
    
    def load_de_data(
        self,
        X_expr: Path
    ) -> pd.DataFrame:
        """
        Load differential expression results from a CSV file.
        """
        label_name = "cluster"
        meta_path_cfg = self.cfg.project

        # Metadata
        meta_df: Optional[pd.DataFrame] = None
        if getattr(meta_path_cfg, "meta_path", None):
            meta_path = Path(meta_path_cfg.meta_path)
            if not meta_path.exists():
                raise DataError(f"batch_correction.meta_path does not exist: {meta_path}")
            meta_df = load_meta_matrix(meta_path)
            if meta_df.empty:
                raise DataError(f"Meta file is empty: {meta_path}")
            
            # Check intersection with expression samples
            overlap = set(X_expr.columns) & set(meta_df.index)
            if not overlap:
                raise DataError(
                    "No overlapping sample IDs between expression matrix and meta_df "
                    f"(meta_path={meta_path})."
                )
        cluster_column = self.cfg.project.cluster_column
        labels = load_labels(meta_path, cluster_column, cfg=self.cfg)

        meta_df[label_name] = labels
        survival_cfg = getattr(self.cfg.runtime, "survival_prediction", None)
        has_time_col = (
            survival_cfg is not None
            and hasattr(survival_cfg, "time_col")
            and survival_cfg.time_col
        )
        has_event_col = (
            survival_cfg is not None
            and hasattr(survival_cfg, "event_col")
            and survival_cfg.event_col
        )
        drop_columns =([survival_cfg.time_col] if has_time_col else []) + ([survival_cfg.event_col] if has_event_col else [])
        if drop_columns:
            meta_df = self.clean_meta_from_column(
                meta_df,
                drop_columns=drop_columns
            ) if meta_df is not None else None
            
        return meta_df, label_name, labels
    
    def normalize_raw_counts(
        self,
        X_counts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize raw counts to true TPM and return log1p(TPM)."""
        if getattr(self.cfg.normalization, "gene_lengths_path", None):
            gene_lengths = self.load_precomputed_gene_lengths(
                str(self.cfg.normalization.gene_lengths_path)
            )
        elif getattr(self.cfg.normalization, "gtf_path", None):
            gene_lengths = self.compute_gene_lengths_from_gtf(
                str(self.cfg.normalization.gtf_path)
            )
        else:
            raise DataError("No gene lengths source provided in configuration.")
        
        if X_counts.empty:
            raise DataError("Raw counts matrix is empty.")

        if gene_lengths is None or gene_lengths.empty:
            raise DataError("Gene lengths must be provided.")

        missing = set(X_counts.index) - set(gene_lengths.index)

        if missing:
            missing_list = list(missing)
            X_counts = X_counts.loc[X_counts.index.intersection(gene_lengths.index)]

            if X_counts.shape[0] < self.cfg.de.k_genes:
                raise DataError(
                    f"After dropping genes without gene lengths, only {X_counts.shape[0]} genes remain, "
                    f"which is less than de.k_genes={self.cfg.de.k_genes}. "
                    f"Example missing genes: {missing_list[:10]}"
                    f"X_counts.shape: {X_counts.shape}"
                )

        lengths = gene_lengths.loc[X_counts.index].astype(float)

        if (lengths <= 0).any():
            raise DataError("Gene lengths contain non-positive values.")

        X = X_counts.apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            raise DataError("Counts contain non-numeric values.")
        if (X < 0).any().any():
            raise DataError("Counts contain negative values.")
        rpk = X.divide(lengths / 1000, axis=0)
        sf = rpk.sum(axis=0)
        if (sf <= 0).any():
            raise DataError("Some samples have zero RPK total.")

        tpm = rpk.divide(sf, axis=1) * 1e6
        return np.log1p(tpm)
    

    def compute_gene_lengths_from_gtf(self, gtf_path: str) -> pd.Series:
        """Compute transcript lengths (sum of exon spans) from a GTF."""
        records = []
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if fields[2] != "exon":
                    continue
                chrom, _, _, start, end, _, _, _, attrs = fields
                start, end = int(start), int(end)

                # Extract transcript_id (fallback to gene_id)
                tid = None
                for a in attrs.split(";"):
                    a = a.strip()
                    if a.startswith("transcript_id"):
                        tid = a.split('"')[-2]
                        break
                if tid is None:
                    for a in attrs.split(";"):
                        if a.strip().startswith("gene_id"):
                            tid = a.split('"')[-2]
                            break

                if tid:
                    records.append((tid, end - start + 1))

        df = pd.DataFrame(records, columns=["id", "len"])
        lengths = df.groupby("id")["len"].sum()
        return lengths

    def load_precomputed_gene_lengths(self,path: str) -> pd.Series:
        """Load cached gene lengths (pickle, csv, or parquet)."""
        p = Path(path)
        if not p.exists():
            raise DataError(f"Gene length file not found: {path}")

        if p.suffix == ".pkl":
            return pd.read_pickle(p)
        if p.suffix == ".csv":
            df = pd.read_csv(p, index_col=0)
            return df.iloc[:, 0]
        if p.suffix == ".txt":
            df = pd.read_csv(p, index_col=0, sep="\t")
            return df.iloc[:, 0]
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            return df.set_index(df.columns[0]).iloc[:, 0]

        raise DataError(f"Unsupported file format: {p.suffix}")
