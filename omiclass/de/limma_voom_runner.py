import pandas as pd
import numpy as np
import json
import os
import rpy2.robjects as ro
from typing import Optional, Sequence, Union, Tuple
from pathlib import Path
import tempfile, shutil
from omiclass.utils.batch import _assert_unique, _align_samples, _write_csv 
from omiclass.plotting.volcano_plot import plot_volcano_from_limma
from omiclass.utils.io import load_matrix
from omiclass.de.gene_selection import compute_target_gene_count, select_marker_genes
from omiclass.errors.errors import RBackendError, GCError
from omiclass.reporting.de_report import write_de_geneset_report_html
from omiclass.utils.debug import debug_volcano_plot_from_limma
try:
    # rpy2 >= 3.5
    from rpy2.rinterface_lib.embedded import RRuntimeError
except Exception:
    # rpy2 <= 3.4 fallback
    from rpy2.rinterface import RRuntimeError
class LimmaVoomRunner:
    def __init__(self, de_cfg, cfg):
        self.r_loaded = False
        self.r_voom_func = None
        self.de_cfg = de_cfg
        rt_cfg = cfg.runtime
        self.cfg = cfg
        self.rt_cfg = rt_cfg
        
    @staticmethod
    def _generate_automated_contrasts(meta_df: pd.DataFrame, design_col: str) -> list[str]:
        if design_col not in meta_df.columns:
            return []
            
        levels = sorted(meta_df[design_col].dropna().astype(str).unique().tolist())
        N = len(levels)
        if N < 2:
            return []
        
        coef_names = {}
        # check if "lvl" is number 
        if all(lvl.isdigit() for lvl in levels):
            coef_names = {lvl: f"{design_col} {lvl}" for lvl in levels}
        else:
            coef_names = {lvl: f"{lvl}" for lvl in levels}

        contrast_list: list[str] = []

        if N == 2:
            contrast_list.append(f"{coef_names[levels[0]]} - {coef_names[levels[1]]}")
        else:
            for target_level in levels:
                other_levels = [lvl for lvl in levels if lvl != target_level]
                n_others = len(other_levels)
                others_part = " + ".join(coef_names[l] for l in other_levels)
                contrast_string = (
                    f"{coef_names[target_level]} - ({others_part})"
                )
                contrast_list.append(contrast_string)

        return contrast_list

    @staticmethod
    def _compute_close_class_pairs(
            expr_df: pd.DataFrame,
            meta_df: pd.DataFrame,
            design_col: str,
            *,
            pca_components: int,
            pair_count: Optional[int],
    ) -> tuple[pd.DataFrame, list[dict], list[tuple[str, str, float]]]:
        if expr_df.empty or design_col not in meta_df.columns:
            return pd.DataFrame(), [], []

        y = meta_df[design_col].astype(str)
        classes = sorted(y.dropna().unique().tolist())
        if len(classes) < 2:
            return pd.DataFrame(), [], []

        X = expr_df.T.loc[meta_df.index]
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        values = X.to_numpy(dtype=float)
        if values.size == 0:
            return pd.DataFrame(), [], []

        if np.nanmin(values) >= 0:
            values = np.log1p(values)
        values = values - values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        values = values / std

        max_components = min(int(pca_components), values.shape[0] - 1, values.shape[1])
        if max_components <= 0:
            scores = values
        else:
            u, s, _ = np.linalg.svd(values, full_matrices=False)
            scores = u[:, :max_components] * s[:max_components]

        score_df = pd.DataFrame(scores, index=X.index)
        centroids = score_df.groupby(y).mean()
        distances = pd.DataFrame(0.0, index=centroids.index, columns=centroids.index)
        pairs: list[tuple[str, str, float]] = []

        for i, class_a in enumerate(classes):
            for class_b in classes[i + 1:]:
                if class_a not in centroids.index or class_b not in centroids.index:
                    continue
                dist = float(np.linalg.norm(centroids.loc[class_a].to_numpy() - centroids.loc[class_b].to_numpy()))
                distances.loc[class_a, class_b] = dist
                distances.loc[class_b, class_a] = dist
                pairs.append((class_a, class_b, dist))

        pairs = sorted(pairs, key=lambda x: x[2])
        n_pairs = pair_count if pair_count is not None else max(1, len(classes))
        selected = pairs[: min(int(n_pairs), len(pairs))]
        pair_dicts = [
            {"class_a": class_a, "class_b": class_b}
            for class_a, class_b, _ in selected
        ]
        return distances, pair_dicts, selected


    def run_limma_voom(self, 
                       raw_counts_df: pd.DataFrame, 
                       meta_df: pd.DataFrame, 
                       output_dir: Union[str, Path], 
                       design_col: str, 
                       covariates: Optional[Sequence[str]] = None, 
                       out_csv: str = "limma_voom_de_results.csv",
                       r_script_path: Optional[Union[str, Path]] = None, 
                       min_count: float = 20, 
                       min_samples_frac: float = 0.5, 
                       gene_list: Optional[Sequence[str]] = None,
                       surface_markers: bool = False,
            ) -> dict:
        return self._run_limma_backend(
            expr_df=raw_counts_df,
            meta_df=meta_df,
            output_dir=output_dir,
            design_col=design_col,
            covariates=covariates,
            out_csv=out_csv,
            r_script_path=r_script_path,
            min_count=min_count,
            min_samples_frac=min_samples_frac,
            gene_list=gene_list,
            surface_markers=surface_markers,
            backend="voom",
            analysis_label="Differential Expression",
            feature_label_singular="gene",
            feature_label_plural="genes",
        )

    def run_limma(
            self,
            expr_df: pd.DataFrame,
            meta_df: pd.DataFrame,
            output_dir: Union[str, Path],
            design_col: str,
            covariates: Optional[Sequence[str]] = None,
            out_csv: str = "limma_de_results.csv",
            r_script_path: Optional[Union[str, Path]] = None,
            min_count: float = 0,
            min_samples_frac: float = 0.0,
            gene_list: Optional[Sequence[str]] = None,
            surface_markers: bool = False,
            analysis_label: str = "Differential Protein Abundance",
            feature_label_singular: str = "protein",
            feature_label_plural: str = "proteins",
    ) -> dict:
        return self._run_limma_backend(
            expr_df=expr_df,
            meta_df=meta_df,
            output_dir=output_dir,
            design_col=design_col,
            covariates=covariates,
            out_csv=out_csv,
            r_script_path=r_script_path,
            min_count=min_count,
            min_samples_frac=min_samples_frac,
            gene_list=gene_list,
            surface_markers=surface_markers,
            backend="limma",
            analysis_label=analysis_label,
            feature_label_singular=feature_label_singular,
            feature_label_plural=feature_label_plural,
        )

    def _run_limma_backend(
            self,
            expr_df: pd.DataFrame,
            meta_df: pd.DataFrame,
            output_dir: Union[str, Path],
            design_col: str,
            covariates: Optional[Sequence[str]],
            out_csv: str,
            r_script_path: Optional[Union[str, Path]],
            min_count: float,
            min_samples_frac: float,
            gene_list: Optional[Sequence[str]],
            surface_markers: bool,
            backend: str,
            analysis_label: str,
            feature_label_singular: str,
            feature_label_plural: str,
    ) -> dict:
        min_count = self.rt_cfg.min_features if self.rt_cfg.min_features is not None else min_count
        min_samples_frac = self.de_cfg.min_samples_frac if self.de_cfg.min_samples_frac is not None else min_samples_frac

        
        _assert_unique(expr_df, "expr_df")
        _assert_unique(meta_df, "meta_df")

        X, M = _align_samples(expr_df, meta_df, required_meta_cols=[design_col] + (covariates if covariates else []))
        
        if design_col not in M.columns:
            raise ValueError(f"design_col '{design_col}' not in meta_df")
        if covariates:
            missing = [c for c in covariates if c not in M.columns]
            if missing:
                raise ValueError(f"Covariates missing in meta_df: {missing}")
        
        output_dir = Path(output_dir)
        base_output_dir = output_dir
        output_dir = output_dir / "de_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        if r_script_path is None:
            r_script_path = Path(__file__).resolve().parent / "limma_voom_de.R"
        else:
            r_script_path = Path(r_script_path)
        
        temp_dir = Path(tempfile.mkdtemp())
                
        try:
            expr_filepath = temp_dir / "counts_data.csv"
            meta_filepath = temp_dir / "meta_data.csv"
            json_filepath = temp_dir / "config.json"
            output_filepath = output_dir / out_csv

            _write_csv(X, expr_filepath)
            _write_csv(M, meta_filepath)
            
            design_cols = [design_col]
            if covariates:
                design_cols.extend(covariates)
            
            design_str = " + ".join(design_cols)
            
            config_data = {
                "expr_filepath": str(expr_filepath),
                "meta_filepath": str(meta_filepath),
                "output_filepath": str(output_filepath),
                "design_str": design_str,
                "design_col": design_col,
                "min_count": min_count,
                "min_samples_frac": min_samples_frac,
                "fdr_threshold": self.de_cfg.fdr_thresh if self.de_cfg.fdr_thresh is not None else 0.05,
                "logfc_threshold": self.de_cfg.logfc_thresh if self.de_cfg.logfc_thresh is not None else 1.0,
                "contrast_mode": "one_vs_rest",
            }
            with open(json_filepath, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            if not self.r_loaded:
                try:
                    ro.r.source(str(r_script_path))
                    if backend == "voom":
                        self.r_voom_func = ro.globalenv['execute_limma_voom_de']
                    else:
                        self.r_voom_func = ro.globalenv['execute_limma_de']
                    self.r_loaded = True
                except RRuntimeError as e:
                    raise RBackendError(
                        f"Failed to load R script '{r_script_path}': {e}"
                    ) from e
            
            try:
                r_result_str = self.r_voom_func(str(json_filepath))[0]
            except RRuntimeError as e:
                raise RBackendError(
                    "R function execute_limma_voom_de crashed: " + str(e)
                ) from e
            
            r_result_dict = json.loads(r_result_str)
            if r_result_dict.get('status') == 'error':
                raise RBackendError(
                    f"R execution reported error: {r_result_dict.get('message')}"
                )
            
            df = pd.read_csv(output_filepath)
            df.set_index('gene_id', inplace=True)
            plot_output_filepath = output_dir
            plot_output_filepath.mkdir(parents=True, exist_ok=True)
            summary = None

            if self.de_cfg.geneset_selection:
                df = pd.read_csv(output_filepath)
                df.set_index('gene_id', inplace=True)
                gs_cfg = self.cfg.de_gene_selection
                target_k = compute_target_gene_count(
                    n_classes=M[design_col].nunique(),
                    min_genes=gs_cfg.min_genes,
                    max_genes_per_class=gs_cfg.max_genes_per_class,
                    max_genes=gs_cfg.max_genes,
                )
                frac_total = float(gs_cfg.one_vs_rest_fraction) + float(gs_cfg.close_pair_fraction)
                close_pair_budget = 0 if frac_total <= 0 else int(round(target_k * (float(gs_cfg.close_pair_fraction) / frac_total)))

                train_artifacts_dir = base_output_dir / "train_artifacts"
                train_artifacts_dir.mkdir(parents=True, exist_ok=True)

                distance_matrix, pair_dicts, close_pairs = self._compute_close_class_pairs(
                    X,
                    M,
                    design_col,
                    pca_components=gs_cfg.pca_components,
                    pair_count=gs_cfg.close_pair_count,
                )
                if distance_matrix.empty:
                    pd.DataFrame().to_csv(train_artifacts_dir / "class_distance_matrix.csv")
                else:
                    distance_matrix.to_csv(train_artifacts_dir / "class_distance_matrix.csv")

                pairs_df = pd.DataFrame(
                    [
                        {"class_a": a, "class_b": b, "distance": d, "rank": i + 1}
                        for i, (a, b, d) in enumerate(close_pairs)
                    ],
                    columns=["class_a", "class_b", "distance", "rank"],
                )
                pairs_df.to_csv(train_artifacts_dir / "selected_close_class_pairs.csv", index=False)

                pairwise_df = pd.DataFrame()
                if close_pair_budget > 0 and pair_dicts:
                    pairwise_output_filepath = output_dir / out_csv.replace(".csv", "_pairwise_close_class.csv")
                    pairwise_config = dict(config_data)
                    pairwise_config.update(
                        {
                            "output_filepath": str(pairwise_output_filepath),
                            "contrast_mode": "pairwise",
                            "pairwise_pairs": pair_dicts,
                        }
                    )
                    pairwise_json_filepath = temp_dir / "pairwise_config.json"
                    with open(pairwise_json_filepath, 'w') as f:
                        json.dump(pairwise_config, f, indent=4)
                    try:
                        pairwise_result_str = self.r_voom_func(str(pairwise_json_filepath))[0]
                    except RRuntimeError as e:
                        raise RBackendError(
                            "R function crashed during close-class pairwise DE: " + str(e)
                        ) from e
                    pairwise_result_dict = json.loads(pairwise_result_str)
                    if pairwise_result_dict.get('status') == 'error':
                        raise RBackendError(
                            f"Pairwise R execution reported error: {pairwise_result_dict.get('message')}"
                        )
                    pairwise_df = pd.read_csv(pairwise_output_filepath)
                    pairwise_df.set_index('gene_id', inplace=True)

                summary = select_marker_genes(
                    df,
                    K=target_k,
                    logfc_thresh=self.de_cfg.logfc_thresh if self.de_cfg.logfc_thresh is not None else 1.0,
                    fdr_thresh=self.de_cfg.fdr_thresh if self.de_cfg.fdr_thresh is not None else 0.05,
                    n_classes=M[design_col].nunique(),
                    surface_markers=surface_markers,
                    min_surface_markers=self.rt_cfg.min_features if self.rt_cfg.min_features is not None else 20,
                    pairwise_de_results=pairwise_df,
                    one_vs_rest_fraction=gs_cfg.one_vs_rest_fraction,
                    close_pair_fraction=gs_cfg.close_pair_fraction,
                    min_genes_per_class=gs_cfg.min_genes_per_class,
                    max_genes_per_class=gs_cfg.max_genes_per_class,
                    artifacts_dir=train_artifacts_dir,
                    close_pairs=close_pairs,
                )
                gene_list = summary.combined
                feature_list_path = output_dir / "limma_selected_features.txt"
                feature_list_path.write_text("\n".join(gene_list))
                # Backward-compatible RNA-era artifact name
                gene_list_path = output_dir / "limma_voom_selected_genes.txt"
                gene_list_path.write_text("\n".join(gene_list))

            volcano_svgs = []
            fdr_thresh = self.de_cfg.fdr_thresh if self.de_cfg.fdr_thresh is not None else 0.05
            logfc_thresh = self.de_cfg.logfc_thresh if self.de_cfg.logfc_thresh is not None else 1.0
            report_k = len(gene_list) if gene_list is not None else (
                compute_target_gene_count(
                    n_classes=M[design_col].nunique(),
                    min_genes=self.cfg.de_gene_selection.min_genes,
                    max_genes_per_class=self.cfg.de_gene_selection.max_genes_per_class,
                    max_genes=self.cfg.de_gene_selection.max_genes,
                )
            )

            # if Contrast came from R, it should exist
            if "Contrast" in df.columns.tolist():
                df2 = df.dropna(subset=["Contrast"])
                for contrast_name, sub in df2.groupby("Contrast"):
                    # write a temp CSV for this contrast
                    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(contrast_name))
                    safe = safe.split("_")[0] + "_vs_all"
                    sub_path = plot_output_filepath / f"limma_voom_de_{safe}.csv"
                    sub.to_csv(sub_path, index=True)

                    svg = plot_volcano_from_limma(
                        de_results=sub_path,
                        fdr_thresh=fdr_thresh,
                        logfc_thresh=logfc_thresh,
                        title=f"{analysis_label}: {contrast_name}",
                        markers=gene_list if gene_list is not None else [],
                    )
                    # remove temp sub csv and plot directory
                    if sub_path.exists():
                        os.remove(sub_path)

                    names = None
                    project_cfg = getattr(self.cfg, "project", None)
                    if project_cfg is not None:
                        names = getattr(project_cfg, "classes_names", None)
                    try: 
                        if " vs " in str(contrast_name):
                            fix_name = str(contrast_name)
                        else:
                            fix_name = f"{names[(int(contrast_name)-1)]} vs all" if names is not None and int(contrast_name) -1 < len(names) else f"Class {contrast_name}"
                    except Exception:
                        fix_name = f"Class {contrast_name} vs all"
                    volcano_svgs.append((f"{fix_name}", svg))
            else:
                # fallback: original single volcano
                svg = plot_volcano_from_limma(
                    de_results=output_filepath,
                    fdr_thresh=fdr_thresh,
                    logfc_thresh=logfc_thresh,
                    title=f"{analysis_label} Results",
                    markers=gene_list if gene_list is not None else [],
                )
                volcano_svgs.append(("Volcano", svg))

            write_de_geneset_report_html(
                output_dir=base_output_dir,
                de_results_csv=output_filepath,
                volcano_svgs=volcano_svgs,
                design_col=design_col,
                n_classes=M[design_col].nunique(),
                K=report_k,
                fdr_thresh=fdr_thresh,
                logfc_thresh=logfc_thresh,
                min_count=min_count,
                min_samples_frac=min_samples_frac,
                geneset_summary=summary.__dict__ if summary is not None else None,
                combined_genes=gene_list,
                covariates=covariates,
                cfg=self.cfg,
                analysis_label=analysis_label,
                feature_label_singular=feature_label_singular,
                feature_label_plural=feature_label_plural,
            )


            # return gene_list in r_result_dict
            r_result_dict['selected_genes'] = gene_list
            r_result_dict['selected_features'] = gene_list

            return r_result_dict
    
        except Exception as e:
            raise GCError(f"LimmaRunner failed: {e}") from e
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
