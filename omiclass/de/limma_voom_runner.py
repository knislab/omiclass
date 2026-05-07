import pandas as pd
import numpy as np
import json
import os
import rpy2.robjects as ro
from typing import Optional, Sequence, Union, Tuple
from pathlib import Path
import tempfile, shutil
from generic_classifier.utils.batch import _assert_unique, _align_samples, _write_csv 
from generic_classifier.plotting.volcano_plot import plot_volcano_from_limma
from generic_classifier.utils.io import load_matrix
from generic_classifier.de.gene_selection import select_marker_genes
from generic_classifier.errors.errors import RBackendError, GCError
from generic_classifier.reporting.de_report import write_de_geneset_report_html
from generic_classifier.utils.debug import debug_volcano_plot_from_limma
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


    def run_limma_voom(self, 
                       raw_counts_df: pd.DataFrame, 
                       meta_df: pd.DataFrame, 
                       output_dir: Union[str, Path], 
                       design_col: str, 
                       covariates: Optional[Sequence[str]] = None, 
                       out_csv: str = "limma_voom_de_results.csv",
                       r_script_path: Union[str, Path] = Path("generic_classifier/de/limma_voom_de.R"), 
                       min_count: float = 20, 
                       min_samples_frac: float = 0.5, 
                       gene_list: Optional[Sequence[str]] = None,
                       surface_markers: bool = False,
            ) -> dict:
        min_count = self.rt_cfg.min_features if self.rt_cfg.min_features is not None else min_count
        min_samples_frac = self.de_cfg.min_samples_frac if self.de_cfg.min_samples_frac is not None else min_samples_frac

        
        _assert_unique(raw_counts_df, "raw_counts_df")
        _assert_unique(meta_df, "meta_df")

        X, M = _align_samples(raw_counts_df, meta_df, required_meta_cols=[design_col] + (covariates if covariates else []))
        
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
            }
            with open(json_filepath, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            if not self.r_loaded:
                try:
                    ro.r.source(str(r_script_path))
                    self.r_voom_func = ro.globalenv['execute_limma_voom_de']
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

            if self.de_cfg.geneset_selection:
                df = pd.read_csv(output_filepath)
                df.set_index('gene_id', inplace=True)
                summary = select_marker_genes(
                    df,
                    K=self.de_cfg.k_genes if self.de_cfg.k_genes else 100,
                    logfc_thresh=self.de_cfg.logfc_thresh if self.de_cfg.logfc_thresh is not None else 1.0,
                    fdr_thresh=self.de_cfg.fdr_thresh if self.de_cfg.fdr_thresh is not None else 0.05,
                    n_classes=M[design_col].nunique(),
                    surface_markers=surface_markers,
                    min_surface_markers=self.rt_cfg.min_features if self.rt_cfg.min_features is not None else 20,
                )
                gene_list = summary.combined
                # ssave gene list text
                gene_list_path = output_dir / "limma_voom_selected_genes.txt"
                gene_list_path.write_text("\n".join(gene_list))

            volcano_svgs = []

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
                        fdr_thresh=0.05,
                        logfc_thresh=1.0,
                        title=f"Limma-Voom DE: {contrast_name}",
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
                        fix_name = f"{names[(int(contrast_name)-1)]} vs all" if names is not None and int(contrast_name) -1 < len(names) else f"Class {contrast_name}"
                    except Exception:
                        fix_name = f"Class {contrast_name} vs all"
                    volcano_svgs.append((f"{fix_name}", svg))
            else:
                # fallback: original single volcano
                svg = plot_volcano_from_limma(
                    de_results=output_filepath,
                    fdr_thresh=0.05,
                    logfc_thresh=1.0,
                    title="Limma-Voom DE Results",
                    markers=gene_list if gene_list is not None else [],
                )
                volcano_svgs.append(("Volcano", svg))

            write_de_geneset_report_html(
                output_dir=base_output_dir,
                de_results_csv=output_filepath,
                volcano_svgs=volcano_svgs,
                design_col=design_col,
                n_classes=M[design_col].nunique(),
                K=self.de_cfg.k_genes if self.de_cfg.k_genes else 100,
                fdr_thresh=self.de_cfg.fdr_thresh if self.de_cfg.fdr_thresh is not None else 0.05,
                logfc_thresh=self.de_cfg.logfc_thresh if self.de_cfg.logfc_thresh is not None else 1.0,
                min_count=min_count,
                min_samples_frac=min_samples_frac,
                geneset_summary=summary.__dict__,
                combined_genes=gene_list,
                covariates=covariates,
                cfg=self.cfg,
            )


            # return gene_list in r_result_dict
            r_result_dict['selected_genes'] = gene_list

            return r_result_dict
    
        except Exception as e:
            raise GCError(f"LimmaRunner failed: {e}") from e
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)