import pandas as pd
import json
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
try:
    # rpy2 >= 3.5
    from rpy2.rinterface_lib.embedded import RRuntimeError
except Exception:
    # rpy2 <= 3.4 fallback
    from rpy2.rinterface import RRuntimeError

from typing import Optional, Sequence, Union, Tuple
from pathlib import Path
import tempfile, shutil, logging
from omiclass.utils.batch import _assert_unique, _align_samples, _write_csv
from omiclass.plotting.limma_plotter import limma_plotter
from omiclass.errors.errors import RBackendError, GCError

class LimmaRunner:
    def __init__(self):
        self.r_loaded = False
        self.r_limma_func = None

    @staticmethod
    def _series_with_category_names(s: pd.Series, mapping: Optional[dict]) -> pd.Series:
        if not mapping:
            return s

        def _lookup(value):
            if pd.isna(value):
                return pd.NA
            for key in (value, str(value)):
                if key in mapping:
                    return mapping[key]
            try:
                numeric_key = int(value)
            except (TypeError, ValueError):
                return value
            return mapping.get(numeric_key, mapping.get(str(numeric_key), value))

        return s.map(_lookup)

    @staticmethod
    def _restore_category_columns(
        meta_df: Optional[pd.DataFrame],
        columns: Sequence[str],
        category_maps: Optional[dict],
    ) -> Optional[pd.DataFrame]:
        if meta_df is None or not category_maps:
            return meta_df

        out = meta_df.copy()
        for col in columns:
            if col not in out.columns or col not in category_maps:
                continue
            out[col] = LimmaRunner._series_with_category_names(
                out[col],
                category_maps.get(col),
            ).astype("string")
        out.attrs.update(meta_df.attrs)
        return out

    def _run_qc_and_plotting(
        self,
        expr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        label_series: dict[str, pd.Series],
        output_dir: Path,
        plot_n_pcs: int,
        prefix: str, # e.g., "orig" or "corr"
    ) -> dict: 
        # 1. Initialize Plotter
        plotter = limma_plotter(expr_df, n_pcs=plot_n_pcs)

        pca_svgs: dict[str, dict] = {}
        box_svgs: dict[str, str] = {}


        for name, s in label_series.items():
            # PCA plots
            svg1, svg2 = plotter.plot_pca_pairs(
                labels=s,
                label_name=name,
                title_prefix=f"By {name} ({prefix.capitalize()})",
            )
            pca_svgs[name] = {"pc1_pc2": svg1, "pc3_pc4": svg2}

            # Boxplots
            svg = plotter.plot_pc_boxplots(
                group_labels=s,
                group_name=name,
                pcs=(1, 2, 3, 4),
                title=f"PC1–4 by {name} ({prefix.capitalize()})",
            )
            box_svgs[name] = svg

        # Heatmap
        heatmap_labels: dict[str, pd.Series] = {
            name: s for name, s in label_series.items()
            if pd.Series(s).dropna().nunique() >= 2
        }

        if heatmap_labels:
            heatmap_svg = plotter.plot_pc_batch_heatmap(
                labels_dict=heatmap_labels,
                title=f"({prefix} correction)",
            )
        else:
            heatmap_svg = None

        # Silhouette scores
        sil_scores = plotter.compute_silhouette_scores(label_series)

        dendrogram_svg = plotter.plot_sample_dendrogram(
            label_dict=label_series,
            title=f"Hierarchical clustering ({prefix.capitalize()})",
            top_var_genes=2000, 
            zscore_genes=True,
            metric="correlation",
            method="average",
        )

        gene_assoc = None
        top_gene_boxplot_svg = None
        batch_like_name = self.batch_like_name
        if batch_like_name is None:
            batch_like_name = next(
                (n for n in label_series.keys() if any(k in n.lower() for k in ("batch", "center", "centrum", "site"))),
                None
            )
        if batch_like_name is not None:
            gene_assoc = plotter.gene_batch_association_summary(
                labels=label_series[batch_like_name],
                label_name=batch_like_name,
                fdr=0.1,
            )
            top_gene_boxplot_svg = plotter.plot_top_batch_genes_boxplot(
                labels=label_series[batch_like_name],
                label_name=batch_like_name,
                gene_table=gene_assoc["table"],
                top_n=12,
                title=f"Top batch-associated features by {batch_like_name} ({prefix.capitalize()})",
            )

        # Return results
        return {
            "plots": {
                "pca": pca_svgs,
                "boxplots": box_svgs,
                "heatmap": heatmap_svg,
                "dendrogram": dendrogram_svg,
                "top_batch_gene_boxplots": top_gene_boxplot_svg,
            },
            "silhouette_scores": sil_scores,
            "gene_level": gene_assoc, 
        }

    def run_limma(self, 
                  expr_df: pd.DataFrame, 
                  meta_df: Optional[pd.DataFrame], 
                  output_dir: Union[str, Path], 
                  cov_matrix: Optional[pd.DataFrame] = None, 
                  batch_col: Optional[str] = None, 
                  design_cols: Optional[Sequence[str]] = None, 
                  n_pc: int = 0, 
                  na_fill: Optional[Union[str, int, float]] = None,
                  r_script_path: Optional[Union[str, Path]] = None,
                  out_csv: str = "limma_results.csv",
                  plot_n_pcs: int = 10,
                  eval_cols: Optional[Sequence[str]] = None,
            ) -> dict:
        
        self.batch_like_name = batch_col
        category_maps = getattr(meta_df, "attrs", {}).get("category_maps", {}) if meta_df is not None else {}

        # Validate inputs and align samples
        if not isinstance(expr_df, pd.DataFrame):
            raise TypeError("expr_df must be a pandas DataFrame")
        _assert_unique(expr_df, "expr_df")
        if meta_df is not None:
            if not isinstance(meta_df, pd.DataFrame):
                raise TypeError("meta_df must be a pandas DataFrame")
            _assert_unique(meta_df, "meta_df")
        if cov_matrix is not None:
            if not isinstance(cov_matrix, pd.DataFrame):
                raise TypeError("cov_matrix must be a pandas DataFrame")
            _assert_unique(cov_matrix, "cov_matrix")

        # Filter low-variance genes
        required_meta_cols = [batch_col] + (design_cols if design_cols else [])
        X, M = _align_samples(expr_df, meta_df, required_meta_cols=required_meta_cols)
        if M is not None:
            M = self._restore_category_columns(
                M,
                required_meta_cols,
                category_maps,
            )
        v = X.var(axis=1, skipna=True, numeric_only=True)
        X = X.loc[v > 0]
        if M is not None:
            M = M.loc[X.columns]
        if cov_matrix is not None:
            common_cov_samples = X.columns.intersection(cov_matrix.index)
            if len(common_cov_samples) == 0:
                raise ValueError("No overlapping samples between expr_df.columns and cov_matrix.index")
            cov_matrix = cov_matrix.loc[common_cov_samples]
            X = X.loc[:, common_cov_samples]
            if M is not None:
                M = M.loc[common_cov_samples]
            else:
                M = pd.DataFrame(index=X.columns)

        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if r_script_path is None:
            r_script_path = Path(__file__).resolve().parent / "limma_correction.R"
        else:
            r_script_path = Path(r_script_path)
        
        # Determine correction types
        have_design = bool(design_cols)
        have_batch  = batch_col is not None
        have_covs   = cov_matrix is not None
        have_pcs    = n_pc > 0

        # Check basic requirements
        if have_design and M is None:
            raise ValueError("design_cols provided but meta_df is None")
        if have_batch and M is None:
            raise ValueError("batch_col provided but meta_df is None")
        if not (have_batch or have_covs or have_pcs):
            raise ValueError("You must provide at least one of: batch_col, cov_matrix, or n_pc")

        # Handle NAs in meta_df
        if M is not None and na_fill is not None:
            used_cols = set()
            if have_batch:   used_cols.add(batch_col)
            if have_design:  used_cols |= set(design_cols)
            used_cols = [c for c in used_cols if c in M.columns]
            if used_cols:
                categorical_cols = [c for c in used_cols if c in category_maps]
                numeric_cols = [c for c in used_cols if c not in category_maps]
                if numeric_cols:
                    block = (
                        M.loc[:, numeric_cols]
                        .replace({True: 1.0, False: 0.0})
                        .apply(pd.to_numeric, errors="coerce")
                        .astype("float64")
                        .fillna(float(na_fill))
                    )
                    M.loc[:, numeric_cols] = block
                if categorical_cols:
                    M.loc[:, categorical_cols] = M.loc[:, categorical_cols].fillna("__missing__")
        
        # Final checks and design string creation
        design_str = None
        if have_design:
            missing = [c for c in design_cols if c not in M.columns]
            if missing:
                raise ValueError(f"design_cols missing in meta_df: {missing}")
            # Combine design columns for the R formula (e.g., "condition + age")
            design_str = " + ".join(design_cols) 

        if have_batch and batch_col not in M.columns:
            raise ValueError(f"batch_col '{batch_col}' not in meta_df")

        # Check for mutually exclusive batch types
        if sum([have_batch, have_covs, have_pcs]) > 1:
             raise ValueError("The 'run_limma' function only supports one batch removal type at a time: batch_col OR cov_matrix OR n_pc. Please select only one.")

        # 2. Determine State and extra parameters
        state = None
        extra1 = None
        extra2 = design_str # extra2 is always the design string or None

        # State Map: 1: batch, 2: batch + design, 3: covariates, 4: covariates + design 5: pcs, 6: pcs + design
        
        if have_batch:
            state = 2 if have_design else 1
            extra1 = batch_col
        elif have_covs:
            state = 4 if have_design else 3
            # extra1 will be the path to the covariate matrix CSV
        elif have_pcs:
            state = 6 if have_design else 5
            extra1 = n_pc
        
        # 3. Prepare Files and Configuration
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            expr_filepath = temp_dir / "expr_data.csv"
            meta_filepath = temp_dir / "meta_data.csv"
            cov_filepath = temp_dir / "cov_matrix.csv"
            json_filepath = temp_dir / "config.json"
            output_filepath = output_dir / out_csv

            # Write input CSVs
            _write_csv(X, expr_filepath)
            _write_csv(M, meta_filepath)
            
            # Special handling for covariate matrix (State 3, 4)
            final_extra1 = extra1
            if have_covs:
                C = cov_matrix.loc[X.columns]
                if not C.index.equals(X.columns):
                    raise ValueError("cov_matrix rows must be in the same order as expr_df columns")
                _write_csv(C, cov_filepath)
                final_extra1 = str(cov_filepath) # Pass the path to R

            # Create JSON configuration
            config_data = {
                "state": state,
                "expr_filepath": str(expr_filepath),
                "meta_filepath": str(meta_filepath),
                "output_filepath": str(output_filepath),
                "extra1": final_extra1,
                "extra2": extra2
            }
            with open(json_filepath, 'w') as f:
                json.dump(config_data, f, indent=4)

            # Define Label Series for QC
            label_series: dict[str, pd.Series] = {}
            if M is not None:
                # 1) batch_col (if relevant and present)
                if batch_col and batch_col in M.columns:
                    label_series[batch_col] = self._series_with_category_names(
                        M[batch_col],
                        category_maps.get(batch_col),
                    )

                # 2) design_cols (if any and present)
                if design_cols:
                    for dc in design_cols:
                        if dc in M.columns and dc not in label_series:
                            label_series[dc] = self._series_with_category_names(
                                M[dc],
                                category_maps.get(dc),
                            )

                # 3) extra evaluation columns
                if eval_cols:
                    for c in eval_cols:
                        if c in M.columns and c not in label_series:
                            label_series[c] = self._series_with_category_names(
                                M[c],
                                category_maps.get(c),
                            )
                            
            result_dict = {}
            if M is not None and label_series:
                orig_qc_res = self._run_qc_and_plotting(
                    expr_df=X,
                    meta_df=M,
                    label_series=label_series,
                    output_dir=output_dir,
                    plot_n_pcs=plot_n_pcs,
                    prefix="orig", # Original results
                )
                # Store original results with distinct keys
                result_dict["plots_orig"] = orig_qc_res["plots"]
                result_dict["silhouette_scores_orig"] = orig_qc_res["silhouette_scores"]
                result_dict["gene_level_orig"] = orig_qc_res.get("gene_level")
            # 4. Load R code and Call Function
            if not self.r_loaded:
                try:
                    # Load the R script and get the main function
                    ro.r.source(str(r_script_path))
                    self.r_limma_func = ro.globalenv['execute_limma_batch_correction']
                    self.r_loaded = True
                except RRuntimeError as e:
                    raise RBackendError("R backend failed: " + str(e)) from e
            
            try:
                r_result_str = self.r_limma_func(str(json_filepath))[0]
            except RRuntimeError as e:
                raise RBackendError(
                    f"R function execute_limma_batch_correction crashed: {e}"
                ) from e
            
            # 5. Handle Output
            r_result_dict = json.loads(r_result_str)
            result_dict.update(r_result_dict)
            
            if result_dict['status'] == 'error':
                raise RuntimeError(f"R execution failed: {result_dict['message']}")
            
            if M is not None and output_filepath.exists() and label_series:
                expr_corr = pd.read_csv(output_filepath, index_col=0)
                
                # Align labels to the corrected expression data's columns (important if filtering happened)
                # Although M columns should match expr_corr columns, this is safer.
                aligned_label_series = {k: s.reindex(expr_corr.columns) for k, s in label_series.items()}

                corr_qc_res = self._run_qc_and_plotting(
                    expr_df=expr_corr,
                    meta_df=M,
                    label_series=aligned_label_series,
                    output_dir=output_dir,
                    plot_n_pcs=plot_n_pcs,
                    prefix="corr", # Corrected results
                )

                # Store corrected results with standard keys (overwriting any previous placeholder keys)
                result_dict["plots"] = corr_qc_res["plots"]
                result_dict["silhouette_scores"] = corr_qc_res["silhouette_scores"]
                result_dict["gene_level"] = corr_qc_res.get("gene_level")

            # Final cleanup and state
            result_dict["state"] = state
            return result_dict
    
        except Exception as e:
            raise GCError(f"LimmaRunner failed: {e}") from e
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
