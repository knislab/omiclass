import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, f_oneway
from typing import Dict, Optional, Sequence, Tuple, Union, List
from generic_classifier.utils.io import fig_to_svg_str 
from generic_classifier.utils.toolbox import _bh_fdr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist


LabelVector = Union[pd.Series, Sequence]
VAR_EPS = 1e-12


class limma_plotter:
    def __init__(
        self,
        expr: pd.DataFrame,
        n_pcs: int = 10,
        scale_for_pca: bool = False,
        scatter_figsize: Tuple[float, float] = (7, 5),
        boxplot_height: float = 3.5,
        boxplot_width_per_pc: float = 3.5,
        scatter_point_size: float = 70,
        heatmap_cmap: str = "viridis",
        heatmap_max_logp: float = 20,
    ):
        self.expr = expr.copy()
        self.n_pcs = n_pcs
        self.scale_for_pca = scale_for_pca
        self.scatter_figsize = scatter_figsize
        self.boxplot_height = boxplot_height
        self.boxplot_width_per_pc = boxplot_width_per_pc
        self.scatter_point_size = scatter_point_size
        self.heatmap_cmap = heatmap_cmap
        self.heatmap_max_logp = heatmap_max_logp

        self.pc_df, self.pca = self._compute_pca()

    # -------------------------------------------------------

    def _compute_pca(self):
        X = self.expr.T.values
        if self.scale_for_pca: X = StandardScaler().fit_transform(X)
        safe_n_pcs = min(self.n_pcs, X.shape[0], X.shape[1])
        pca = PCA(n_components=safe_n_pcs)
        self.n_pcs = safe_n_pcs
        pcs = pca.fit_transform(X)
        df = pd.DataFrame(pcs, index=self.expr.columns, columns=[f"PC{i+1}" for i in range(self.n_pcs)])
        return df, pca

    def _align_labels(self, labels: LabelVector) -> pd.Series:
        if isinstance(labels, pd.Series):
            return labels.reindex(self.expr.columns)
        return pd.Series(labels, index=self.expr.columns)


    def _build_meta(self, label_dict: Dict[str, LabelVector]):
        df = self.pc_df.copy()
        for name, vec in label_dict.items():
            df[name] = self._align_labels(vec).values
        return df

    # -------------------------------------------------------

    def plot_pca_pairs(self, labels: LabelVector, label_name: str, title_prefix: str = ""):
        lab = self._align_labels(labels)
        df = self.pc_df.copy()
        df[label_name] = lab.astype("string").fillna("NA").values

        fig2 = None
        svg2 = None
        fig1, ax1 = plt.subplots(figsize=self.scatter_figsize)
        sns.scatterplot(data=df, x="PC1", y="PC2", hue=label_name, s=self.scatter_point_size, ax=ax1)
        ax1.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax1.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)")
        svg1 = fig_to_svg_str(fig1)
        plt.close(fig1)

        if self.n_pcs >= 4:
            fig2, ax2 = plt.subplots(figsize=self.scatter_figsize)
            sns.scatterplot(data=df, x="PC3", y="PC4", hue=label_name, s=self.scatter_point_size, ax=ax2)
            ax2.set_xlabel(f"PC3 ({self.pca.explained_variance_ratio_[2]*100:.1f}%)")
            ax2.set_ylabel(f"PC4 ({self.pca.explained_variance_ratio_[3]*100:.1f}%)")
            svg2 = fig_to_svg_str(fig2)
            plt.close(fig2)

        return svg1, svg2

    # -------------------------------------------------------

    def plot_sample_dendrogram(
        self,
        label_dict: Dict[str, LabelVector],
        title: str = "",
        top_var_genes: int = 2000,
        zscore_genes: bool = True,
        metric: str = "correlation",
        method: str = "average",
        truncate_if_n_samples_gt: int = 300,
        truncate_p: int = 80,
    ) -> Optional[str]:
        if self.expr is None or self.expr.empty:
            return None

        # ---------- prepare matrix (genes x samples) ----------
        X = self.expr.copy()

        # keep top variable genes (optional)
        if top_var_genes and X.shape[0] > int(top_var_genes):
            var = X.var(axis=1, ddof=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = X.loc[var.nlargest(int(top_var_genes)).index]

        # cluster samples => rows=samples, cols=genes
        M = X.T

        if zscore_genes:
            denom = M.std(axis=0, ddof=1).replace(0, np.nan)
            M = (M - M.mean(axis=0)) / denom
            M = M.fillna(0.0)

        n_samples = int(M.shape[0])
        if n_samples < 2:
            return None

        # ---------- build row_colors (index must be samples) ----------
        row_colors = None
        label_names: List[str] = []

        if label_dict:
            cols = []
            label_names = list(label_dict.keys())
            for name, vec in label_dict.items():
                # align to self.expr.columns then reindex to M.index (samples actually clustered)
                s = self._align_labels(vec).reindex(M.index)
                s = s.where(s.notna(), "NA").astype(str)

                cats = sorted(pd.unique(s.values))
                palette = sns.color_palette("tab20", n_colors=max(3, len(cats)))
                cmap = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
                cols.append(s.map(cmap))

            row_colors = pd.concat(cols, axis=1)
            row_colors.columns = label_names

        has_colors = row_colors is not None and row_colors.shape[1] > 0
        k = int(row_colors.shape[1]) if has_colors else 0

        # ---------- linkage ----------
        try:
            D = pdist(M.values, metric=metric)
            Z = linkage(D, method=method)
        except Exception:
            # fail safe: no dendrogram possible
            return None

        # ---------- figure layout ----------
        height = min(12.0, max(5.0, 0.02 * n_samples + 4.0))
        width = 10.0 + (0.6 * k if k > 0 else 0.0)

        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(
            nrows=1,
            ncols=(1 + (1 if has_colors else 0)),
            width_ratios=([0.9, 9.1] if has_colors else [1.0]),
            wspace=0.05,
        )

        ax_colors = fig.add_subplot(gs[0, 0]) if has_colors else None
        ax_d = fig.add_subplot(gs[0, 1] if has_colors else gs[0, 0])

        truncate = n_samples > int(truncate_if_n_samples_gt)

        dendro_kwargs = dict(no_labels=True, orientation="left", ax=ax_d)
        if truncate:
            dendro_kwargs.update(dict(truncate_mode="lastp", p=int(truncate_p)))

        den = dendrogram(Z, **dendro_kwargs)

        # style dendrogram axis
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        for sp in ax_d.spines.values():
            sp.set_visible(False)

        # optional title (kept minimal)
        if title:
            ax_d.set_title(str(title), fontsize=10, pad=6)

        fig.tight_layout()

        # ---------- render color bars robustly ----------
        if has_colors and ax_colors is not None:
            # Ensure row_colors aligns to the clustered sample axis
            if not row_colors.index.equals(M.index):
                row_colors = row_colors.reindex(M.index)

            ordered = None

            # 1) If leaves are valid sample positions (non-truncated), use them
            leaves = den.get("leaves", [])
            leaf_idx = [
                int(i) for i in leaves
                if isinstance(i, (int, np.integer)) and 0 <= int(i) < n_samples
            ]
            if leaf_idx:
                ordered = row_colors.iloc[leaf_idx, :]

            # 2) If ivl contains sample ids (usually only if labels are shown), use it
            if ordered is None or ordered.shape[0] == 0:
                ivl = den.get("ivl", [])
                ivl = [lab for lab in ivl if lab in row_colors.index]
                if ivl:
                    ordered = row_colors.reindex(ivl)

            # 3) Last resort: match the number of displayed leaves by slicing
            if ordered is None or ordered.shape[0] == 0:
                target_n = len(den.get("ivl", []))  # may be 0 when no_labels=True
                if not target_n:
                    target_n = min(int(truncate_p) if truncate else n_samples, n_samples)
                ordered = row_colors.iloc[:target_n, :]

            # If still empty, skip color panel (but keep dendrogram)
            if ordered is not None and ordered.shape[0] > 0:
                # fill missing with neutral gray RGB tuples
                ordered = ordered.applymap(lambda x: (0.8, 0.8, 0.8) if pd.isna(x) else x)

                # build RGB cube (n_rows, n_cols, 3)
                arr = np.zeros((ordered.shape[0], ordered.shape[1], 3), dtype=float)
                for j in range(ordered.shape[1]):
                    col_vals = ordered.iloc[:, j].tolist()
                    arr[:, j, :] = np.asarray([c[:3] for c in col_vals], dtype=float)

                ax_colors.imshow(arr, aspect="auto", interpolation="nearest", origin="lower")
                ax_colors.set_xticks(range(k))
                ax_colors.set_xticklabels(label_names, rotation=90, fontsize=8)
                ax_colors.set_yticks([])
                for sp in ax_colors.spines.values():
                    sp.set_visible(False)
            else:
                # hide color axis if we couldn't render it
                ax_colors.axis("off")

        # ---------- finalize ----------
        svg = fig_to_svg_str(fig)
        plt.close(fig)
        return svg

    # -------------------------------------------------------

    def gene_batch_association_summary(
        self,
        labels: LabelVector,
        label_name: str,
        fdr: float = 0.05,
        min_group_n: int = 3,
    ) -> Optional[dict]:
        if self.expr is None or self.expr.empty:
            return None

        lab = self._align_labels(labels)
        lab = lab.reindex(self.expr.columns)
        keep_samples = lab.notna()
        if keep_samples.sum() < 2:
            return None

        X = self.expr.loc[:, keep_samples]
        y = lab.loc[keep_samples]

        nunique = y.nunique(dropna=True)
        if nunique < 2:
            return None

        genes = X.index.astype(str)
        pvals = np.full(len(genes), np.nan, dtype=float)
        effects = np.full(len(genes), np.nan, dtype=float)  # mean diff for binary; NaN for multi by default

        if nunique == 2:
            g1, g2 = list(pd.unique(y.dropna()))
            idx1 = (y == g1).values
            idx2 = (y == g2).values
            if idx1.sum() < min_group_n or idx2.sum() < min_group_n:
                return None

            A = X.loc[:, idx1].values
            B = X.loc[:, idx2].values

            # effect size: mean(A)-mean(B)
            effects = np.nanmean(A, axis=1) - np.nanmean(B, axis=1)

            # Welch t-test gene-by-gene
            for i in range(X.shape[0]):
                a = A[i, :]
                b = B[i, :]
                if np.nanvar(a) < VAR_EPS and np.nanvar(b) < VAR_EPS:
                    continue
                pvals[i] = ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue

        else:
            # multi-class: one-way ANOVA p-value per gene
            groups = [v for v in pd.unique(y.dropna())]
            group_masks = [(y == g).values for g in groups]
            if any(m.sum() < min_group_n for m in group_masks):
                return None

            for i in range(X.shape[0]):
                vecs = [X.iloc[i, :].values[m] for m in group_masks]
                vecs = [v[np.isfinite(v)] for v in vecs]
                if len(vecs) < 2 or any(len(v) < min_group_n for v in vecs):
                    continue
                try:
                    pvals[i] = f_oneway(*vecs).pvalue
                except Exception:
                    pvals[i] = np.nan

        qvals = _bh_fdr(pvals)

        df = pd.DataFrame(
            {
                "gene": genes,
                "p_value": pvals,
                "q_value": qvals,
                "effect": effects,
            }
        ).sort_values(["q_value", "p_value"], ascending=[True, True])

        sig = df["q_value"].notna() & (df["q_value"] <= float(fdr))
        out = {
            "label_name": label_name,
            "n_genes_tested": int(df["p_value"].notna().sum()),
            "n_significant_fdr": int(sig.sum()),
            "fdr": float(fdr),
            "table": df,
        }
        return out
    # -------------------------------------------------------
    def plot_top_batch_genes_boxplot(
        self,
        labels: LabelVector,
        label_name: str,
        gene_table: pd.DataFrame,
        top_n: int = 12,
        title: str = "",
    ) -> Optional[str]:
        if gene_table is None or gene_table.empty:
            return None

        lab = self._align_labels(labels).reindex(self.expr.columns)
        keep = lab.notna()
        if keep.sum() < 2:
            return None

        top_genes = gene_table["gene"].dropna().astype(str).head(int(top_n)).tolist()
        top_genes = [g for g in top_genes if g in self.expr.index.astype(str)]
        if not top_genes:
            return None

        X = self.expr.loc[top_genes, keep].copy()
        y = lab.loc[keep].astype(str)

        long = (X.T.assign(**{label_name: y.values}).melt(id_vars=[label_name], var_name="gene", value_name="expr"))

        fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(top_genes)), 4.5))
        sns.boxplot(data=long, x="gene", y="expr", hue=label_name, ax=ax)
        ax.set_xlabel("Gene")
        ax.set_ylabel("Expression")
        plt.xticks(rotation=90)
        plt.tight_layout()
        svg = fig_to_svg_str(fig)
        plt.close(fig)
        return svg
    # -------------------------------------------------------

    def plot_pc_boxplots(self, group_labels: LabelVector, group_name: str, pcs=(1,2,3,4), title="", save_path=None):
        group = self._align_labels(group_labels)
        df = self.pc_df.copy()
        df[group_name] = group.values

        fig, axes = plt.subplots(1, len(pcs), figsize=(self.boxplot_width_per_pc*len(pcs), self.boxplot_height))
        if len(pcs) == 1: axes = [axes]

        for ax, pc in zip(axes, pcs):
            col = f"PC{pc}"
            df.boxplot(column=col, by=group_name, ax=ax)
            ax.set_title(col)
            ax.set_xlabel(group_name)
            ax.set_ylabel(col)

        plt.tight_layout()
        svg = fig_to_svg_str(fig)
        plt.close(fig)
        return svg

    # -------------------------------------------------------

    def compute_silhouette_scores(self, labels_dict: Dict[str, LabelVector]):
        scores = {}
        pc_cols = [f"PC{i+1}" for i in range(self.n_pcs)]
        df = self.pc_df.copy()

        for name, vec in labels_dict.items():
            df[name] = self._align_labels(vec).values
            sub = df.dropna(subset=[name])
            if sub[name].nunique() < 2: continue
            scores[name] = silhouette_score(sub[pc_cols].values, sub[name].values)

        return scores

    # -------------------------------------------------------

    def plot_pc_batch_heatmap(
        self,
        binary_labels: Dict[str, LabelVector],
        multi_label: Optional[LabelVector] = None,
        multi_label_name: str = "centrum",
        title: str = "",
        save_path: Optional[str] = None,
    ):
        labels = dict(binary_labels)
        if multi_label is not None: labels[multi_label_name] = multi_label
        meta = self._build_meta(labels)

        results = {}
        pcs = [f"PC{i+1}" for i in range(min(self.n_pcs, 10))]

        for col, vec in binary_labels.items():
            m = meta.dropna(subset=[col]) if col == "IGHV" else meta
            if m[col].nunique() != 2: continue
            g1, g2 = m[col].dropna().unique()
            vals = []
            for pc in pcs:
                a = m.loc[m[col]==g1, pc]; b = m.loc[m[col]==g2, pc]
                if len(a)<2 or len(b)<2: vals.append(np.nan); continue
                if np.var(a) < VAR_EPS and np.var(b) < VAR_EPS: vals.append(np.nan); continue
                
                p = ttest_ind(a, b, equal_var=False).pvalue
                vals.append(min(-np.log10(p) if p>0 else self.heatmap_max_logp, self.heatmap_max_logp))
            results[col] = vals

        if multi_label is not None:
            m = meta.dropna(subset=[multi_label_name])
            vals = []
            for pc in pcs:
                groups = [m.loc[m[multi_label_name]==v, pc].dropna() for v in m[multi_label_name].unique()]
                groups = [g for g in groups if len(g)>1]
                if len(groups)<2: vals.append(np.nan); continue
                p = f_oneway(*groups).pvalue
                vals.append(min(-np.log10(p) if p>0 else self.heatmap_max_logp, self.heatmap_max_logp))
            results[multi_label_name] = vals

        df_res = pd.DataFrame(results, index=pcs)
        if "centrum" in df_res.columns: df_res.rename(columns={"centrum": "center"}, inplace=True)

        vmax = min(self.heatmap_max_logp, np.nanmax(df_res.values)*1.05 if np.isfinite(df_res.values).any() else 1)

        fig, ax = plt.subplots(figsize=(1.8*df_res.shape[1], 0.5*df_res.shape[0]+2))
        sns.heatmap(df_res, annot=True, cmap=self.heatmap_cmap, vmin=0, vmax=vmax, ax=ax)
        ax.set_xlabel("Variables"); ax.set_ylabel("PCs")
        svg = fig_to_svg_str(fig)
        plt.close(fig)

        return svg