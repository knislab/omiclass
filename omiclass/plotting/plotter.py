# generic-classifier/plotting/plotter.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns


from ..utils.toolbox import as_numpy_y, _safe_row_normalized_cm
from ..utils.io import ensure_path as _ensure, fig_to_svg_str


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize

class FinalSummaryPlotter:
    def __init__(self, *, labels: List[str], y_true, per_model: Dict[str, Dict[str, np.ndarray]], out_dir: Path, class_name_map: Optional[Dict[int, str]] = None,):
        self.labels = list(labels)
        self.K = len(self.labels)
        self.y_true = as_numpy_y(y_true)
        self.per_model = self.clean_models_names(per_model)
        self.out_dir = Path(out_dir)

        if class_name_map is not None:
            self.class_name_map = {int(k): str(v) for k, v in class_name_map.items()}
        else:
            self.class_name_map = {i: str(lab) for i, lab in enumerate(self.labels)}

    def clean_models_names(self, per_model: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        cleaned = {}
        for name, res in per_model.items():
            clean_name = name.replace("_", " ").title()
            cleaned[clean_name] = res
        return cleaned

    def plot_leaderboard_bar(self) -> Path:
        rows = []
        for name, res in self.per_model.items():
            y_pred = as_numpy_y(res["y_pred"])
            rows.append({
                "model": name,
                "accuracy": accuracy_score(self.y_true, y_pred),
                "precision": precision_score(self.y_true, y_pred, average="macro", zero_division=0),
                "recall":    recall_score(self.y_true, y_pred, average="macro", zero_division=0),
                "macro_f1":  f1_score(self.y_true, y_pred, average="macro", zero_division=0),
            })
        df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)

        fig, ax = plt.subplots(figsize=(7, 3 + 0.35*len(df)))
        
        # Calculate percentage scores
        df['macro_f1_pct'] = df['macro_f1'] * 100 
        df['accuracy_pct'] = df['accuracy'] * 100 
        
        ax.barh(df["model"], df["macro_f1_pct"])
        ax.invert_yaxis()
        
        for i, r in df.reset_index(drop=True).iterrows():
            # Annotation text shows percentage values
            ax.text(r["macro_f1_pct"], i, 
                    f"  F1={r['macro_f1_pct']:.1f}% | Acc={r['accuracy_pct']:.1f}%",
                    va="center", ha="left", fontsize=9)
            
        ax.set_xlabel("Macro-F1 (%)") # Update Axis Label
        ax.set_xlim(0, 100)           # Set x-axis limit to percentage scale
        
        fig.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.90)
        svg = fig_to_svg_str(fig)
        plt.close(fig)
        return svg

    def plot_confusion_matrices(self) -> str:
        model_items = list(self.per_model.items())
        n_models = len(model_items)
        n_rows = 2
        n_cols = max(1, math.ceil(n_models / 2))
        label_index = self._get_label_index()
        class_names = [self.class_name_map.get(int(i), str(i)) for i in label_index]
        K = len(label_index)

        # figure size scales with number of classes and grid width
        fig_w = n_cols * (4.5 + 0.3 * K)
        fig_h = n_rows * (4.5 + 0.3 * K)
        
        # Increase the width slightly to accommodate the external colorbar
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w + 1, fig_h))
        
        if n_rows == 1 or n_cols == 1:
            axes = np.atleast_2d(axes) # Ensure 2D for consistent indexing

        im = None
        for idx, (name, res) in enumerate(model_items):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]
            y_pred = as_numpy_y(res["y_pred"])
            cm_counts, cm = _safe_row_normalized_cm(self.y_true, y_pred, labels=label_index)

            cm_percentage = cm * 100
            im = sns.heatmap(
                cm_percentage, 
                annot=True,
                fmt='.0f',
                cmap='Blues',
                cbar=False,
                vmin=0, vmax=100,
                ax=ax
            )
            ax.set_xticks(np.arange(K) + 0.5)
            ax.set_yticks(np.arange(K) + 0.5)
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names, rotation=0)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(name)
            
        # hide any unused subplots
        total_slots = n_rows * n_cols
        for k in range(n_models, total_slots):
            axes.flatten()[k].axis("off")

        # --- COLORBAR LOGIC ---
        if n_models > 0:
            # We create a new axis on the right specifically for the colorbar
            # This prevents it from overlapping the matrices
            fig.subplots_adjust(right=0.88) # Make room on the right
            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
            
            # Reference the mapping from the last heatmap
            mappable = axes.flatten()[n_models-1].collections[0]
            cbar = fig.colorbar(mappable, cax=cbar_ax)
            cbar.set_label("Row-normalized Score", rotation=90, labelpad=15)

        fig.subplots_adjust(left=0.08, bottom=0.15, wspace=0.4, hspace=0.4)
        svg = fig_to_svg_str(fig)
        plt.close(fig)
        return svg


    def plot_macro_pr_and_roc(self) -> Tuple[Path, Path]:
        label_index = self._get_label_index()
        K = len(label_index)
        Y = np.column_stack([(self.y_true == k).astype(int) for k in label_index])
        grid_pr = np.linspace(0, 1, 500)
        grid_fpr = np.linspace(0, 1, 500)

        # PR
        fig_pr, ax_pr = plt.subplots(figsize=(6.5, 5))
        for name, res in self.per_model.items():
            P = np.asarray(res["y_proba"])
            P = self._ensure_proba_shape(P, K)

            prs = []
            for k in range(K):
                precision, recall, _ = precision_recall_curve(Y[:, k], P[:, k])
                order = np.argsort(recall)
                recall_s = recall[order]; precision_s = precision[order]
                prs.append(np.interp(grid_pr, recall_s, precision_s,left=precision_s[0], right=precision_s[-1]))
            macro_precision = np.mean(np.vstack(prs), axis=0)
            ap_macro = average_precision_score(Y, P[:, :K], average="macro")
            ax_pr.plot(grid_pr, macro_precision, label=f"{name} (AUPRC={ap_macro:.3f})")
        ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
        ax_pr.legend(fontsize=8)
        fig_pr.tight_layout()
        svg_pr =fig_to_svg_str(fig_pr)
        plt.close(fig_pr)

        # ROC
        fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5))
        ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        for name, res in self.per_model.items():
            P = np.asarray(res["y_proba"])
            P = self._ensure_proba_shape(P, K)

            tprs, aucs = [], []
            for k in range(K):
                fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
                tprs.append(np.interp(grid_fpr, fpr, tpr, left=0, right=1))
                aucs.append(auc(fpr, tpr))
            macro_tpr = np.mean(np.vstack(tprs), axis=0)
            macro_auc = float(np.mean(aucs))
            ax_roc.plot(grid_fpr, macro_tpr, label=f"{name} (AUROC={macro_auc:.3f})")
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(fontsize=8)
        fig_roc.tight_layout()
        svg_roc = fig_to_svg_str(fig_roc)
        plt.close(fig_roc)

        return svg_pr, svg_roc

    def plot_metric_barplot(self) -> Path:
        records = []
        for name, res in self.per_model.items():
            y_pred = as_numpy_y(res["y_pred"])
            cleaned_name = None
            if len(name) > 18:
                spaces = [i for i, c in enumerate(name) if c == " "]
                
                if spaces:
                    mid_idx = spaces[len(spaces) // 2]
                    cleaned_name = name[:mid_idx] + "\n" + name[mid_idx + 1:]
                else:
                    cleaned_name = name
            else:
                cleaned_name = name
            records.append({
                "model":     cleaned_name,
                "Accuracy":  accuracy_score(self.y_true, y_pred),
                "Precision": precision_score(self.y_true, y_pred, average="macro", zero_division=0),
                "Recall":    recall_score(self.y_true, y_pred, average="macro", zero_division=0),
                "Macro F1":  f1_score(self.y_true, y_pred, average="macro", zero_division=0),
            })
        
        df = pd.DataFrame(records).melt(
            id_vars="model",
            value_vars=["Accuracy", "Precision", "Recall", "Macro F1"],
            var_name="Metric",
            value_name="Score"
        )
        # sort by f1 score for better visual grouping
        f1_order = df[df["Metric"] == "Macro F1"].sort_values("Score", ascending=False)["model"]
        df["model"] = pd.Categorical(df["model"], categories=f1_order, ordered=True)
        
        # Convert scores to percentage for display/ylim
        df['Score_pct'] = df['Score'] * 100 

        fig_w = max(6, 1.5 * len(df["model"].unique())) * 1.75
        fig_h = 6
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # Use seaborn.barplot as requested
        ax = sns.barplot(
            data=df, 
            x="model", 
            y="Score_pct", # Plot percentage score
            hue="Metric", 
            errorbar=None,
            ax=ax,
            palette="Set1" # Use a distinct color palette
        )

        # Annotate the bars with percentage scores (manual annotation)
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 1: # Only annotate visibly meaningful bars
                    ax.annotate(
                        f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='left', 
                        va='bottom',
                        fontsize=8,
                        rotation=35
                    )


        ax.set_ylabel("Score (%)")
        ax.set_xlabel("Model")

        # Set y-axis limit to percentage scale
        ax.set_ylim(0, 105) 
        
        # Adjust legend and tick labels
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title_fontsize=11)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        
        fig.tight_layout()
        svg = fig_to_svg_str(fig)
        plt.close(fig)
        return svg

    # ---------- utils ----------
    def _ensure_proba_shape(self, P: np.ndarray, K: int) -> np.ndarray:
        """
        Return probabilities with shape (n, K). Handles binary (n,) or (n,1) or (n,2).
        Pads/truncates to K if mismatched (strictly for plotting).
        """
        if P.ndim == 1:
            P = np.column_stack([1 - P, P])
        if P.shape[1] == 1 and K == 2:
            P = np.column_stack([1 - P[:, 0], P[:, 0]])
        if P.shape[1] != K:
            if P.shape[1] > K:
                P = P[:, :K]
            else:
                pad = np.zeros((P.shape[0], K - P.shape[1]))
                P = np.hstack([P, pad])
        return P


    def _get_label_index(self) -> np.ndarray:
        # If we were given an explicit mapping, its keys define the label IDs.
        if hasattr(self, "class_name_map") and self.class_name_map:
            return np.array(sorted(self.class_name_map.keys()), dtype=int)

        # Fallback: infer from data (safer than arange(K))
        present = np.unique(self.y_true)
        for _, res in self.per_model.items():
            present = np.unique(np.concatenate([present, np.unique(as_numpy_y(res["y_pred"]))]))
        return np.array(sorted(present), dtype=int)