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
    def __init__(self, *, labels: List[str], y_true, per_model: Dict[str, Dict[str, np.ndarray]], per_model_train: Dict[str, Dict[str, np.ndarray]], out_dir: Path, class_name_map: Optional[Dict[int, str]] = None, y_true_train=None,):
        self.labels = list(labels)
        self.K = len(self.labels)
        self.y_true = as_numpy_y(y_true)
        self.y_true_train = as_numpy_y(y_true_train) if y_true_train is not None else self.y_true
        self.per_model = self.clean_models_names(per_model)
        self.out_dir = Path(out_dir)
        self.per_model_train = self.clean_models_names(per_model_train)


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

    @staticmethod
    def _interpolate_pr_curve(grid: np.ndarray, recall: np.ndarray, precision: np.ndarray) -> np.ndarray:
        order = np.argsort(recall)
        recall_s = recall[order]
        precision_s = precision[order]
        unique_recall, first_idx = np.unique(recall_s, return_index=True)
        return np.interp(grid, unique_recall, precision_s[first_idx])

    def plot_leaderboard_bar(self) -> Path:
        rows = []
        for name, res in self.per_model.items():
            y_pred = as_numpy_y(res["y_pred"])
            rows.append({
                "model": name,
                "accuracy": accuracy_score(self.y_true, y_pred),
                "precision": precision_score(self.y_true, y_pred, labels=self._get_label_index(), average="macro", zero_division=0),
                "recall":    recall_score(self.y_true, y_pred, labels=self._get_label_index(), average="macro", zero_division=0),
                "macro_f1":  f1_score(self.y_true, y_pred, labels=self._get_label_index(), average="macro", zero_division=0),
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
        svg = fig_to_svg_str(fig, pdf_name="leaderboard_bar")
        plt.close(fig)
        return svg
    
    def _resolve_model_key(self, key: str) -> str:
        if key in self.per_model:
            return key

        target = self._normalize_model_key(key)

        for existing_key in self.per_model.keys():
            if self._normalize_model_key(existing_key) == target:
                return existing_key

        raise KeyError(
            f"best_model_key='{key}' was not found in per_model. "
            f"Available models: {list(self.per_model.keys())}"
        )

    def plot_confusion_matrices(self, best_model_key: str | None = None) -> str:
        if best_model_key is not None:
            resolved_key = self._resolve_model_key(best_model_key)
            model_items = [(resolved_key, self.per_model[resolved_key])]
        else:
            model_items = list(self.per_model.items())

        n_models = len(model_items)
        if n_models == 0:
            raise ValueError("No models found in per_model.")

        label_index = self._get_label_index()
        class_names = [self.class_name_map.get(int(i), str(i)) for i in label_index]
        K = len(label_index)

        fig_w = 5.0 + 0.35 * K
        fig_h = 4.8 + 0.35 * K

        fig, ax = plt.subplots(1, 1, figsize=(fig_w + 1, fig_h))

        name, res = model_items[0]
        y_pred = as_numpy_y(res["y_pred"])
        cm_counts, cm = _safe_row_normalized_cm(self.y_true, y_pred, labels=label_index)

        cm_percentage = cm * 100
        sns.heatmap(
            cm_percentage,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=True,
            vmin=0,
            vmax=100,
            ax=ax,
            cbar_kws={"label": "Row-normalized Score"},
        )

        ax.set_xticks(np.arange(K) + 0.5)
        ax.set_yticks(np.arange(K) + 0.5)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names, rotation=0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)

        fig.tight_layout()
        svg = fig_to_svg_str(fig, pdf_name="confusion_matrix")
        plt.close(fig)
        return svg


    def plot_macro_pr_and_roc(self, bold_model_key: Optional[str] = None) -> Tuple[Path, Path]:
        label_index = self._get_label_index()
        K = len(label_index)
        Y = np.column_stack([(self.y_true_train == k).astype(int) for k in label_index])
        grid_pr = np.linspace(0, 1, 500)
        grid_fpr = np.linspace(0, 1, 500)
        selected_model_key = self._normalize_model_key(bold_model_key)
        is_binary = K == 2

        def _is_selected(name: str) -> bool:
            return bool(
                selected_model_key
                and self._normalize_model_key(name) == selected_model_key
            )

        def _bold_legend_model(ax):
            if not selected_model_key:
                return

            legend = ax.get_legend()
            if legend is None:
                return

            for text in legend.get_texts():
                model_name = text.get_text().split(" (", 1)[0]
                if _is_selected(model_name):
                    text.set_fontweight("bold")

        # PR
        fig_pr, ax_pr = plt.subplots(figsize=(6.5, 5))
        pr_items = []
        for name, res in self.per_model_train.items():
            P = np.asarray(res["y_proba"])
            P = self._ensure_proba_shape(P, K)

            if is_binary:
                precision, recall, _ = precision_recall_curve(Y[:, 1], P[:, 1])
                pr_curve = self._interpolate_pr_curve(grid_pr, recall, precision)
                ap_score = average_precision_score(Y[:, 1], P[:, 1])
            else:
                prs = []
                for k in range(K):
                    precision, recall, _ = precision_recall_curve(Y[:, k], P[:, k])
                    prs.append(
                        self._interpolate_pr_curve(grid_pr, recall, precision)
                    )

                pr_curve = np.mean(np.vstack(prs), axis=0)
                ap_score = average_precision_score(Y, P[:, :K], average="macro")

            pr_items.append((name, pr_curve, ap_score))

        pr_items.sort(key=lambda x: x[2], reverse=True)

        pr_metric_label = "AP" if is_binary else "macro AP"
        pr_title = "Precision–Recall Curve" if is_binary else "Macro Precision–Recall Curve"
        for name, pr_curve, ap_score in pr_items:
            selected = _is_selected(name)
            ax_pr.plot(
                grid_pr,
                pr_curve,
                label=f"{name} ({pr_metric_label}={ap_score:.3f})",
                linewidth=3.0 if selected else 1.5,
                zorder=10 if selected else 2,
            )
        ax_pr.set_title(pr_title)
        ax_pr.legend(fontsize=8)
        _bold_legend_model(ax_pr)
        fig_pr.tight_layout()
        svg_pr =fig_to_svg_str(fig_pr, pdf_name="macro_precision_recall")
        plt.close(fig_pr)

        # ROC
        fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5))
        ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        roc_items = []

        for name, res in self.per_model_train.items():
            P = np.asarray(res["y_proba"])
            P = self._ensure_proba_shape(P, K)

            if is_binary:
                fpr, tpr, _ = roc_curve(Y[:, 1], P[:, 1])
                roc_curve_values = np.interp(grid_fpr, fpr, tpr, left=0, right=1)
                auc_score = auc(fpr, tpr)
            else:
                tprs, aucs = [], []
                for k in range(K):
                    fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
                    tprs.append(np.interp(grid_fpr, fpr, tpr, left=0, right=1))
                    aucs.append(auc(fpr, tpr))

                roc_curve_values = np.mean(np.vstack(tprs), axis=0)
                auc_score = float(np.mean(aucs))

            roc_items.append((name, roc_curve_values, auc_score))

        roc_items.sort(key=lambda x: x[2], reverse=True)

        roc_metric_label = "AUROC" if is_binary else "macro AUROC"
        roc_title = "ROC Curve" if is_binary else "Macro ROC Curve"
        for name, roc_curve_values, auc_score in roc_items:
            selected = _is_selected(name)
            ax_roc.plot(
                grid_fpr,
                roc_curve_values,
                label=f"{name} ({roc_metric_label}={auc_score:.3f})",
                linewidth=3.0 if selected else 1.5,
                zorder=10 if selected else 2,
            )
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(roc_title)
        ax_roc.legend(fontsize=8)
        _bold_legend_model(ax_roc)
        fig_roc.tight_layout()
        svg_roc = fig_to_svg_str(fig_roc, pdf_name="macro_roc")
        plt.close(fig_roc)

        return svg_pr, svg_roc

    def plot_metric_barplot(self) -> Path:
        records = []
        for name, res in self.per_model_train.items():
            y_pred = as_numpy_y(res["y_pred"])
            if y_pred.shape[0] != self.y_true_train.shape[0]:
                continue
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
                "Accuracy":  accuracy_score(self.y_true_train, y_pred),
                "Precision": precision_score(self.y_true_train, y_pred, labels=self._get_label_index(), average="macro", zero_division=0),
                "Recall":    recall_score(self.y_true_train, y_pred, labels=self._get_label_index(), average="macro", zero_division=0),
                "Macro F1":  f1_score(self.y_true_train, y_pred, labels=self._get_label_index(), average="macro", zero_division=0),
            })
        if not records:
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.axis("off")
            ax.text(0.5, 0.5, "No training CV payload available", ha="center", va="center")
            svg = fig_to_svg_str(fig, pdf_name="metric_barplot")
            plt.close(fig)
            return svg

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
        svg = fig_to_svg_str(fig, pdf_name="metric_barplot")
        plt.close(fig)
        return svg

    # ---------- utils ----------
    @staticmethod
    def _normalize_model_key(name: Optional[str]) -> str:
        if name is None:
            return ""
        return "".join(char for char in str(name).casefold() if char.isalnum())

    def _ensure_proba_shape(self, P: np.ndarray, K: int) -> np.ndarray:
        from omiclass.eval.prediction import validate_probabilities
        return validate_probabilities(P, np.arange(K), len(P))


    def _get_label_index(self) -> np.ndarray:
        # Explicit numeric labels define probability-column order, including absent classes.
        try:
            labels = np.asarray([int(label) for label in self.labels])
            if len(labels) and len(set(labels)) == len(labels):
                return labels
        except (ValueError, TypeError):
            pass
        # Historical display-name callers retain their mapped-label behavior.
        if hasattr(self, "class_name_map") and self.class_name_map:
            return np.array(sorted(self.class_name_map.keys()), dtype=int)

        # Fallback: infer from data (safer than arange(K))
        present = np.unique(self.y_true)
        for _, res in self.per_model.items():
            present = np.unique(np.concatenate([present, np.unique(as_numpy_y(res["y_pred"]))]))
        return np.array(sorted(present), dtype=int)
