from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


class PlottingFramework:
    """Regenerate publication plots from ``pdf_plots/base_files`` only.

    The pipeline writes compact CSV/JSON inputs through the ``save_*`` methods.
    Loading and plotting those files never touches an HTML report or a model.
    """

    MANIFEST = "manifest.json"

    def __init__(self, run_dir: str | Path, output_dir: str | Path | None = None):
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        run_dir = Path(run_dir)
        self.pdf_dir = run_dir if run_dir.name == "pdf_plots" else run_dir / "pdf_plots"
        self.base_dir = self.pdf_dir / "base_files"
        self.output_dir = Path(output_dir) if output_dir else self.pdf_dir / "publication"
        self.manifest = self._read_manifest(self.base_dir)

    @staticmethod
    def _read_manifest(base_dir: Path) -> dict[str, Any]:
        path = base_dir / PlottingFramework.MANIFEST
        if not path.exists():
            raise FileNotFoundError(
                f"Plot base-data manifest not found: {path}. Run the pipeline with PDF: true."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))

        # Repair manifests from early PDF-framework runs where PCA CSVs were
        # written but the top-level pca entry was accidentally omitted.
        if "pca" not in manifest:
            pca_dir = base_dir / "pca"
            required = {
                "before_scores": pca_dir / "before_scores.csv",
                "after_scores": pca_dir / "after_scores.csv",
                "before_associations": pca_dir / "before_associations.csv",
                "after_associations": pca_dir / "after_associations.csv",
                "explained_variance": pca_dir / "explained_variance.json",
            }
            if all(file.exists() for file in required.values()):
                score_columns = pd.read_csv(required["before_scores"], nrows=0).columns
                metadata_columns = [
                    column for column in score_columns
                    if column != "sample_id" and not re.fullmatch(r"PC\d+", str(column))
                ]
                cohort_column = next(
                    (column for column in metadata_columns if str(column).casefold() == "cohort"),
                    metadata_columns[0] if metadata_columns else None,
                )
                manifest["pca"] = {
                    key: file.relative_to(base_dir).as_posix()
                    for key, file in required.items()
                }
                manifest["pca"].update(
                    {"cohort_column": cohort_column, "heatmap_vmax": None}
                )
                path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return manifest

    @staticmethod
    def _update_manifest(base_dir: Path, **updates: Any) -> Path:
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / PlottingFramework.MANIFEST
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current.update(updates)
        path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def _safe_name(value: str) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
        return name or "plot"

    @staticmethod
    def save_classification_data(
        base_dir: str | Path,
        *,
        variants: Mapping[str, Mapping[str, Any]],
        selected_model: str,
        class_name_map: Mapping[int, str],
        is_b_all: bool,
        curve_variant: str = "corrected",
        curve_models: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Path:
        """Persist y_true, y_pred and y_proba for each available evaluation."""
        base_dir = Path(base_dir)
        class_dir = base_dir / "classification"
        class_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}

        for variant, payload in variants.items():
            y_true = np.asarray(payload["y_true"]).reshape(-1)
            y_pred = np.asarray(payload["y_pred"]).reshape(-1)
            proba = np.asarray(payload["y_proba"])
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            labels = list(payload.get("labels") or sorted(class_name_map))
            if len(labels) != proba.shape[1]:
                labels = list(range(proba.shape[1]))
            if not (len(y_true) == len(y_pred) == proba.shape[0]):
                raise ValueError(f"Inconsistent classification payload lengths for '{variant}'.")

            index = payload.get("sample_ids")
            if index is None:
                index = np.arange(len(y_true))
            frame = pd.DataFrame({"sample_id": np.asarray(index), "y_true": y_true, "y_pred": y_pred})
            for col, label in enumerate(labels):
                frame[f"proba__{label}"] = proba[:, col]
            filename = f"{PlottingFramework._safe_name(variant)}.csv"
            frame.to_csv(class_dir / filename, index=False)
            files[variant] = f"classification/{filename}"

        curve_files: dict[str, str] = {}
        curve_dir = class_dir / "curves"
        for model, payload in (curve_models or {}).items():
            y_true = np.asarray(payload["y_true"]).reshape(-1)
            proba = np.asarray(payload["y_proba"])
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            labels = list(payload.get("labels") or sorted(class_name_map))
            if len(labels) != proba.shape[1]:
                labels = list(range(proba.shape[1]))
            if len(y_true) != proba.shape[0]:
                raise ValueError(f"Inconsistent curve payload lengths for '{model}'.")

            curve_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({"y_true": y_true})
            for col, label in enumerate(labels):
                frame[f"proba__{label}"] = proba[:, col]
            filename = f"{PlottingFramework._safe_name(model)}.csv"
            frame.to_csv(curve_dir / filename, index=False)
            curve_files[str(model)] = f"classification/curves/{filename}"

        return PlottingFramework._update_manifest(
            base_dir,
            classification={
                "files": files,
                "selected_model": str(selected_model),
                "class_name_map": {str(k): str(v) for k, v in class_name_map.items()},
                "is_b_all": bool(is_b_all),
                "curve_variant": curve_variant if curve_variant in files else next(iter(files), None),
                "curve_files": curve_files,
                "curve_model_order": list(curve_files),
            },
        )

    @staticmethod
    def save_pca_data(
        base_dir: str | Path,
        *,
        before_scores: pd.DataFrame,
        after_scores: pd.DataFrame,
        metadata: pd.DataFrame,
        before_variance: Sequence[float],
        after_variance: Sequence[float],
        before_associations: pd.DataFrame,
        after_associations: pd.DataFrame,
        cohort_column: Optional[str],
        heatmap_vmax: Optional[float],
    ) -> Path:
        base_dir = Path(base_dir)
        pca_dir = base_dir / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)

        metadata = metadata.copy()
        metadata.index = metadata.index.map(str)
        before = before_scores.copy()
        after = after_scores.copy()
        before.index = before.index.map(str)
        after.index = after.index.map(str)
        before.join(metadata, how="left").to_csv(pca_dir / "before_scores.csv", index_label="sample_id")
        after.join(metadata, how="left").to_csv(pca_dir / "after_scores.csv", index_label="sample_id")
        before_associations.to_csv(pca_dir / "before_associations.csv", index_label="PC")
        after_associations.to_csv(pca_dir / "after_associations.csv", index_label="PC")
        (pca_dir / "explained_variance.json").write_text(
            json.dumps(
                {
                    "before": [float(x) for x in before_variance],
                    "after": [float(x) for x in after_variance],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return PlottingFramework._update_manifest(
            base_dir,
            pca={
                "before_scores": "pca/before_scores.csv",
                "after_scores": "pca/after_scores.csv",
                "before_associations": "pca/before_associations.csv",
                "after_associations": "pca/after_associations.csv",
                "explained_variance": "pca/explained_variance.json",
                "cohort_column": cohort_column,
                "heatmap_vmax": heatmap_vmax,
            },
        )

    def _base_file(self, relative: str) -> Path:
        return self.base_dir / relative

    def _save(self, fig, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return path

    def _classification_frame(self, variant: str) -> pd.DataFrame:
        config = self.manifest.get("classification", {})
        relative = config.get("files", {}).get(variant)
        if not relative:
            raise KeyError(f"No saved classification data for variant '{variant}'.")
        return pd.read_csv(self._base_file(relative))

    def plot_confusion_matrix(self, variant: str) -> Path:
        config = self.manifest["classification"]
        frame = self._classification_frame(variant)
        labels = sorted(set(frame["y_true"]).union(frame["y_pred"]))
        cm = confusion_matrix(frame["y_true"], frame["y_pred"], labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        names = config.get("class_name_map", {})
        display = [names.get(str(int(x)) if float(x).is_integer() else str(x), str(x)) for x in labels]

        fig, ax = plt.subplots(figsize=(6 + 0.35 * len(labels), 4.8 + 0.35 * len(labels)))
        sns.heatmap(normalized, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=display, yticklabels=display, cbar_kws={"label": "Row-normalized (%)"}, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title=f"{config['selected_model']} — {variant}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self._save(fig, f"confusion_matrix_{self._safe_name(variant)}.pdf")

    def plot_confusion_matrices(self) -> list[Path]:
        config = self.manifest.get("classification", {})
        variants = list(config.get("files", {}))
        if config.get("is_b_all"):
            preferred = config.get("curve_variant")
            variants = [preferred] if preferred in variants else variants[:1]
        return [self.plot_confusion_matrix(variant) for variant in variants]

    @staticmethod
    def _display_model_name(name: Any) -> str:
        return str(name).replace("_", " ").strip().title()

    def _curve_arrays_from_file(self, relative: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        frame = pd.read_csv(self._base_file(relative))
        probability_columns = [c for c in frame.columns if c.startswith("proba__")]
        return (
            frame["y_true"].to_numpy(),
            frame[probability_columns].to_numpy(dtype=float),
            [c.split("proba__", 1)[1] for c in probability_columns],
        )

    def _curve_model_arrays(self) -> list[tuple[str, np.ndarray, np.ndarray, list[str]]]:
        config = self.manifest["classification"]
        curve_files = config.get("curve_files") or {}
        if curve_files:
            model_order = config.get("curve_model_order") or list(curve_files)
            return [
                (model, *self._curve_arrays_from_file(curve_files[model]))
                for model in model_order
                if model in curve_files
            ]

        # Backward compatibility for manifests created before per-model curve
        # data was persisted.
        relative = config.get("files", {}).get(config.get("curve_variant"))
        if not relative:
            raise KeyError("No saved classification curve data was found.")
        return [(config["selected_model"], *self._curve_arrays_from_file(relative))]

    def plot_pr_curve(self) -> Path:
        recall_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._curve_model_arrays():
            curves, scores = [], []
            for col, label in enumerate(labels):
                binary = (y_true.astype(str) == str(label)).astype(int)
                if np.unique(binary).size < 2:
                    continue
                precision, recall, _ = precision_recall_curve(binary, probabilities[:, col])
                order = np.argsort(recall)
                curves.append(np.interp(recall_grid, recall[order], precision[order]))
                scores.append(average_precision_score(binary, probabilities[:, col]))
            if curves:
                items.append((model, np.mean(curves, axis=0), float(np.mean(scores))))
        if not items:
            raise ValueError("PR curve requires at least one class with positive and negative samples.")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                recall_grid,
                curve,
                label=f"{display_name} (AUPRC={score:.3f})",
            )
        ax.set(xlabel="Recall", ylabel="Precision")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save(fig, "precision_recall_curve.pdf")

    def plot_roc_curve(self) -> Path:
        fpr_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._curve_model_arrays():
            curves, scores = [], []
            for col, label in enumerate(labels):
                binary = (y_true.astype(str) == str(label)).astype(int)
                if np.unique(binary).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(binary, probabilities[:, col])
                curves.append(np.interp(fpr_grid, fpr, tpr, left=0, right=1))
                scores.append(auc(fpr, tpr))
            if curves:
                items.append((model, np.mean(curves, axis=0), float(np.mean(scores))))
        if not items:
            raise ValueError("ROC curve requires at least one class with positive and negative samples.")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5")
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                fpr_grid,
                curve,
                label=f"{display_name} (AUROC={score:.3f})",
            )
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return self._save(fig, "roc_curve.pdf")

    def plot_pc_metadata_heatmaps(self) -> list[Path]:
        config = self.manifest.get("pca", {})
        frames = {
            "before": pd.read_csv(self._base_file(config["before_associations"]), index_col=0),
            "after": pd.read_csv(self._base_file(config["after_associations"]), index_col=0),
        }
        vmax = config.get("heatmap_vmax")
        if vmax is None:
            finite = [df.to_numpy(dtype=float) for df in frames.values()]
            maxima = [float(np.nanmax(x)) for x in finite if np.isfinite(x).any()]
            vmax = max(maxima) if maxima else 1.0
        vmax = max(float(vmax), 1.0)
        paths = []
        for state, frame in frames.items():
            if frame.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, 1.8 * frame.shape[1]), 0.5 * frame.shape[0] + 2))
            sns.heatmap(frame, annot=True, cmap="viridis", vmin=0, vmax=vmax, ax=ax)
            ax.set(xlabel="Metadata variables", ylabel="Principal components", title=f"PC associations — {state} correction")
            fig.tight_layout()
            paths.append(self._save(fig, f"pc_metadata_associations_{state}.pdf"))
        return paths

    def plot_pca_scatter(self, state: str) -> Path:
        if state not in {"before", "after"}:
            raise ValueError("state must be 'before' or 'after'")
        config = self.manifest["pca"]
        frame = pd.read_csv(self._base_file(config[f"{state}_scores"]), index_col=0)
        if "PC1" not in frame or "PC2" not in frame:
            raise ValueError(f"Saved {state} PCA data does not contain both PC1 and PC2.")
        cohort = config.get("cohort_column")
        if not cohort or cohort not in frame.columns:
            raise KeyError("No cohort metadata column was saved for PCA scatter plotting.")
        variance = json.loads(self._base_file(config["explained_variance"]).read_text(encoding="utf-8"))[state]
        x_label = f"PC1 ({variance[0] * 100:.1f}%)" if len(variance) > 0 else "PC1"
        y_label = f"PC2 ({variance[1] * 100:.1f}%)" if len(variance) > 1 else "PC2"
        frame[cohort] = frame[cohort].astype("string").fillna("NA")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=frame, x="PC1", y="PC2", hue=cohort, s=45, ax=ax)
        ax.set(xlabel=x_label, ylabel=y_label, title=f"PCA by {cohort} — {state} correction")
        fig.tight_layout()
        return self._save(fig, f"pca_pc1_pc2_{state}.pdf")

    def plot_pca_scatters(self) -> list[Path]:
        return [self.plot_pca_scatter("before"), self.plot_pca_scatter("after")]

    def generate_all(self) -> list[Path]:
        paths: list[Path] = []
        if self.manifest.get("classification"):
            paths.extend(self.plot_confusion_matrices())
            paths.extend([self.plot_pr_curve(), self.plot_roc_curve()])
        if self.manifest.get("pca"):
            paths.extend(self.plot_pc_metadata_heatmaps())
            if self.manifest["pca"].get("cohort_column"):
                paths.extend(self.plot_pca_scatters())
        return paths
