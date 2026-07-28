from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

from omiclass.plotting.display_names import display_model_name

CLASS_DISPLAY_NAMES = {
    "Her2": "HER2",
    "her2": "HER2",
    "Ph": "Ph-like",
    "ph": "Ph-like",
    "ph-like": "Ph-like",
    "PAX5alt": "PAX5alt",
    "iAMP21": "iAMP21",
    "TCF3-PBX1": "TCF3-PBX1",
    "ETV6-RUNX1": "ETV6-RUNX1",
    "mutated": "M-CLL",
    "unmutated": "U-CLL",
}

def display_class_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return CLASS_DISPLAY_NAMES.get(text) or CLASS_DISPLAY_NAMES.get(text.casefold(), text)



class PlottingFramework:
    """Regenerate publication plots from ``pdf_plots/base_files`` only.

    The pipeline writes compact CSV/JSON inputs through the ``save_*`` methods.
    Loading and plotting those files never touches an HTML report or a model.
    """

    MANIFEST = "manifest.json"

    def __init__(
        self,
        run_dir: str | Path,
        output_dir: str | Path | None = None,
        uncorrected_run_dir: str | Path | None = None,
        transparent: bool = False,
        output_format: str = "pdf",
    ):
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        run_dir = Path(run_dir)
        self.run_dir = run_dir.parent if run_dir.name == "pdf_plots" else run_dir
        self.pdf_dir = run_dir if run_dir.name == "pdf_plots" else run_dir / "pdf_plots"
        self.base_dir = self.pdf_dir / "base_files"
        self.output_dir = Path(output_dir) if output_dir else self.pdf_dir / "publication"
        self.uncorrected_run_dir = Path(uncorrected_run_dir) if uncorrected_run_dir else None
        self.transparent = bool(transparent)
        self.output_format = str(output_format).lower().lstrip(".")
        if self.output_format not in {"pdf", "png"}:
            raise ValueError("output_format must be 'pdf' or 'png'.")
        try:
            self.manifest = self._read_manifest(self.base_dir)
        except FileNotFoundError:
            if not ((self.run_dir / "train_artifacts" / "tables").exists() or (self.run_dir / "leaderboard.csv").exists()):
                raise
            self.manifest = {}
        self.selected_model = self.manifest.get("classification", {}).get("selected_model")

    @staticmethod
    def _base_dir_from_run(run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        pdf_dir = run_dir if run_dir.name == "pdf_plots" else run_dir / "pdf_plots"
        return pdf_dir / "base_files"

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
    def _class_label_key(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

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
        curve_source: str | None = None,
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
                "curve_source": curve_source,
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

    def _base_file(self, relative: str, base_dir: Path | None = None) -> Path:
        return (base_dir or self.base_dir) / relative

    def _save(self, fig, filename: str, show: bool = False) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = (self.output_dir / filename).with_suffix(f".{self.output_format}")
        fig.savefig(
            path,
            format=self.output_format,
            dpi=300,
            bbox_inches="tight",
            transparent=self.transparent,
        )
        if show:
            plt.show()
        plt.close(fig)
        return path

    @staticmethod
    def _plot_title(default: str, title: str | None = None) -> str:
        return default if title is None else title

    @staticmethod
    def _preferred_classification_variant(
        config: Mapping[str, Any],
        preferred: Sequence[str] = (),
    ) -> str:
        files = config.get("files", {})
        for variant in preferred:
            if variant in files:
                return variant
        curve_variant = config.get("curve_variant")
        if curve_variant in files:
            return str(curve_variant)
        if files:
            return next(iter(files))
        raise KeyError("No saved classification data was found.")

    def _classification_frame(
        self,
        variant: str,
        *,
        manifest: Mapping[str, Any] | None = None,
        base_dir: Path | None = None,
    ) -> pd.DataFrame:
        config = (manifest or self.manifest).get("classification", {})
        relative = config.get("files", {}).get(variant)
        if not relative:
            raise KeyError(f"No saved classification data for variant '{variant}'.")
        return pd.read_csv(self._base_file(relative, base_dir))

    def plot_confusion_matrix(
        self,
        variant: str | None = None,
        show: bool = False,
        *,
        run_dir: str | Path | None = None,
        label: str | None = None,
        preferred_variants: Sequence[str] = (),
        title: str | None = None,
    ) -> Path:
        if run_dir is None:
            manifest = self.manifest
            base_dir = self.base_dir
        else:
            base_dir = self._base_dir_from_run(run_dir)
            manifest = self._read_manifest(base_dir)

        config = manifest["classification"]
        variant = variant or self._preferred_classification_variant(config, preferred_variants)
        frame = self._classification_frame(variant, manifest=manifest, base_dir=base_dir)
        labels = sorted(set(frame["y_true"]).union(frame["y_pred"]))
        cm = confusion_matrix(frame["y_true"], frame["y_pred"], labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        names = config.get("class_name_map", {})
        display = [
            display_class_label(names.get(self._class_label_key(x), self._class_label_key(x)))
            for x in labels
        ]
        plot_label = label or variant

        fig, ax = plt.subplots(figsize=(6 + 0.35 * len(labels), 4.8 + 0.35 * len(labels)))
        sns.heatmap(normalized, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=display, yticklabels=display, cbar_kws={"label": "Row-normalized (%)"}, ax=ax)
        default_title = f"{self._display_model_name(config['selected_model'])} — {plot_label}"
        ax.set(xlabel="Predicted", ylabel="True", title=self._plot_title(default_title, title))
        ax.tick_params(axis="x", rotation=45 if config.get("is_b_all") else 0)
        fig.tight_layout()
        return self._save(fig, f"confusion_matrix_{self._safe_name(plot_label)}.pdf", show=show)

    def plot_confusion_matrices(
        self,
        uncorrected_run_dir: str | Path | None = None,
        show: bool = False,
        title: str | None = None,
        corrected_title: str | None = None,
        uncorrected_title: str | None = None,
    ) -> list[Path]:
        config = self.manifest.get("classification", {})
        variants = list(config.get("files", {}))
        if config.get("is_b_all"):
            preferred = config.get("curve_variant")
            variants = [preferred] if preferred in variants else variants[:1]
            return [self.plot_confusion_matrix(variant, show=show, title=title) for variant in variants]

        uncorrected_run_dir = uncorrected_run_dir or self.uncorrected_run_dir
        if uncorrected_run_dir is not None:
            corrected_variant = self._preferred_classification_variant(
                config,
                preferred=("corrected", "holdout_test"),
            )
            return [
                self.plot_confusion_matrix(
                    corrected_variant,
                    label="corrected",
                    show=show,
                    title=corrected_title or title,
                ),
                self.plot_confusion_matrix(
                    run_dir=uncorrected_run_dir,
                    label="uncorrected",
                    preferred_variants=("uncorrected", "corrected", "holdout_test"),
                    show=show,
                    title=uncorrected_title or title,
                ),
            ]

        def variant_title(variant: str) -> str | None:
            if len(variants) <= 1:
                return title
            if variant == "uncorrected":
                return uncorrected_title or title
            if variant in {"corrected", "holdout_test"}:
                return corrected_title or title
            return title

        return [
            self.plot_confusion_matrix(
                variant,
                show=show,
                title=variant_title(str(variant)),
            )
            for variant in variants
        ]

    @staticmethod
    def _display_model_name(name: Any) -> str:
        return display_model_name(name)

    def _is_selected_model(self, model: Any) -> bool:
        return self.selected_model is not None and str(model) == str(self.selected_model)

    def _sort_curve_items(
        self,
        items: list[tuple[Any, np.ndarray, float]],
    ) -> list[tuple[Any, np.ndarray, float]]:
        return [
            item
            for _, item in sorted(
                enumerate(items),
                key=lambda indexed: (
                    -indexed[1][2],
                    0 if self._is_selected_model(indexed[1][0]) else 1,
                    indexed[0],
                ),
            )
        ]

    def _style_curve_legend(self, ax, items: Sequence[tuple[Any, np.ndarray, float]]):
        legend = ax.legend(fontsize=8)
        if legend is None:
            return None
        for text, (model, _, _) in zip(legend.get_texts(), items):
            if self._is_selected_model(model):
                text.set_fontweight("bold")
        return legend

    @staticmethod
    def _model_color_map(models: Sequence[Any]) -> dict[str, Any]:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not colors:
            colors = list(plt.get_cmap("tab10").colors)
        ordered_models = sorted({str(model) for model in models})
        return {
            model: colors[idx % len(colors)]
            for idx, model in enumerate(ordered_models)
        }

    @staticmethod
    def _interpolate_pr_curve(
        recall_grid: np.ndarray,
        recall: np.ndarray,
        precision: np.ndarray,
    ) -> np.ndarray:
        order = np.argsort(recall)
        recall_s = recall[order]
        precision_s = precision[order]
        unique_recall, first_idx = np.unique(recall_s, return_index=True)
        return np.interp(recall_grid, unique_recall, precision_s[first_idx])

    @staticmethod
    def _read_table(path: str | Path, *, index_col: int | str | None = None) -> pd.DataFrame:
        path = Path(path)
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep, index_col=index_col, low_memory=False)

    def _selected_model_from_outputs(self, model_name: str | None = None) -> str:
        if model_name:
            return str(model_name)
        if self.selected_model:
            return str(self.selected_model)

        leaderboard_path = self.run_dir / "leaderboard.csv"
        if leaderboard_path.exists():
            leaderboard = self._read_table(leaderboard_path)
            if "model" in leaderboard.columns:
                candidates = leaderboard
                if "status" in candidates.columns:
                    ok = candidates["status"].astype(str).str.casefold().eq("ok")
                    if ok.any():
                        candidates = candidates.loc[ok]
                if not candidates.empty:
                    return str(candidates.iloc[0]["model"])

        tables_dir = self.run_dir / "train_artifacts" / "tables"
        metric_paths = sorted(tables_dir.glob("*_test_metrics.csv"))
        metric_paths = [
            path
            for path in metric_paths
            if not path.name.endswith("_threshold_optimized_test_metrics.csv")
        ]
        if len(metric_paths) == 1:
            return metric_paths[0].name[: -len("_test_metrics.csv")]

        raise FileNotFoundError(
            "Could not determine the selected model from the PDF manifest, "
            f"{leaderboard_path}, or {tables_dir}."
        )

    def _holdout_table_path(
        self,
        model_name: str,
        table: str,
        *,
        threshold_optimized: bool = False,
        run_dir: str | Path | None = None,
    ) -> Path:
        if table not in {"test_metrics", "per_class"}:
            raise ValueError("table must be 'test_metrics' or 'per_class'")

        root = Path(run_dir) if run_dir is not None else self.run_dir
        tables_dir = root / "train_artifacts" / "tables"
        suffix = (
            f"_threshold_optimized_{table}.csv"
            if threshold_optimized
            else f"_{table}.csv"
        )
        path = tables_dir / f"{model_name}{suffix}"
        if path.exists():
            return path

        available = sorted(path.name for path in tables_dir.glob(f"*{suffix}"))
        raise FileNotFoundError(
            f"Holdout {table.replace('_', ' ')} file not found for model '{model_name}': {path}. "
            f"Available matching files: {available}"
        )

    @staticmethod
    def _mean_metric(frame: pd.DataFrame, column: str) -> float:
        if column not in frame.columns:
            return float("nan")
        values = pd.to_numeric(frame[column], errors="coerce")
        return float(values.mean()) if values.notna().any() else float("nan")

    def selected_model_holdout_test_scores(
        self,
        model_name: str | None = None,
        *,
        threshold_optimized: bool = False,
    ) -> pd.DataFrame:
        """Return selected-model holdout accuracy, macro precision, recall and F1.

        Values are read from the existing ``train_artifacts/tables`` output
        files. Precision, recall and F1 are macro averages over the saved
        per-class table; F1 prefers the saved ``f1_macro`` value when present.
        """
        selected = self._selected_model_from_outputs(model_name)
        metrics_path = self._holdout_table_path(
            selected,
            "test_metrics",
            threshold_optimized=threshold_optimized,
        )
        per_class_path = self._holdout_table_path(
            selected,
            "per_class",
            threshold_optimized=threshold_optimized,
        )

        metrics = self._read_table(metrics_path)
        if metrics.empty:
            raise ValueError(f"Holdout test metrics file is empty: {metrics_path}")
        row = metrics.iloc[0]
        per_class = self._read_table(per_class_path)

        f1 = pd.to_numeric(pd.Series([row.get("f1_macro")]), errors="coerce").iloc[0]
        if pd.isna(f1):
            f1 = self._mean_metric(per_class, "f1")

        scores = pd.DataFrame(
            [
                {
                    "model": selected,
                    "n_test": row.get("n_test", np.nan),
                    "accuracy": row.get("accuracy", np.nan),
                    "precision": self._mean_metric(per_class, "precision"),
                    "recall": self._mean_metric(per_class, "recall"),
                    "f1": f1,
                }
            ]
        )
        for column in ["n_test", "accuracy", "precision", "recall", "f1"]:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")
        return scores

    def _paired_uncorrected_framework(self) -> "PlottingFramework":
        if self.uncorrected_run_dir is None:
            raise ValueError(
                "uncorrected_run_dir is not set; cannot compare corrected and uncorrected results."
            )
        return PlottingFramework(
            run_dir=self.uncorrected_run_dir,
            output_dir=self.output_dir,
            transparent=self.transparent,
            output_format=self.output_format,
        )

    def _classification_variant_frame_for_comparison(
        self,
        *,
        preferred_variants: Sequence[str],
        label: str,
    ) -> pd.DataFrame:
        config = self.manifest.get("classification", {})
        variant = self._preferred_classification_variant(config, preferred_variants)
        frame = self._classification_frame(variant)
        missing = {"y_true", "y_pred"} - set(frame.columns)
        if missing:
            raise KeyError(
                f"Saved {label} classification data is missing required columns: {sorted(missing)}"
            )
        return frame

    @staticmethod
    def _format_model_summary(scores: pd.DataFrame) -> str:
        if scores.empty:
            return "model unavailable, n_test unavailable"
        row = scores.iloc[0]
        model = row.get("model", "")
        n_test = pd.to_numeric(pd.Series([row.get("n_test")]), errors="coerce").iloc[0]
        n_label = "NA" if pd.isna(n_test) else f"{n_test:.0f}"
        return f"{PlottingFramework._display_model_name(model)} (n_test={n_label})"

    @staticmethod
    def _stable_label_order(
        labels: Sequence[Any],
        class_order: Sequence[Any] | None = None,
        class_name_map: Mapping[str, str] | None = None,
    ) -> list[Any]:
        labels_list = list(labels)
        if class_order is not None:
            class_name_map = class_name_map or {}
            labels_by_key: dict[str, Any] = {}
            for label in labels_list:
                label_key = PlottingFramework._class_label_key(label)
                labels_by_key.setdefault(label_key, label)
                display_name = class_name_map.get(label_key)
                if display_name:
                    labels_by_key.setdefault(str(display_name), label)
                    labels_by_key.setdefault(display_class_label(display_name), label)

            requested = []
            for requested_label in class_order:
                matched = labels_by_key.get(str(requested_label))
                if matched is None:
                    matched = labels_by_key.get(display_class_label(requested_label))
                if matched in labels_list and matched not in requested:
                    requested.append(matched)
            remaining = [label for label in labels_list if label not in requested]
            return requested + sorted(remaining, key=lambda value: str(value))

        def key(value: Any) -> tuple[int, float, str]:
            try:
                return (0, float(value), str(value))
            except (TypeError, ValueError):
                return (1, 0.0, str(value))

        return sorted(labels_list, key=key)

    @staticmethod
    def _class_display_name(label: Any, class_name_map: Mapping[str, str]) -> str:
        key = PlottingFramework._class_label_key(label)
        return display_class_label(class_name_map.get(key, key))

    def _per_class_recall_from_predictions(
        self,
        frame: pd.DataFrame,
        *,
        labels: Sequence[Any],
    ) -> dict[Any, float]:
        y_true = frame["y_true"]
        y_pred = frame["y_pred"]
        recalls: dict[Any, float] = {}
        for label in labels:
            true_mask = y_true.astype(str).eq(str(label))
            support = int(true_mask.sum())
            if support == 0:
                recalls[label] = float("nan")
            else:
                correct = y_pred.loc[true_mask].astype(str).eq(str(label)).sum()
                recalls[label] = float(correct / support)
        return recalls

    def plot_corrected_vs_uncorrected_classification_results(
        self,
        model_name: str | None = None,
        *,
        class_order: Sequence[Any] | None = None,
        threshold_optimized: bool = False,
        show: bool = False,
        title: str | None = None,
    ) -> list[Path]:
        """Compare corrected and uncorrected selected-model classification results.

        The macro-performance plot uses the saved holdout metric tables. The
        per-class recall plot recomputes recall directly from saved
        ``y_true``/``y_pred`` rows in ``pdf_plots/base_files/classification``.
        """
        uncorrected_framework = self._paired_uncorrected_framework()

        corrected_scores = self.selected_model_holdout_test_scores(
            model_name=model_name,
            threshold_optimized=threshold_optimized,
        )
        uncorrected_scores = uncorrected_framework.selected_model_holdout_test_scores(
            model_name=None,
            threshold_optimized=threshold_optimized,
        )

        corrected_frame = self._classification_variant_frame_for_comparison(
            preferred_variants=("corrected", "holdout_test"),
            label="corrected",
        )
        saved_corrected_model = self.manifest.get("classification", {}).get("selected_model")
        if (
            model_name is not None
            and saved_corrected_model is not None
            and str(model_name) != str(saved_corrected_model)
        ):
            raise ValueError(
                "Per-class recall requires saved y_true/y_pred rows for the requested corrected model. "
                f"The saved classification predictions are for '{saved_corrected_model}', not '{model_name}'."
            )
        uncorrected_frame = uncorrected_framework._classification_variant_frame_for_comparison(
            preferred_variants=("uncorrected", "holdout_test", "corrected"),
            label="uncorrected",
        )

        paths: list[Path] = []
        metrics = ["accuracy", "precision", "recall", "f1"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
        corrected_values = corrected_scores.iloc[0][metrics].astype(float).to_numpy() * 100.0
        uncorrected_values = uncorrected_scores.iloc[0][metrics].astype(float).to_numpy() * 100.0

        x = np.arange(len(metrics))
        width = 0.34
        fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
        bars_uncorrected = ax.bar(
            x - width / 2,
            uncorrected_values,
            width,
            label="Uncorrected",
            color="#8c8c8c",
        )
        bars_corrected = ax.bar(
            x + width / 2,
            corrected_values,
            width,
            label="Corrected",
            color="#2f6f9f",
        )

        finite_values = np.concatenate(
            [
                corrected_values[np.isfinite(corrected_values)],
                uncorrected_values[np.isfinite(uncorrected_values)],
            ]
        )
        ax.set_ylim(75, 103)
        value_offset = 0.6
        delta_offset = 1.5
        for bars, values in (
            (bars_uncorrected, uncorrected_values),
            (bars_corrected, corrected_values),
        ):
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + value_offset,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        for bar, corrected, uncorrected in zip(
            bars_corrected,
            corrected_values,
            uncorrected_values,
        ):
            if not (np.isfinite(corrected) and np.isfinite(uncorrected)):
                continue
            delta = corrected - uncorrected
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                corrected + delta_offset,
                f"{delta:+.1f} pp",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#1f4e73" if delta >= 0 else "#9b2d2d",
            )

        corrected_summary = self._format_model_summary(corrected_scores)
        uncorrected_summary = self._format_model_summary(uncorrected_scores)
        ax.set(
            xticks=x,
            xticklabels=metric_labels,
            ylabel="Score (%)",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title(
            self._plot_title("Corrected vs Uncorrected Performance", title),
            fontsize=13,
            pad=16,
        )
        ax.legend(frameon=False, loc="upper right")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        paths.append(
            self._save(
                fig,
                "corrected_vs_uncorrected_macro_performance.pdf",
                show=show,
            )
        )

        raw_labels = list(
            pd.unique(
                pd.concat(
                    [
                        corrected_frame["y_true"],
                        corrected_frame["y_pred"],
                        uncorrected_frame["y_true"],
                        uncorrected_frame["y_pred"],
                    ],
                    ignore_index=True,
                )
            )
        )
        class_name_map = {
            **uncorrected_framework.manifest.get("classification", {}).get("class_name_map", {}),
            **self.manifest.get("classification", {}).get("class_name_map", {}),
        }
        labels = self._stable_label_order(
            raw_labels,
            class_order=class_order,
            class_name_map=class_name_map,
        )
        if not labels:
            raise ValueError("No class labels were found in the saved classification predictions.")

        uncorrected_recalls = uncorrected_framework._per_class_recall_from_predictions(
            uncorrected_frame,
            labels=labels,
        )
        corrected_recalls = self._per_class_recall_from_predictions(
            corrected_frame,
            labels=labels,
        )
        labels = sorted(
            labels,
            key=lambda label: (
                -(
                    corrected_recalls[label] - uncorrected_recalls[label]
                    if np.isfinite(corrected_recalls[label])
                    and np.isfinite(uncorrected_recalls[label])
                    else float("-inf")
                ),
                str(label),
            ),
        )

        def spread_positions(values: Sequence[float]) -> list[float]:
            finite_values = [float(value) for value in values if np.isfinite(value)]
            if not finite_values:
                return [float(value) for value in values]
            min_gap = min(6.0, 90.0 / max(len(finite_values), 1))
            indexed = sorted(
                [(idx, float(value)) for idx, value in enumerate(values)],
                key=lambda item: item[1],
            )
            adjusted = [float(value) for value in values]
            last = -np.inf
            for idx, value in indexed:
                if not np.isfinite(value):
                    continue
                adjusted[idx] = min(102.0, max(value, last + min_gap))
                last = adjusted[idx]
            overflow = adjusted[indexed[-1][0]] - 102.0 if indexed else 0.0
            if overflow > 0:
                for idx, _ in indexed:
                    adjusted[idx] = max(-2.0, adjusted[idx] - overflow)
            return adjusted

        left_values = [uncorrected_recalls[label] for label in labels]
        right_values = [corrected_recalls[label] for label in labels]
        left_label_y = spread_positions([value * 100.0 for value in left_values])
        right_label_y = spread_positions([value * 100.0 for value in right_values])
        recall_ylim = (70, 106)
        label_ymax = recall_ylim[1] - 2.5

        fig, ax = plt.subplots(
            figsize=(9.4, max(5.8, 0.58 * len(labels) + 3.2)),
            constrained_layout=True,
        )
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for idx, label in enumerate(labels):
            left = uncorrected_recalls[label]
            right = corrected_recalls[label]
            delta = right - left if np.isfinite(left) and np.isfinite(right) else np.nan
            left_pct = left * 100.0 if np.isfinite(left) else left
            right_pct = right * 100.0 if np.isfinite(right) else right
            delta_pct = delta * 100.0 if np.isfinite(delta) else delta
            display = self._class_display_name(label, class_name_map)
            color = color_cycle[idx % len(color_cycle)] if color_cycle else None
            ax.plot([0, 1], [left_pct, right_pct], marker="o", linewidth=1.7, color=color)
            left_x = -0.045 - 0.012 * (idx % 2)
            right_x = 1.045 + 0.018 * (idx % 3)
            if np.isfinite(left_pct):
                ax.text(
                    left_x,
                    min(left_label_y[idx], label_ymax),
                    f"{left_pct:.1f}%",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=color,
                )
            if np.isfinite(right_pct):
                ax.text(
                    right_x,
                    min(right_label_y[idx], label_ymax),
                    f"{display} {right_pct:.1f}% ({delta_pct:+.1f} pp)",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=color,
                )

        ax.set(
            xlim=(-0.24, 1.52),
            ylim=recall_ylim,
            xticks=[0, 1],
            xticklabels=["Uncorrected", "Corrected"],
            ylabel="Recall (%)",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_title("Per-Class Recall: Corrected vs Uncorrected", fontsize=13, pad=16)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.grid(axis="x", color="0.94", linewidth=0.8)
        ax.set_axisbelow(True)
        paths.append(
            self._save(
                fig,
                "corrected_vs_uncorrected_per_class_recall.pdf",
                show=show,
            )
        )

        return paths

    def print_selected_model_holdout_test_scores(
        self,
        model_name: str | None = None,
        *,
        threshold_optimized: bool = False,
        digits: int = 3,
        uncorr: bool = False,
    ) -> pd.DataFrame:
        """Print selected-model holdout test scores and return them as a DataFrame.

        If uncorr=True, also print the best/selected model from self.uncorrected_run_dir.
        """
        corrected_scores = self.selected_model_holdout_test_scores(
            model_name=model_name,
            threshold_optimized=threshold_optimized,
        )
        corrected_scores.insert(0, "run", "corrected")

        scores = corrected_scores

        if uncorr:
            if self.uncorrected_run_dir is None:
                raise ValueError(
                    "uncorrected_run_dir is not set; cannot print uncorrected scores."
                )

            uncorrected_framework = PlottingFramework(
                run_dir=self.uncorrected_run_dir,
                output_dir=self.output_dir,
                transparent=self.transparent,
                output_format=self.output_format,
            )

            uncorrected_scores = uncorrected_framework.selected_model_holdout_test_scores(
                model_name=None,  # use best/selected model from the uncorrected run
                threshold_optimized=threshold_optimized,
            )
            uncorrected_scores.insert(0, "run", "uncorrected")

            scores = pd.concat(
                [corrected_scores, uncorrected_scores],
                ignore_index=True,
            )

        label = "threshold-optimized holdout" if threshold_optimized else "holdout"
        print(f"Selected model {label} test scores (macro precision/recall/F1):")

        float_formatter = lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"

        print(
            scores.to_string(
                index=False,
                formatters={
                    "model": self._display_model_name,
                    "n_test": lambda value: "" if pd.isna(value) else f"{value:.0f}",
                    "accuracy": float_formatter,
                    "precision": float_formatter,
                    "recall": float_formatter,
                    "f1": float_formatter,
                },
            )
        )

        return scores
    
    def print_model_holdout_test_scores(self, model_name: str | None = None, *, threshold_optimized: bool = False, digits: int = 4) -> pd.DataFrame:
        """Print selected-model holdout test scores and return them as a DataFrame."""
        scores = self.selected_model_holdout_test_scores(
            model_name=model_name,
            threshold_optimized=threshold_optimized,
        )
        label = "threshold-optimized holdout" if threshold_optimized else "holdout"
        print(f"Selected model {label} test scores (macro precision/recall/F1):")
        float_formatter = lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        print(
            scores.to_string(
                index=False,
                formatters={
                    "model": self._display_model_name,
                    "n_test": lambda value: "" if pd.isna(value) else f"{value:.0f}",
                    "accuracy": float_formatter,
                    "precision": float_formatter,
                    "recall": float_formatter,
                    "f1": float_formatter,
                },
            )
        )
        return scores

    def _curve_arrays_from_file(self, relative: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        frame = pd.read_csv(self._base_file(relative))
        probability_columns = [c for c in frame.columns if c.startswith("proba__")]
        return (
            frame["y_true"].to_numpy(),
            frame[probability_columns].to_numpy(dtype=float),
            [c.split("proba__", 1)[1] for c in probability_columns],
        )

    def _prediction_model_name(self, model_name: str | None = None) -> str | None:
        if model_name:
            return str(model_name)

        pred_dir = self.run_dir / "survival_prediction"
        htmls = sorted(pred_dir.glob("*_survival_prediction.html"))
        if htmls:
            suffix = "_survival_prediction"
            stem = htmls[0].stem
            return stem[: -len(suffix)] if stem.endswith(suffix) else stem

        selected = self.manifest.get("classification", {}).get("selected_model")
        if selected:
            return str(selected)

        models = sorted((self.run_dir / "train_artifacts").glob("*.joblib"))
        return models[0].stem if models else None

    def _prediction_model_paths(self, model_name: str | None = None) -> list[Path]:
        model_dir = self.run_dir / "train_artifacts"
        if model_name is not None:
            path = model_dir / f"{model_name}.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Prediction model artifact not found: {path}")
            return [path]

        paths = sorted(model_dir.glob("*.joblib"))
        if paths:
            return paths

        selected = self._prediction_model_name()
        if selected is None:
            raise FileNotFoundError(f"No .joblib model artifacts found in {model_dir}.")
        path = model_dir / f"{selected}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Prediction model artifact not found: {path}")
        return [path]

    def _load_truth_labels(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
    ) -> pd.Series:
        frame = self._read_table(true_label_path)
        if sample_col is not None:
            if sample_col not in frame.columns:
                raise KeyError(f"sample_col '{sample_col}' was not found in {true_label_path}.")
            frame = frame.set_index(sample_col)
        else:
            sample_candidates = {"sample", "sample_id", "sampleid", "patient", "patient_id", "id"}
            first_col = str(frame.columns[0]).casefold() if len(frame.columns) else ""
            if first_col in sample_candidates:
                frame = frame.set_index(frame.columns[0])
            else:
                indexed = self._read_table(true_label_path, index_col=0)
                if label_col is None or label_col in indexed.columns or indexed.shape[1] == 1:
                    frame = indexed

        if label_col is None:
            candidates = ["y_true", "true_label", "label", "class", "cluster", "subtype"]
            label_col = next((col for col in candidates if col in frame.columns), None)
            if label_col is None and frame.shape[1] == 1:
                label_col = str(frame.columns[0])
        if label_col is None or label_col not in frame.columns:
            raise KeyError(
                "Could not determine the true-label column. Pass label_col='...' "
                f"for {true_label_path}. Available columns: {list(frame.columns)}"
            )

        labels = frame[label_col].dropna()
        labels.index = labels.index.map(str)
        return labels

    def _align_truth_to_probability_labels(
        self,
        truth: pd.Series,
        probability_labels: Sequence[Any],
    ) -> pd.Series:
        labels_as_str = {str(label) for label in probability_labels}
        out = truth.astype("string")
        if set(out.dropna().astype(str)).intersection(labels_as_str):
            return out.astype(str)

        class_name_map = self.manifest.get("classification", {}).get("class_name_map", {})
        reverse: dict[str, str] = {}
        for key, value in class_name_map.items():
            reverse.setdefault(str(value), str(key))
            reverse.setdefault(display_class_label(value), str(key))
        mapped = out.astype(str).map(reverse)
        if mapped.notna().any():
            return mapped.fillna(out.astype(str))
        return out.astype(str)

    def _prediction_matrix_for_models(self) -> pd.DataFrame:
        pred_dir = self.run_dir / "survival_prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = self.run_dir / "limma_re-correction" / "limma_corrected_expression.csv"
        prediction_path = pred_dir / "model_predictions.csv"
        if not matrix_path.exists():
            raise FileNotFoundError(
                "Could not rebuild prediction probabilities from this run; missing "
                f"{matrix_path}"
            )

        from omiclass.utils.io import load_matrix

        expr = load_matrix(matrix_path)
        X = expr.T
        X.index = X.index.map(str)
        if prediction_path.exists():
            pred_index = self._read_table(prediction_path, index_col=0).index.map(str)
            X = X.loc[[sample for sample in pred_index if sample in X.index]]
        return X

    def _prediction_probability_frames(
        self,
        model_name: str | None = None,
    ) -> list[tuple[str, pd.DataFrame]]:
        pred_dir = self.run_dir / "survival_prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)
        proba_dir = pred_dir / "model_probabilities"
        proba_dir.mkdir(parents=True, exist_ok=True)

        import joblib

        X_base = self._prediction_matrix_for_models()
        frames: list[tuple[str, pd.DataFrame]] = []
        for model_path in self._prediction_model_paths(model_name):
            current_model_name = model_path.stem
            proba_path = proba_dir / f"{self._safe_name(current_model_name)}.csv"
            if proba_path.exists():
                proba = self._read_table(proba_path, index_col=0)
                proba.index = proba.index.map(str)
                frames.append((current_model_name, proba))
                continue

            model = joblib.load(model_path)
            X = X_base.copy()
            from omiclass.train.cv_engine import _bulkformer_representation_from_model
            representation = _bulkformer_representation_from_model(model)
            if representation is not None:
                expected = list(representation.feature_names_in_)
                missing = [feature for feature in expected if feature not in X.columns]
                extra = [feature for feature in X.columns if feature not in set(expected)]
                strict_extra = getattr(representation, "strategy", "full") != "full"
                if missing or (extra and strict_extra):
                    raise ValueError(
                        f"Embedding schema mismatch; missing={missing[:5]}, extra={extra[:5]}."
                    )
                X = X.loc[:, expected]
            expected_features = getattr(model, "feature_names_in_", None)
            if expected_features is None:
                expected_features = getattr(model, "_feature_names_in", None)
            if expected_features is not None and representation is None:
                X = X.reindex(columns=expected_features)
                X = X.fillna(0.0)

            if not hasattr(model, "predict_proba"):
                raise AttributeError(f"Model '{current_model_name}' does not expose predict_proba().")
            probabilities = model.predict_proba(X)
            columns = list(getattr(model, "classes_", range(probabilities.shape[1])))
            proba = pd.DataFrame(probabilities, index=X.index, columns=columns)
            proba.to_csv(proba_path)
            frames.append((current_model_name, proba))
        return frames

    def _prediction_probability_frame(self, model_name: str | None = None) -> tuple[str, pd.DataFrame]:
        frames = self._prediction_probability_frames(model_name)
        if not frames:
            raise FileNotFoundError("Could not build prediction probability frames.")
        return frames[0]

    def _prediction_model_arrays(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
        model_name: str | None = None,
    ) -> list[tuple[str, np.ndarray, np.ndarray, list[str]]]:
        truth = self._load_truth_labels(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
        )
        pred_dir = self.base_dir / "prediction" / "curves"
        pred_dir.mkdir(parents=True, exist_ok=True)
        curve_files: dict[str, str] = {}
        arrays: list[tuple[str, np.ndarray, np.ndarray, list[str]]] = []

        for current_model_name, proba in self._prediction_probability_frames(model_name):
            common = proba.index.intersection(truth.index)
            if common.empty:
                continue

            aligned_proba = proba.loc[common]
            aligned_truth = self._align_truth_to_probability_labels(
                truth.loc[common],
                aligned_proba.columns,
            )

            frame = pd.DataFrame({"sample_id": common, "y_true": aligned_truth.to_numpy()})
            labels = [str(col) for col in aligned_proba.columns]
            for col in aligned_proba.columns:
                frame[f"proba__{col}"] = aligned_proba[col].to_numpy(dtype=float)
            filename = f"{self._safe_name(current_model_name)}.csv"
            frame.to_csv(pred_dir / filename, index=False)
            curve_files[current_model_name] = f"prediction/curves/{filename}"
            arrays.append(
                (
                    current_model_name,
                    aligned_truth.to_numpy(),
                    aligned_proba.to_numpy(dtype=float),
                    labels,
                )
            )

        if not arrays:
            raise ValueError(
                "No overlapping sample IDs between prediction probabilities and true labels."
            )

        prediction = dict(self.manifest.get("prediction", {}))
        prediction.update(
            {
                "curve_files": curve_files,
                "curve_model_order": list(curve_files),
                "curve_source": "clean_and_predict",
                "truth_label_path": str(true_label_path),
            }
        )
        self._update_manifest(self.base_dir, prediction=prediction)
        self.manifest = self._read_manifest(self.base_dir)

        return arrays

    def _recover_holdout_curve_files(self) -> bool:
        """Backfill all-model holdout curve CSVs from existing training-run artifacts."""
        try:
            cfg_path = self.run_dir / "config.freeze.yaml"
            split_path = self.run_dir / "split.json"
            matrix_path = self.run_dir / "limma_correction" / "limma_corrected_expression.csv"
            model_dir = self.run_dir / "train_artifacts"
            missing = [
                str(path)
                for path in (cfg_path, split_path, matrix_path, model_dir)
                if not path.exists()
            ]
            if missing:
                print(
                    "[PlottingFramework] Could not recover all-model holdout PR/ROC "
                    f"base files; missing: {', '.join(missing)}"
                )
                return False

            import joblib
            import yaml
            from omiclass.config.schema import Config
            from omiclass.eval.holdout import evaluate_holdout
            from omiclass.utils.io import load_labels, load_matrix

            cfg = Config.model_validate(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})
            split = json.loads(split_path.read_text(encoding="utf-8"))
            test_index = [str(sample) for sample in split.get("test_index", [])]
            if not test_index:
                print("[PlottingFramework] Could not recover holdout curves; split.json has no test_index.")
                return False

            expr = load_matrix(matrix_path)
            X_all = expr.T
            X_all.index = X_all.index.map(str)
            y_all, common_index = load_labels(
                cfg.project.meta_path,
                cfg.project.cluster_column,
                X_all.index,
                cfg=cfg,
            )
            common_index = common_index.map(str)
            X_all = X_all.loc[common_index]
            y_all.index = y_all.index.map(str)
            y_all = y_all.loc[common_index]

            available_test = [sample for sample in test_index if sample in X_all.index]
            if not available_test:
                print("[PlottingFramework] Could not recover holdout curves; no test samples match the saved matrix.")
                return False

            X_test = X_all.loc[available_test]
            y_test = y_all.loc[available_test]
            model_paths = sorted(model_dir.glob("*.joblib"))
            if not model_paths:
                print(f"[PlottingFramework] Could not recover holdout curves; no .joblib models in {model_dir}.")
                return False

            curve_dir = self.base_dir / "classification" / "curves"
            curve_dir.mkdir(parents=True, exist_ok=True)
            curve_files: dict[str, str] = {}

            for model_path in model_paths:
                model_name = model_path.stem
                report = evaluate_holdout(joblib.load(model_path), X_test, y_test)
                payload = report["plot_payload"]
                y_true = np.asarray(y_test).reshape(-1)
                probabilities = np.asarray(payload["y_proba"])
                labels = list(payload.get("labels") or sorted(np.unique(y_true.astype(int))))
                if probabilities.ndim == 1:
                    probabilities = np.column_stack([1.0 - probabilities, probabilities])
                if probabilities.shape[1] != len(labels):
                    labels = [str(i) for i in range(probabilities.shape[1])]

                frame = pd.DataFrame({"y_true": y_true})
                for col, label in enumerate(labels):
                    frame[f"proba__{label}"] = probabilities[:, col]
                filename = f"{self._safe_name(model_name)}.csv"
                frame.to_csv(curve_dir / filename, index=False)
                curve_files[model_name] = f"classification/curves/{filename}"

            classification = dict(self.manifest.get("classification", {}))
            classification.update(
                {
                    "curve_files": curve_files,
                    "curve_model_order": list(curve_files),
                    "curve_source": "holdout_test_recovered",
                }
            )
            self._update_manifest(self.base_dir, classification=classification)
            self.manifest = self._read_manifest(self.base_dir)
            print(
                "[PlottingFramework] Recovered all-model holdout PR/ROC base files "
                f"for {len(curve_files)} models."
            )
            return True
        except Exception as exc:
            print(
                "[PlottingFramework] Could not recover all-model holdout PR/ROC "
                f"base files from existing run artifacts: {exc}"
            )
            return False

    def _curve_model_arrays(self) -> list[tuple[str, np.ndarray, np.ndarray, list[str]]]:
        config = self.manifest["classification"]
        curve_files = config.get("curve_files") or {}
        curve_source = str(config.get("curve_source") or "")
        needs_holdout_recovery = not curve_files or (
            bool(curve_files) and not curve_source.startswith("holdout_test")
        )
        if needs_holdout_recovery and self._recover_holdout_curve_files():
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

    def plot_pr_curve(self, show: bool = False, title: str | None = None) -> Path:
        recall_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._curve_model_arrays():
            if len(labels) == 2:
                positive_col = 1
                positive_label = labels[positive_col]
                binary = (y_true.astype(str) == str(positive_label)).astype(int)
                if np.unique(binary).size < 2:
                    continue
                precision, recall, _ = precision_recall_curve(binary, probabilities[:, positive_col])
                curve = self._interpolate_pr_curve(recall_grid, recall, precision)
                score = average_precision_score(binary, probabilities[:, positive_col])
                items.append((model, curve, float(score)))
            else:
                curves, scores = [], []
                for col, label in enumerate(labels):
                    binary = (y_true.astype(str) == str(label)).astype(int)
                    if np.unique(binary).size < 2:
                        continue
                    precision, recall, _ = precision_recall_curve(binary, probabilities[:, col])
                    curves.append(self._interpolate_pr_curve(recall_grid, recall, precision))
                    scores.append(average_precision_score(binary, probabilities[:, col]))
                if curves:
                    items.append((model, np.mean(curves, axis=0), float(np.mean(scores))))
        if not items:
            raise ValueError("PR curve requires at least one class with positive and negative samples.")
        items = self._sort_curve_items(items)
        is_binary = all(len(labels) == 2 for _, _, _, labels in self._curve_model_arrays())
        metric_label = "AP" if is_binary else "macro AP"
        default_title = "Precision–Recall Curve" if is_binary else "Macro Precision–Recall Curve"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        color_map = self._model_color_map([model for model, _, _ in items])
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                recall_grid,
                curve,
                color=color_map[str(model)],
                label=f"{display_name} ({metric_label}={score:.3f})",
            )
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=self._plot_title(default_title, title),
        )
        self._style_curve_legend(ax, items)
        fig.tight_layout()
        return self._save(fig, "precision_recall_curve.pdf", show=show)

    def plot_roc_curve(self, show: bool = False, title: str | None = None) -> Path:
        fpr_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._curve_model_arrays():
            if len(labels) == 2:
                positive_col = 1
                positive_label = labels[positive_col]
                binary = (y_true.astype(str) == str(positive_label)).astype(int)
                if np.unique(binary).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(binary, probabilities[:, positive_col])
                curve = np.interp(fpr_grid, fpr, tpr, left=0, right=1)
                score = auc(fpr, tpr)
                items.append((model, curve, float(score)))
            else:
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
        items = self._sort_curve_items(items)
        is_binary = all(len(labels) == 2 for _, _, _, labels in self._curve_model_arrays())
        metric_label = "AUROC" if is_binary else "macro AUROC"
        default_title = "ROC Curve" if is_binary else "Macro ROC Curve"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5")
        color_map = self._model_color_map([model for model, _, _ in items])
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                fpr_grid,
                curve,
                color=color_map[str(model)],
                label=f"{display_name} ({metric_label}={score:.3f})",
            )
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=self._plot_title(default_title, title),
        )
        self._style_curve_legend(ax, items)
        fig.tight_layout()
        return self._save(fig, "roc_curve.pdf", show=show)

    def plot_prediction_pr_curve(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
        model_name: str | None = None,
        show: bool = False,
        title: str | None = None,
    ) -> Path:
        recall_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._prediction_model_arrays(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
            model_name=model_name,
        ):
            if len(labels) == 2:
                positive_col = 1
                positive_label = labels[positive_col]
                binary = (y_true.astype(str) == str(positive_label)).astype(int)
                if np.unique(binary).size < 2:
                    continue
                precision, recall, _ = precision_recall_curve(binary, probabilities[:, positive_col])
                curve = self._interpolate_pr_curve(recall_grid, recall, precision)
                score = average_precision_score(binary, probabilities[:, positive_col])
                items.append((model, curve, float(score)))
            else:
                curves, scores = [], []
                for col, label in enumerate(labels):
                    binary = (y_true.astype(str) == str(label)).astype(int)
                    if np.unique(binary).size < 2:
                        continue
                    precision, recall, _ = precision_recall_curve(binary, probabilities[:, col])
                    curves.append(self._interpolate_pr_curve(recall_grid, recall, precision))
                    scores.append(average_precision_score(binary, probabilities[:, col]))
                if curves:
                    items.append((model, np.mean(curves, axis=0), float(np.mean(scores))))
        if not items:
            raise ValueError("Prediction PR curve requires at least one class with positive and negative samples.")

        items = self._sort_curve_items(items)
        is_binary = all(
            len(labels) == 2
            for _, _, _, labels in self._prediction_model_arrays(
                true_label_path,
                label_col=label_col,
                sample_col=sample_col,
                model_name=model_name,
            )
        )
        metric_label = "AP" if is_binary else "macro AP"
        default_title = "Precision–Recall Curve" if is_binary else "Macro Precision–Recall Curve"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        color_map = self._model_color_map([model for model, _, _ in items])
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                recall_grid,
                curve,
                color=color_map[str(model)],
                label=f"{display_name} ({metric_label}={score:.3f})",
            )
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            title=self._plot_title(default_title, title),
        )
        self._style_curve_legend(ax, items)
        fig.tight_layout()
        return self._save(fig, "prediction_precision_recall_curve.pdf", show=show)

    def plot_prediction_roc_curve(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
        model_name: str | None = None,
        show: bool = False,
        title: str | None = None,
    ) -> Path:
        fpr_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._prediction_model_arrays(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
            model_name=model_name,
        ):
            if len(labels) == 2:
                positive_col = 1
                positive_label = labels[positive_col]
                binary = (y_true.astype(str) == str(positive_label)).astype(int)
                if np.unique(binary).size < 2:
                    continue
                fpr, tpr, _ = roc_curve(binary, probabilities[:, positive_col])
                curve = np.interp(fpr_grid, fpr, tpr, left=0, right=1)
                score = auc(fpr, tpr)
                items.append((model, curve, float(score)))
            else:
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
            raise ValueError("Prediction ROC curve requires at least one class with positive and negative samples.")

        items = self._sort_curve_items(items)
        is_binary = all(
            len(labels) == 2
            for _, _, _, labels in self._prediction_model_arrays(
                true_label_path,
                label_col=label_col,
                sample_col=sample_col,
                model_name=model_name,
            )
        )
        metric_label = "AUROC" if is_binary else "macro AUROC"
        default_title = "ROC Curve" if is_binary else "Macro ROC Curve"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5")
        color_map = self._model_color_map([model for model, _, _ in items])
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                fpr_grid,
                curve,
                color=color_map[str(model)],
                label=f"{display_name} ({metric_label}={score:.3f})",
            )
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=self._plot_title(default_title, title),
        )
        self._style_curve_legend(ax, items)
        fig.tight_layout()
        return self._save(fig, "prediction_roc_curve.pdf", show=show)

    def plot_prediction_curves(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
        model_name: str | None = None,
        show: bool = False,
        title: str | None = None,
    ) -> list[Path]:
        return [
            self.plot_prediction_pr_curve(
                true_label_path,
                label_col=label_col,
                sample_col=sample_col,
                model_name=model_name,
                show=show,
                title=title,
            ),
            self.plot_prediction_roc_curve(
                true_label_path,
                label_col=label_col,
                sample_col=sample_col,
                model_name=model_name,
                show=show,
                title=title,
            ),
        ]

    def plot_pc_metadata_heatmaps(
        self,
        show: bool = False,
        filename_prefix: str = "",
        column_order: Sequence[str] | None = None,
        title: str | None = None,
        corrected_title: str | None = None,
        uncorrected_title: str | None = None,

    ) -> list[Path]:
        config = self.manifest.get("pca", {})
        frames = {
            "before": pd.read_csv(self._base_file(config["before_associations"]), index_col=0),
            "after": pd.read_csv(self._base_file(config["after_associations"]), index_col=0),
        }
        if column_order is not None:
            requested = [col for col in column_order if col in next(iter(frames.values())).columns]

            if not requested:
                raise ValueError(
                    "None of the requested column_order columns were found in the PC association heatmap."
                )

            frames = {
                state: frame[[col for col in column_order if col in frame.columns]]
                for state, frame in frames.items()
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
            default_title = f"PC associations — {state} correction"
            state_title = (
                uncorrected_title
                if state == "before"
                else corrected_title
                if state == "after"
                else None
            )
            ax.set(
                xlabel="Metadata variables",
                ylabel="Principal components",
                title=self._plot_title(default_title, state_title or title),
            )
            fig.tight_layout()
            paths.append(self._save(fig, f"{filename_prefix}pc_metadata_associations_{state}.pdf", show=show))
        return paths

    def plot_pca_scatter(
        self,
        state: str,
        show: bool = False,
        filename_prefix: str = "",
        colored_by: Union[pd.Series, None] = None,
        title: str | None = None,
    ) -> Path:
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
        if colored_by is not None:
            sns.scatterplot(data=frame, x="PC1", y="PC2", hue=colored_by, s=45, ax=ax)
            default_title = f"PC1 vs PC2 colored by {colored_by.name or 'metadata'} — {state} correction"
        else:
            sns.scatterplot(data=frame, x="PC1", y="PC2", hue=cohort, s=45, ax=ax)
            default_title = f"PC1 vs PC2 colored by {cohort} — {state} correction"
        ax.set(xlabel=x_label, ylabel=y_label, title=self._plot_title(default_title, title))
        fig.tight_layout()
        return self._save(fig, f"{filename_prefix}pca_pc1_pc2_{state}.pdf", show=show)

    def plot_pca_scatters(
        self,
        show: bool = False,
        filename_prefix: str = "",
        colored_by: Union[pd.Series, None] = None,
        title: str | None = None,
        corrected_title: str | None = None,
        uncorrected_title: str | None = None,
    ) -> list[Path]:
        if colored_by is not None:
            return [
                self.plot_pca_scatter("before", show=show, filename_prefix=filename_prefix, title=uncorrected_title or title),
                self.plot_pca_scatter("after", show=show, filename_prefix=filename_prefix, title=corrected_title or title),
                self.plot_pca_scatter("before", show=show, filename_prefix=filename_prefix, colored_by=colored_by, title=uncorrected_title or title),
                self.plot_pca_scatter("after", show=show, filename_prefix=filename_prefix, colored_by=colored_by, title=corrected_title or title),
            ]
        return [
            self.plot_pca_scatter("before", show=show, filename_prefix=filename_prefix, title=uncorrected_title or title),
            self.plot_pca_scatter("after", show=show, filename_prefix=filename_prefix, title=corrected_title or title),
        ]

    def plot_prediction_pca_scatters(self, show: bool = False, title: str | None = None) -> list[Path]:
        return self.plot_pca_scatters(show=show, filename_prefix="prediction_", title=title)

    def plot_prediction_pc_metadata_heatmaps(self, show: bool = False, title: str | None = None) -> list[Path]:
        return self.plot_pc_metadata_heatmaps(show=show, filename_prefix="prediction_", title=title)

    def generate_prediction_plots(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
        model_name: str | None = None,
        show: bool = False,
    ) -> list[Path]:
        paths = self.plot_prediction_curves(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
            model_name=model_name,
            show=show,
        )
        if self.manifest.get("pca"):
            paths.extend(self.plot_prediction_pc_metadata_heatmaps(show=show))
            if self.manifest["pca"].get("cohort_column"):
                paths.extend(self.plot_prediction_pca_scatters(show=show))
        return paths

    def generate_all(self, show: bool = False) -> list[Path]:
        paths: list[Path] = []
        if self.manifest.get("classification"):
            paths.extend(self.plot_confusion_matrices())
            paths.extend([self.plot_pr_curve(), self.plot_roc_curve(show=show)])
        if self.manifest.get("pca"):
            paths.extend(self.plot_pc_metadata_heatmaps(show=show))
            if self.manifest["pca"].get("cohort_column"):
                paths.extend(self.plot_pca_scatters(show=show))
        return paths
