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

    def __init__(
        self,
        run_dir: str | Path,
        output_dir: str | Path | None = None,
        uncorrected_run_dir: str | Path | None = None,
    ):
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        run_dir = Path(run_dir)
        self.run_dir = run_dir.parent if run_dir.name == "pdf_plots" else run_dir
        self.pdf_dir = run_dir if run_dir.name == "pdf_plots" else run_dir / "pdf_plots"
        self.base_dir = self.pdf_dir / "base_files"
        self.output_dir = Path(output_dir) if output_dir else self.pdf_dir / "publication"
        self.uncorrected_run_dir = Path(uncorrected_run_dir) if uncorrected_run_dir else None
        self.manifest = self._read_manifest(self.base_dir)
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
        path = self.output_dir / filename
        fig.savefig(path, format="pdf", bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

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
        display = [names.get(str(int(x)) if float(x).is_integer() else str(x), str(x)) for x in labels]
        plot_label = label or variant

        fig, ax = plt.subplots(figsize=(6 + 0.35 * len(labels), 4.8 + 0.35 * len(labels)))
        sns.heatmap(normalized, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=display, yticklabels=display, cbar_kws={"label": "Row-normalized (%)"}, ax=ax)
        ax.set(xlabel="Predicted", ylabel="True", title=f"{config['selected_model']} — {plot_label}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return self._save(fig, f"confusion_matrix_{self._safe_name(plot_label)}.pdf", show=show)

    def plot_confusion_matrices(
        self,
        uncorrected_run_dir: str | Path | None = None,
        show: bool = False,
    ) -> list[Path]:
        config = self.manifest.get("classification", {})
        variants = list(config.get("files", {}))
        if config.get("is_b_all"):
            preferred = config.get("curve_variant")
            variants = [preferred] if preferred in variants else variants[:1]
            return [self.plot_confusion_matrix(variant, show=show) for variant in variants]

        uncorrected_run_dir = uncorrected_run_dir or self.uncorrected_run_dir
        if uncorrected_run_dir is not None:
            corrected_variant = self._preferred_classification_variant(
                config,
                preferred=("corrected", "holdout_test"),
            )
            return [
                self.plot_confusion_matrix(corrected_variant, label="corrected", show=show),
                self.plot_confusion_matrix(
                    run_dir=uncorrected_run_dir,
                    label="uncorrected",
                    preferred_variants=("uncorrected", "corrected", "holdout_test"),
                    show=show,
                ),
            ]

        return [self.plot_confusion_matrix(variant) for variant in variants]

    @staticmethod
    def _display_model_name(name: Any) -> str:
        return str(name).replace("_", " ").strip().title()

    def _is_selected_model(self, model: Any) -> bool:
        return self.selected_model is not None and str(model) == str(self.selected_model)

    @staticmethod
    def _sort_curve_items(
        items: list[tuple[Any, np.ndarray, float]],
    ) -> list[tuple[Any, np.ndarray, float]]:
        return sorted(items, key=lambda item: item[2], reverse=True)

    def _style_curve_legend(self, ax, items: Sequence[tuple[Any, np.ndarray, float]]):
        legend = ax.legend(fontsize=8)
        if legend is None:
            return None
        for text, (model, _, _) in zip(legend.get_texts(), items):
            if self._is_selected_model(model):
                text.set_fontweight("bold")
        return legend

    @staticmethod
    def _read_table(path: str | Path, *, index_col: int | str | None = None) -> pd.DataFrame:
        path = Path(path)
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        return pd.read_csv(path, sep=sep, index_col=index_col, low_memory=False)

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
        reverse = {str(v): str(k) for k, v in class_name_map.items()}
        mapped = out.astype(str).map(reverse)
        if mapped.notna().any():
            return mapped.fillna(out.astype(str))
        return out.astype(str)

    def _prediction_probability_frame(self, model_name: str | None = None) -> tuple[str, pd.DataFrame]:
        pred_dir = self.run_dir / "survival_prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)
        model_name = self._prediction_model_name(model_name)
        if not model_name:
            raise FileNotFoundError("Could not determine the prediction model name.")

        proba_path = pred_dir / "model_probabilities.csv"
        if proba_path.exists():
            proba = self._read_table(proba_path, index_col=0)
            proba.index = proba.index.map(str)
            return model_name, proba

        model_path = self.run_dir / "train_artifacts" / f"{model_name}.joblib"
        matrix_path = self.run_dir / "limma_re-correction" / "limma_corrected_expression.csv"
        prediction_path = pred_dir / "model_predictions.csv"
        missing = [str(path) for path in (model_path, matrix_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Could not rebuild prediction probabilities from this run; missing "
                + ", ".join(missing)
            )

        import joblib
        from omiclass.utils.io import load_matrix

        model = joblib.load(model_path)
        expr = load_matrix(matrix_path)
        X = expr.T
        X.index = X.index.map(str)
        if prediction_path.exists():
            pred_index = self._read_table(prediction_path, index_col=0).index.map(str)
            X = X.loc[[sample for sample in pred_index if sample in X.index]]
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is None:
            expected_features = getattr(model, "_feature_names_in", None)
        if expected_features is not None:
            X = X.reindex(columns=expected_features)
        X = X.fillna(0.0)

        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"Model '{model_name}' does not expose predict_proba().")
        probabilities = model.predict_proba(X)
        columns = list(getattr(model, "classes_", range(probabilities.shape[1])))
        proba = pd.DataFrame(probabilities, index=X.index, columns=columns)
        proba.to_csv(proba_path)
        return model_name, proba

    def _prediction_model_arrays(
        self,
        true_label_path: str | Path,
        *,
        label_col: str | None = None,
        sample_col: str | None = None,
        model_name: str | None = None,
    ) -> list[tuple[str, np.ndarray, np.ndarray, list[str]]]:
        model_name, proba = self._prediction_probability_frame(model_name)
        truth = self._load_truth_labels(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
        )
        common = proba.index.intersection(truth.index)
        if common.empty:
            raise ValueError(
                "No overlapping sample IDs between prediction probabilities and true labels."
            )

        proba = proba.loc[common]
        truth = self._align_truth_to_probability_labels(truth.loc[common], proba.columns)

        pred_dir = self.base_dir / "prediction" / "curves"
        pred_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({"sample_id": common, "y_true": truth.to_numpy()})
        labels = [str(col) for col in proba.columns]
        for col in proba.columns:
            frame[f"proba__{col}"] = proba[col].to_numpy(dtype=float)
        filename = f"{self._safe_name(model_name)}.csv"
        frame.to_csv(pred_dir / filename, index=False)

        prediction = dict(self.manifest.get("prediction", {}))
        prediction.update(
            {
                "curve_files": {model_name: f"prediction/curves/{filename}"},
                "curve_model_order": [model_name],
                "curve_source": "clean_and_predict",
                "truth_label_path": str(true_label_path),
            }
        )
        self._update_manifest(self.base_dir, prediction=prediction)
        self.manifest = self._read_manifest(self.base_dir)

        return [(model_name, truth.to_numpy(), proba.to_numpy(dtype=float), labels)]

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

    def plot_pr_curve(self, show: bool = False) -> Path:
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
        items = self._sort_curve_items(items)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(
                recall_grid,
                curve,
                label=f"{display_name} (AUPRC={score:.3f})",
            )
        ax.set(xlabel="Recall", ylabel="Precision")
        self._style_curve_legend(ax, items)
        fig.tight_layout()
        return self._save(fig, "precision_recall_curve.pdf", show=show)

    def plot_roc_curve(self, show: bool = False) -> Path:
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
        items = self._sort_curve_items(items)
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
    ) -> Path:
        recall_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._prediction_model_arrays(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
            model_name=model_name,
        ):
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
            raise ValueError("Prediction PR curve requires at least one class with positive and negative samples.")

        items = self._sort_curve_items(items)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(recall_grid, curve, label=f"{display_name} (AUPRC={score:.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Prediction set precision-recall")
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
    ) -> Path:
        fpr_grid = np.linspace(0, 1, 500)
        items = []
        for model, y_true, probabilities, labels in self._prediction_model_arrays(
            true_label_path,
            label_col=label_col,
            sample_col=sample_col,
            model_name=model_name,
        ):
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
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.5")
        for model, curve, score in items:
            display_name = self._display_model_name(model)
            ax.plot(fpr_grid, curve, label=f"{display_name} (AUROC={score:.3f})")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Prediction set ROC")
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
    ) -> list[Path]:
        return [
            self.plot_prediction_pr_curve(
                true_label_path,
                label_col=label_col,
                sample_col=sample_col,
                model_name=model_name,
                show=show,
            ),
            self.plot_prediction_roc_curve(
                true_label_path,
                label_col=label_col,
                sample_col=sample_col,
                model_name=model_name,
                show=show,
            ),
        ]

    def plot_pc_metadata_heatmaps(
        self,
        show: bool = False,
        filename_prefix: str = "",
    ) -> list[Path]:
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
            paths.append(self._save(fig, f"{filename_prefix}pc_metadata_associations_{state}.pdf", show=show))
        return paths

    def plot_pca_scatter(
        self,
        state: str,
        show: bool = False,
        filename_prefix: str = "",
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
        sns.scatterplot(data=frame, x="PC1", y="PC2", hue=cohort, s=45, ax=ax)
        ax.set(xlabel=x_label, ylabel=y_label, title=f"PCA by {cohort} — {state} correction")
        fig.tight_layout()
        return self._save(fig, f"{filename_prefix}pca_pc1_pc2_{state}.pdf", show=show)

    def plot_pca_scatters(
        self,
        show: bool = False,
        filename_prefix: str = "",
    ) -> list[Path]:
        return [
            self.plot_pca_scatter("before", show=show, filename_prefix=filename_prefix),
            self.plot_pca_scatter("after", show=show, filename_prefix=filename_prefix),
        ]

    def plot_prediction_pca_scatters(self, show: bool = False) -> list[Path]:
        return self.plot_pca_scatters(show=show, filename_prefix="prediction_")

    def plot_prediction_pc_metadata_heatmaps(self, show: bool = False) -> list[Path]:
        return self.plot_pc_metadata_heatmaps(show=show, filename_prefix="prediction_")

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
