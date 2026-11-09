# omiclass/pipeline/steps/train_eval_step.py
from __future__ import annotations

from pathlib import Path
from pprint import pp
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import json
from typing import TYPE_CHECKING

from omiclass.utils.pipeline import make_or_load_split, maybe_log1p_expression
from omiclass.data.processor import DataProcessor
from omiclass.train.trainer import Trainer
from omiclass.train.selection import rank_candidates
from omiclass.utils.splits import apply_split
from omiclass.train.preprocessing import TrainingPreprocessor
from omiclass.eval.holdout import evaluate_holdout, evaluate_holdout_with_thresholds, write_holdout_csvs
from omiclass.eval.thresholds import learn_multiclass_thresholds
from omiclass.eval.prediction import align_probabilities
from omiclass.reporting.per_model_report import finalize_report_for_model
from omiclass.plotting.plotter import FinalSummaryPlotter
from omiclass.reporting.final_summary_report import render_final_summary_html
from omiclass.utils.toolbox import _build_class_name_map
from omiclass.eval.feature_reduction_runner import FeatureReductionStep
from omiclass.utils.pipeline import apply_batch_holdout
from omiclass.train.tune import resolve_cv_folds
from omiclass.config.loaders import freeze_config
from omiclass.plotting.framework import PlottingFramework


if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext


class TrainAndEvalStep:
    name = "train_and_eval"
    
    @staticmethod
    def _is_higher_better_metric(metric_name: str) -> bool:
        m = str(metric_name or "").lower()
        if m.startswith("neg_"):
            return True
        lower_better_tokens = ("loss", "error", "rmse", "mae", "mse")
        return not any(tok in m for tok in lower_better_tokens)

    @staticmethod
    def _is_b_all_run(ctx: RunContext) -> bool:
        name = str(getattr(ctx.cfg.project, "name", "")).lower()
        compact = "".join(ch for ch in name if ch.isalnum())
        return "ball" in compact

    def should_run(self, ctx: RunContext) -> bool:
        if ctx.opts.mode == "clean_and_predict":
            return False
        if ctx.opts.run_training is not None:
            return ctx.opts.run_training
        return ctx.opts.mode in ("full", "train-only", "de-train", "batch-train")

    def run(self, ctx: RunContext) -> None:
        dp = DataProcessor(ctx.cfg)

        # Choose which expression matrix to use: corrected vs original
        using_batch_corrected_matrix = "X_corrected" in ctx.cache
        if using_batch_corrected_matrix:
            X_full_raw = ctx.cache["X_corrected"]
        else:
            X_full_raw = ctx.cache["full_expr"]

        gene_list = ctx.cache["gene_list"]

        X_full_raw = apply_batch_holdout(
            X=X_full_raw,
            cfg=ctx.cfg,
            out_dir=ctx.out_dir,
        )

        # Build X, y for training
        X, y = dp.build_xy_from_corrected(
            X_full_raw=X_full_raw,
            gene_list=gene_list,
            drop_constant_cols=False,
            clean_finite=False,
        )
        y = ctx.cache['labels'].reindex(X.index)
        if y.isna().any():
            raise ValueError('Training population differs from the resolved labelled population')

        # Train/test split
        Xtr, Xte, ytr, yte = apply_split(X, y, ctx.cache['split'])
        split_path = ctx.cache['split_json']
        resolve_cv_folds(ctx.cfg, ytr)
        # Retain fit-time raw counts and covariates only for training IDs.
        meta = ctx.cache.get('de_metadata')
        if ctx.cache.get('reference_deferred'):
            from omiclass.utils.io import load_matrix
            meta = load_matrix(ctx.cfg.project.meta_path, numeric=False)
            numeric_outcome = pd.api.types.is_numeric_dtype(meta[ctx.cfg.project.cluster_column])
            meta = meta.drop(columns=[ctx.cfg.project.cluster_column], errors='ignore')
            meta.attrs['outcome_numeric'] = numeric_outcome
            meta.attrs['outcome_values'] = {value: key for key, value in ctx.cache['label_mapping'].items()}
        raw = ctx.cache.get('raw_expression', X_full_raw).loc[:, Xtr.index]
        prep = TrainingPreprocessor(ctx.cfg, raw_expression=raw,
            metadata=meta.loc[Xtr.index] if meta is not None else None,
            de=ctx.cache.get('de_enabled', False), covariates=ctx.cache.get('de_covariates'),
            already_logged=using_batch_corrected_matrix or ctx.cache.get('_did_raw_count_norm', False),
            cptac=ctx.cache.get('cptac_deferred', False) and not using_batch_corrected_matrix,
            reference=ctx.cache.get('reference_deferred', False),
            work_dir=ctx.out_dir / 'preprocessing/fit-tmp')
        # Train on training set
        train_dir = ctx.out_dir / "train_artifacts"
        trainer = Trainer(ctx.cfg, artifacts_dir=train_dir, preprocessor=prep)
        summary = trainer.run(Xtr, ytr)
        preprocessing_dir = ctx.out_dir / 'preprocessing'
        preprocessing_dir.mkdir(exist_ok=True)
        (preprocessing_dir / 'scoped.json').write_text(json.dumps(dict(
            version=1, train_ids=Xtr.index.tolist(), test_ids=Xte.index.tolist(),
            input_mode=ctx.opts.input_mode,
            fitted_inside_estimators=True, de=prep.de, cptac=prep.cptac, reference=prep.reference,
            correction_transductive=using_batch_corrected_matrix,
            batch_input_log1p=ctx.cache.get('batch_input_log1p', False),
            frozen_raw_count_normalization=(preprocessing_dir / 'raw_count_gene_lengths.joblib').is_file())))
        # 4) Save leaderboard
        leaderboard_csv = ctx.out_dir / "leaderboard.csv"
        summary.metrics.to_csv(leaderboard_csv, index=False)

        print(f"[ 5][Train_and_eval]: Leaderboard saved at: {leaderboard_csv}")
        # 5) Evaluate all final models on test
        test_rows: list[Dict[str, Any]] = []
        test_reports: Dict[str, Path] = {}
        per_model_payload: Dict[str, Dict[str, np.ndarray]] = {}
        per_model_train_payload: Dict[str, Dict[str, np.ndarray]] = {}
        per_model_threshold_payload: Dict[str, Dict[str, np.ndarray]] = {}
        labels_from_first = [str(label) for label in np.unique(ytr)]
        train_label_index = np.array(sorted(np.unique(ytr.astype(int))), dtype=int)

        # Train payload from OOF CV outputs (same source used to create summary.metrics).
        for model_name in summary.metrics.get("model", pd.Series(dtype=str)).tolist():
            cvres = trainer.cv_results.get(model_name)
            if cvres is None:
                continue
            y_pred = np.asarray(getattr(cvres, "oof_pred", None))
            if y_pred.size == 0:
                continue
            y_proba = getattr(cvres, "oof_proba", None)
            if y_proba is None:
                continue  # Hard predictions are not probability estimates for ROC/PR plots.
            per_model_train_payload[model_name] = {
                "y_pred": y_pred,
                "y_proba": align_probabilities(y_proba,
                    getattr(cvres, 'classes', None) if getattr(cvres, 'classes', None) is not None
                    else np.unique(cvres.oof_true), train_label_index),
            }

        # Persist the training-only threshold decision before reading holdout labels.
        threshold_infos = {key: learn_multiclass_thresholds(trainer.cv_results[key])
            for key in summary.artifact_paths if key in trainer.cv_results}
        (train_dir / 'thresholds.json').write_text(json.dumps(threshold_infos, indent=2), encoding='utf-8')
        i = 1
        for key, model_path in summary.artifact_paths.items():
            i += 1
            run_dir = ctx.out_dir
            model = joblib.load(model_path)

            report = evaluate_holdout(model, Xte, yte)
            cvres = trainer.cv_results.get(key)
            threshold_info = threshold_infos.get(key)
            threshold_report = None
            if threshold_info is not None:
                threshold_report = evaluate_holdout_with_thresholds(
                    model=model,
                    X_test=Xte,
                    y_test=yte,
                    threshold_info=threshold_info,
                )

            paths = write_holdout_csvs(
                train_dir,
                key,
                metrics=report,
                threshold_metrics=threshold_report,
            )

            finalize_report_for_model(
                run_dir=run_dir,
                model_name=key,
                test_metrics_csv=run_dir / paths["test_metrics_csv"],
                per_class_csv=run_dir / paths["per_class_csv"],
                model_rel_path=ctx.out_dir / f"train_artifacts/{key}.joblib",
                cfg=ctx.cfg,
                threshold_metrics=threshold_report,

            )

            # regular final summary payload
            pp = report.get("plot_payload", {})
            if not labels_from_first:
                labels_from_first = list(pp.get("labels", []))
            per_model_payload[key] = {
                "y_pred": pp.get("y_pred"),
                "y_proba": (align_probabilities(pp['y_proba'], report['classes'], train_label_index)
                            if pp.get('y_proba') is not None else None),
            }

            # The OOF-defined derivative is always reported, irrespective of holdout F1.
            if threshold_report is not None:
                tpp = threshold_report.get("plot_payload", {})
                per_model_threshold_payload[key] = {
                    "y_pred": tpp.get("y_pred"),
                    "y_proba": align_probabilities(tpp['y_proba'], threshold_report['classes'], train_label_index),
                }

            test_rows.append({
                "model": key,
                "artifact": str(model_path),
                "n_test": report.get("n_test"),
                "accuracy": report.get("accuracy"),
                "f1_macro": report.get("f1_macro"),
                "f1_weighted": report.get("f1_weighted"),
                "threshold_accuracy": threshold_report.get("accuracy") if threshold_report else np.nan,
                "threshold_f1_macro": threshold_report.get("f1_macro") if threshold_report else np.nan,
                "threshold_f1_weighted": threshold_report.get("f1_weighted") if threshold_report else np.nan,
                "threshold_auroc_ovr_macro": threshold_report.get("auroc_ovr_macro") if threshold_report else np.nan,
                "threshold_logloss": threshold_report.get("logloss") if threshold_report else np.nan,
                "threshold_f1_macro_delta": (
                    threshold_report.get("f1_macro") - report.get("f1_macro")
                    if threshold_report and report.get("f1_macro") is not None
                    else np.nan
                ),
                "mean_true_class_proba": report.get("mean_true_class_proba", np.nan),
            })

        combined_test_csv = ctx.out_dir / "test_metrics.csv"
        final_test_res = pd.DataFrame(test_rows)
        final_test_res.to_csv(combined_test_csv, index=False)

        # Select best model ONLY from CV leaderboard (never from holdout ranking).
        scoring_metric = str(
            getattr(getattr(ctx.cfg.runtime, "tuning", None), "scoring_override", None)
            or ctx.cfg.runtime.scoring
        )
        cv_leaderboard, selected_metric_col = rank_candidates(
            summary.metrics, summary.artifact_paths, scoring_metric
        )
        best_model = str(cv_leaderboard.iloc[0]["model"])
        best_model_path = summary.artifact_paths[best_model]

        labels_for_plot = (
            labels_from_first
            if labels_from_first
            else [str(i) for i in sorted(np.unique(yte.astype(int)))]
        )

        class_name_map = _build_class_name_map(y, ctx.cfg)

        if ctx.cfg.PDF:
            is_b_all = self._is_b_all_run(ctx)
            run_variant = (
                "holdout_test"
                if is_b_all
                else ("corrected" if using_batch_corrected_matrix else "uncorrected")
            )
            selected_payload = per_model_payload[best_model]
            pdf_variants: Dict[str, Dict[str, Any]] = {
                run_variant: {
                    "sample_ids": Xte.index.to_numpy(),
                    "y_true": yte.to_numpy(),
                    "y_pred": selected_payload["y_pred"],
                    "y_proba": selected_payload["y_proba"],
                    "labels": train_label_index.copy(),
                }
            }

            PlottingFramework.save_classification_data(
                ctx.out_dir / "pdf_plots" / "base_files",
                variants=pdf_variants,
                selected_model=best_model,
                class_name_map=class_name_map,
                is_b_all=is_b_all,
                curve_variant=run_variant,
                curve_source="holdout_test",
                curve_models={
                    model_name: {
                        "y_true": yte.to_numpy(),
                        "y_proba": payload["y_proba"],
                        "labels": labels_from_first or sorted(np.unique(yte.astype(int))).copy(),
                    }
                    for model_name, payload in per_model_payload.items()
                },
            )
        
        plotter = FinalSummaryPlotter(
            labels=labels_for_plot,
            y_true=yte,
            per_model=per_model_payload,
            out_dir=ctx.out_dir,
            class_name_map=class_name_map,
            per_model_train=per_model_train_payload,
            y_true_train=ytr,
        )

        ctx.cache["split_json"] = split_path
        ctx.cache["leaderboard_csv"] = leaderboard_csv
        ctx.cache["best_model"] = best_model
        ctx.cache["best_model_path"] = best_model_path
        
        scope_path = preprocessing_dir / 'scoped.json'
        scope = json.loads(scope_path.read_text())
        scope.update(selected_model=best_model,
                     model_path=str(best_model_path.resolve().relative_to(ctx.out_dir.resolve())))
        scope_path.write_text(json.dumps(scope, indent=2))
        fr = getattr(ctx.cfg.runtime, "feature_reduction", None)
        if fr and fr.enabled:
            FeatureReductionStep().run(ctx)

        meta_payload = {
            "n_test": int(len(yte)),
            "k": len(labels_for_plot),
            "input_modality": str(getattr(ctx.cfg.project, "modality", "rnaseq")),
            "input_modality_label": (
                "CPTAC MS proteomics"
                if str(getattr(ctx.cfg.project, "modality", "rnaseq")) == "cptac_ms"
                else "RNA-seq"
            ),
            "cv_selected_model": best_model,
            "cv_selection_metric": selected_metric_col,
        }
        if ctx.cache.get("cptac_preprocessing"):
            meta_payload["cptac_preprocessing"] = ctx.cache.get("cptac_preprocessing")

        _ = render_final_summary_html(
            plotter=plotter,
            meta=meta_payload,
            ctx=ctx
        )

        threshold_plotter = None
        if per_model_threshold_payload:
            threshold_plotter = FinalSummaryPlotter(
                labels=labels_for_plot,
                y_true=yte,
                per_model=per_model_threshold_payload,
                out_dir=ctx.out_dir,
                class_name_map=class_name_map,
                per_model_train=per_model_train_payload,
                y_true_train=ytr,
            )

        meta_payload = {
            "train samples": int(len(ytr)),
            "test samples": int(len(yte)),
            "Number of classes": len(labels_for_plot),
            "Selected model": best_model.replace("_", " ").title(),
            "Selected metric": selected_metric_col,
            "Threshold optimized appendix": "yes" if threshold_plotter is not None else "no",
            "Feature reduction diagnostic": ctx.cache.get('feature_reduction_status', {}).get('reason', 'not requested'),
            "input_modality": str(getattr(ctx.cfg.project, "modality", "rnaseq")),
            "input_modality_label": (
                "CPTAC MS proteomics"
                if str(getattr(ctx.cfg.project, "modality", "rnaseq")) == "cptac_ms"
                else "RNA-seq"
            ),
        }

        if ctx.cache.get("cptac_preprocessing"):
            meta_payload["cptac_preprocessing"] = ctx.cache.get("cptac_preprocessing")

        _ = render_final_summary_html(
            plotter=plotter,
            threshold_plotter=threshold_plotter,
            include_threshold_appendix=bool(threshold_plotter is not None),
            meta=meta_payload,
            ctx=ctx,
        )

        # stash for orchestrator result
        ctx.cache["combined_test_csv"] = combined_test_csv
        ctx.cache["labels_for_plot"] = list(labels_for_plot)

        freeze_config(ctx.cfg, ctx.out_dir)
