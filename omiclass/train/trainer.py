# generic-classifier/train/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import joblib
import traceback
from skopt import BayesSearchCV
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression

from ..config.schema import Config
from ..models.registry import get_builder
from .cv_engine import nested_oof_cv, CVResult, calibrate_within_training, _safe_stratified_kfold
from .stacking_engine import train_oof_stacker
from .selection import rank_candidates, selection_summary
from .tune import build_rf_search_space, build_lgbm_param_grid, build_xgb_param_grid
from ..utils.imbalance import assess_imbalance, Severity
from ..utils.toolbox import _to_dict, _aggregate_best_rounds, sanitize_fit_kwargs_for_xgb, _build_class_name_map, unwrap_for_export
from ..reporting.per_model_report import report_cv
from ..utils.param_router import split_init_and_fit_kwargs
from .auto_config import apply_auto_config
from omiclass.errors.errors import TuningError 
from omiclass.errors.faild_model import _failed_row
from sklearn.pipeline import Pipeline as SkPipeline
from omiclass.features.bulkformer_reduction import build_embedding_representation
from omiclass.features.embedding_backend import embedding_backend, embedding_config, embedding_enabled
from .cv_engine import _bulkformer_representation_from_model
from omiclass.utils.surface_markers import apply_surface_marker_filter_with_min
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    ImbPipeline = tuple()
import xgboost as xgb
import numpy as np



@dataclass
class TrainSummary:
    metrics: pd.DataFrame
    artifact_paths: dict[str, Path]  # model_key -> saved model path

class Trainer:
    def __init__(self, cfg: Config, artifacts_dir: Union[str, Path], preprocessor=None):
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.class_name_map = None
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.final_models: dict[str, object] = {} 
        self.cv_results: dict[str, CVResult] = {}
        self.builders: dict[str, Any] = {}

    def _bulkformer_reduction_active(self) -> bool:
        return embedding_enabled(self.cfg)

    def _with_bulkformer_reduction(self, builder):
        def wrapped_builder():
            estimator = builder()
            if self._bulkformer_reduction_active():
                backend = embedding_backend(self.cfg)
                estimator = SkPipeline([
                    ('bulkformer_representation', build_embedding_representation(
                        embedding_config(self.cfg), int(self.cfg.runtime.random_state), backend)),
                    ('classifier', estimator)])
            if self._partition_active():
                from omiclass.train.scoped import PartitionClassifier
                estimator = PartitionClassifier(estimator, getattr(self, 'preprocessor', None),
                    self.cfg.runtime.feature_selection, seed=int(self.cfg.runtime.random_state),
                    inner_cv=int(self.cfg.runtime.tuning.inner_cv or 3))
            return estimator
        return wrapped_builder

    def _partition_active(self):
        return getattr(self, 'preprocessor', None) is not None or self.cfg.runtime.feature_selection.enabled

    def _reduction_space(self, space: dict) -> dict:
        prefix = ('estimator__' if self._partition_active() else '') + ('classifier__' if self._bulkformer_reduction_active() else '')
        return {prefix + key: value for key, value in space.items()}

    def _reduction_fit_kwargs(self, fit_kwargs: dict | None) -> dict:
        values = dict(fit_kwargs or {})
        prefix = ('estimator__' if self._partition_active() else '') + ('classifier__' if self._bulkformer_reduction_active() else '')
        return {prefix + key: value for key, value in values.items()}

    # Internal helpers
    def _safe_inner_cv(self, y: pd.Series, cv: int, seed: int):
        return _safe_stratified_kfold(y, cv, seed)
    
    def _tune_full(self, key: str, base_builder, X: pd.DataFrame, y: pd.Series, fit_kwargs: dict | None = None) -> dict:
        """Run BayesSearchCV on ALL data to get best_params_ for the FINAL model only."""
        if self._tune_iterations() == 0:
            return {}
        # Build search space and prefix
        if key.endswith("forest"):
            prefix = "" if key in ("plain_random_forest", "weighted_random_forest") else "rf__"
            rf_cfg = self.cfg.models.random_forest.defaults
            space = build_rf_search_space(X, rf_cfg.search_space, prefix=prefix)
        else:
            if "xgboost" in key:
                boost_cfg = self.cfg.models.xgboost
                space = build_xgb_param_grid(X, boost_cfg.param_grid)
            else:  # light_gbm
                boost_cfg = self.cfg.models.light_gbm
                space = build_lgbm_param_grid(X, boost_cfg.param_grid)

        space = self._reduction_space(space)
        tun = self.cfg.runtime.tuning
        runt = self.cfg.runtime

        inner = self._safe_inner_cv(y, cv=int(tun.inner_cv or 3), seed=int(runt.random_state))
        if inner == 0:
            # Cannot split safely; skip tuning
            return {}
        if self._partition_active() and runt.feature_selection.enabled:
            from omiclass.train.cv_engine import inner_select_recipe
            _, params, _, _ = inner_select_recipe(model_builder=base_builder, Xtr=X, ytr=y,
                tune_space=space, tune_iter=self._tune_iterations(), tune_scoring=runt.scoring,
                tune_inner_cv=int(tun.inner_cv or 3), seed=int(runt.random_state),
                n_jobs=int(runt.n_jobs), feature_cfg=runt.feature_selection, fit_kwargs=fit_kwargs)
            return params
            
        try:
            model = base_builder()
            search = BayesSearchCV(
                estimator=model,
                search_spaces=space,
                n_iter=int(runt.bayesian_iterations),
                scoring=str(getattr(getattr(runt, "tuning", None), "scoring_override", None) or runt.scoring),
                cv=inner,
                n_jobs=1, # avoid nested parallelism
                random_state=int(runt.random_state),
                refit=False,
                verbose=0
            )
            search.fit(X, y, **(fit_kwargs or {}))
            return search.best_params_
        except Exception as e:
            raise TuningError(f"Tuning failed for {key}: {e}") from e

    def _tune_iterations(self):
        runtime = self.cfg.runtime
        if not runtime.tuning.enabled or runtime.bayesian_iterations == 0:
            return 0
        if runtime.tuning.method != "bayes":
            raise TuningError("Only the existing Bayesian native search is implemented")
        return int(runtime.bayesian_iterations)
        
    def _save_genesets_from_cvres(self, model_key: str,cvres: Any, feature_selection_cfg: Any = None) -> Tuple[List[str], pd.DataFrame, CVResult]:
        out_dir = (self.artifacts_dir / "genesets" / model_key).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        per_fold = list(getattr(cvres, "per_fold_metrics", []) or [])
        n_folds = len(per_fold)

        fold_rows: List[Dict[str, Any]] = []
        fold_genes: List[List[str]] = []

        for i, m in enumerate(per_fold):
            genes = [str(g) for g in (m.get("selected_features") or [])]
            fold_genes.append(genes)
            fold_rows.append(
                {
                    "section": "fold",
                    "outer_fold": i,
                    "selected_k": m.get("n_features"),
                    "inner_best_score": m.get("fs_best_score"),
                    "fs_scoring": m.get("fs_scoring"),
                    "selected_genes": ";".join(genes),
                    "n_selected_genes": len(genes),
                }
            )

        fold_df = pd.DataFrame(fold_rows)

        all_genes = [g for genes in fold_genes for g in genes]
        counts = Counter(all_genes)
        ranked_counts = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))

        gene_rows: List[Dict[str, Any]] = []
        for gene, cnt in ranked_counts:
            gene_rows.append(
                {
                    "section": "gene",
                    "gene": gene,
                    "count_folds": int(cnt),
                    "freq_folds": (cnt / n_folds) if n_folds else 0.0,
                }
            )
        gene_df = pd.DataFrame(gene_rows)

        cfg = _to_dict(feature_selection_cfg)
        consensus_top_n: Optional[int] = cfg.get("consensus_top_n")
        consensus_min_frac: float = float(cfg.get("consensus_min_frac", 0.60))
        consensus_min_folds: Optional[int] = cfg.get("consensus_min_folds", 5)
        consensus_min_genes: int = cfg.get("consensus_min_genes", 20)

        if n_folds == 0:
            consensus_genes: List[str] = []
        else:
            if consensus_min_folds is None:
                consensus_min_folds = max(1, int(consensus_min_frac * n_folds + 1e-9))

            consensus_genes = [g for g, c in counts.items() if c >= consensus_min_folds]

            if isinstance(consensus_top_n, int) and consensus_top_n > 0:
                consensus_genes = [g for g, _ in ranked_counts[:consensus_top_n]]

            consensus_genes = sorted(consensus_genes, key=lambda g: (-counts[g], g))

            if len(consensus_genes) < consensus_min_genes:
                # fallback: take top genes by fold-frequency
                consensus_genes = [g for g, _ in ranked_counts[:consensus_min_genes]]

        if not gene_df.empty:
            gene_df["in_consensus"] = gene_df["gene"].isin(set(consensus_genes))

        consensus_meta = pd.DataFrame(
            [
                {
                    "section": "consensus_meta",
                    "n_folds": n_folds,
                    "consensus_min_frac": consensus_min_frac,
                    "consensus_min_folds": consensus_min_folds if n_folds else 0,
                    "consensus_top_n": consensus_top_n if consensus_top_n is not None else "",
                    "consensus_size": len(consensus_genes),
                }
            ]
        )

        consensus_path = out_dir / "consensus_genes.txt"
        consensus_path.write_text("\n".join(consensus_genes) + ("\n" if consensus_genes else ""))

        full_info = pd.concat([consensus_meta, fold_df, gene_df], ignore_index=True, sort=False)

        cvres.geneset_consensus_genes = consensus_genes
        cvres.geneset_full_info = full_info
        cvres.geneset_cfg_short = {
            "min_features": getattr(feature_selection_cfg, "min_features", None),
            "subset_fracs": getattr(feature_selection_cfg, "subset_fracs", None),
            "inner_splits": getattr(feature_selection_cfg, "inner_splits", None),
            "consensus_min_frac": getattr(feature_selection_cfg, "consensus_min_frac", None),
            "consensus_min_folds": getattr(feature_selection_cfg, "consensus_min_folds", None),
            "consensus_min_genes": getattr(feature_selection_cfg, "consensus_min_genes", 20),
            "consensus_top_n": getattr(feature_selection_cfg, "consensus_top_n", None),
        }

        return cvres, consensus_genes, full_info


    def _compose_raw_params(self, *, model_key: str, model_conf) -> dict:
        r = self.cfg.runtime
        gen = _to_dict(r).copy()
        for k in ("split","imbalance","tuning"):
            gen.pop(k, None)
        # We'll inject these explicitly to avoid accidental override:
        gen.pop("random_state", None)
        gen.pop("n_jobs", None)

        specific = _to_dict(getattr(model_conf, "init_params", {}))
        # add model-level boosting params
        if "xgboost" in model_key:
            if getattr(model_conf, "eval_metric", None) is not None:
                specific["eval_metric"] = getattr(model_conf, "eval_metric")

        if "light_gbm" in model_key:
            if getattr(model_conf, "eval_metric", None) is not None:
                specific["eval_metric"] = getattr(model_conf, "eval_metric")
            if getattr(model_conf, "metric", None) is not None:
                specific["metric"] = getattr(model_conf, "metric")

        merged = {
            **gen,
            **specific,
            "random_state": int(self.cfg.runtime.random_state),
            "n_jobs": int(self.cfg.runtime.n_jobs),
        }
        return merged

    def _final_fit_with_kwargs(self, est, X, y, fit_kwargs: dict | None = None):
        if fit_kwargs:
            est.fit(X, y, **fit_kwargs)
        else:
            est.fit(X, y)
        return est

    def _fit_final(self, builder, X: pd.DataFrame, y: pd.Series, *, calibrate: bool, method: str, cv: int, seed: int, fit_kwargs: dict | None = None, ensemble: bool = False,):
        base = builder()
        if not calibrate:
            return self._final_fit_with_kwargs(base, X, y, fit_kwargs=fit_kwargs)
        return calibrate_within_training(
            fitted_or_builder=base, Xtr_used=X, ytr=y, method=method,
            cv=cv, seed=seed, ensemble=ensemble, fit_kwargs=fit_kwargs,
        )

    def _save_model(self, model, filename: str) -> Path:
        export_model = unwrap_for_export(model)
        path = (self.artifacts_dir / filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(export_model, path)
        return path

    def _candidate_setup(self, key, X, y, model_conf=None):
        """Shared native candidate recipe, also used inside stacking partitions."""
        r = self.cfg.runtime
        is_rf = key.endswith('random_forest')
        if model_conf is None:
            model_conf = (self.cfg.models.random_forest.defaults if is_rf else
                          self.cfg.models.xgboost if 'xgboost' in key else self.cfg.models.light_gbm)
        raw = self._compose_raw_params(model_key=key, model_conf=model_conf)
        if is_rf:
            variant = getattr(self.cfg.models.random_forest.variants, key)
            if variant.class_weight is not None:
                raw['class_weight'] = variant.class_weight
        else:
            raw = apply_auto_config(key, raw, y)
            if key == 'weighted_light_gbm':
                raw['class_weight'] = 'balanced'
        init_params, fit_kwargs = split_init_and_fit_kwargs(key, raw)
        if 'xgboost' in key:
            init_params.update(random_state=int(r.random_state), seed=int(r.random_state), n_jobs=1, nthread=1)
        elif 'light_gbm' in key:
            init_params.update(random_state=int(r.random_state), seed=int(r.random_state),
                feature_fraction_seed=int(r.random_state), bagging_seed=int(r.random_state),
                data_random_seed=int(r.random_state), drop_seed=int(r.random_state),
                deterministic=True, force_col_wise=True, n_jobs=1)
        fit_kwargs.pop('early_stopping_rounds', None)
        fit_kwargs.pop('early_stopping', None)

        def builder():
            estimator = get_builder(key, random_state=int(r.random_state),
                n_jobs=int(r.n_jobs), override_params=init_params)()
            if not is_rf and 'n_jobs' in estimator.get_params(deep=False):
                estimator.set_params(n_jobs=1)
            return estimator

        if is_rf:
            prefix = '' if key in ('plain_random_forest', 'weighted_random_forest') else 'rf__'
            space = build_rf_search_space(X, model_conf.search_space, prefix=prefix)
        elif 'xgboost' in key:
            space = build_xgb_param_grid(X, model_conf.param_grid)
        else:
            space = build_lgbm_param_grid(X, model_conf.param_grid)
        return (self._with_bulkformer_reduction(builder), self._reduction_space(space),
                self._reduction_fit_kwargs(fit_kwargs), getattr(model_conf, 'calibration', None))

    def _fit_stacking_bases(self, X, y):
        """Select/refit the original three base families using this fit's labels."""
        r = self.cfg.runtime
        scoring = str(getattr(r.tuning, 'scoring_override', None) or r.scoring)
        allowed = self._model_keys_for_labels(y)
        families = dict(rf=[key for key in allowed if key.endswith('random_forest')],
                        xgb=[key for key in allowed if 'xgboost' in key],
                        lgbm=[key for key in allowed if 'light_gbm' in key])
        estimators, details = [], {}
        for family, keys in families.items():
            recipes, rows, failures = {}, [], {}
            for key in keys:
                try:
                    builder, space, kwargs, cal = self._candidate_setup(key, X, y)
                    recipes[key] = (builder, kwargs, cal)
                    # A single eligible variant has no model-selection decision.
                    score = {}
                    if len(keys) > 1:
                        cvres = nested_oof_cv(model_builder=builder, X=X, y=y,
                            outer_splits=int(r.cv_folds), seed=int(r.random_state),
                            tune_space=space, tune_iter=self._tune_iterations(), tune_scoring=scoring,
                            tune_inner_cv=int(r.tuning.inner_cv or 3), n_jobs=int(r.n_jobs),
                            feature_cfg=r.feature_selection,
                            eval_calibrated_in_cv=bool(getattr(cal, 'calibrate', False)),
                            calib_method=str(getattr(cal, 'method', 'sigmoid')), calib_cv=int(r.cv_folds),
                            calibration_ensemble=bool(getattr(cal, 'ensemble', False)), fit_kwargs=kwargs)
                        score = selection_summary(cvres.per_fold_metrics, scoring)
                    rows.append(dict(model=key, status='ok', **score))
                except Exception as error:
                    failures[key] = f'{type(error).__name__}: {error}'
            if not rows:
                raise ValueError(f"No completed stacking candidates for {family}: {failures}")
            ordered = ([rows[0]['model']] if len(keys) == 1 else
                       rank_candidates(pd.DataFrame(rows), recipes, scoring)[0].model.tolist())
            for key in ordered:
                try:
                    builder, kwargs, cal = recipes[key]
                    params = self._tune_full(key, builder, X, y, fit_kwargs=kwargs)
                    def tuned_builder():
                        return builder().set_params(**params)
                    estimator = self._fit_final(tuned_builder, X, y,
                        calibrate=bool(getattr(cal, 'calibrate', False)),
                        method=str(getattr(cal, 'method', 'sigmoid')), cv=int(r.cv_folds),
                        seed=int(r.random_state), fit_kwargs=kwargs,
                        ensemble=bool(getattr(cal, 'ensemble', False)))
                    estimators.append((family, estimator))
                    details[family] = dict(selected_key=key, fit_ids=X.index.tolist(),
                        best_params=params, candidate_scores=rows, failures=failures)
                    break
                except Exception as error:
                    failures[key] = f'{type(error).__name__}: {error}'
            else:
                raise ValueError(f"No fitted stacking candidates for {family}: {failures}")
        return estimators, details

    def _write_bulkformer_reduction_artifacts(self, model, cvres: CVResult, model_key: str) -> None:
        representation = _bulkformer_representation_from_model(model)
        if representation is None:
            return
        out = self.artifacts_dir / "bulkformer_reduction"
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "model": model_key,
            "strategy": representation.strategy,
            "requested_supervised_dimensions": representation.supervised_dimensions,
            "requested_pca_components": representation.pca_components,
            "input_dimensions": int(representation.n_features_in_),
            "output_dimensions": int(representation.effective_n_features_),
            "random_state": representation.random_state,
            "input_features": list(map(str, representation.feature_names_in_)),
            "output_features": list(map(str, representation.output_features_)),
        }
        (out / "reduction_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        fold_rows = []
        for metrics in cvres.per_fold_metrics:
            if "bulkformer_reduction_strategy" in metrics:
                fold_rows.append({
                    "fold": metrics.get("fold"),
                    "strategy": metrics.get("bulkformer_reduction_strategy"),
                    "input_dimensions": metrics.get("bulkformer_input_dimensions"),
                    "output_dimensions": metrics.get("bulkformer_output_dimensions"),
                    "selected_dimensions": ";".join(metrics.get("bulkformer_selected_dimensions", [])),
                    "selected_close_pairs": json.dumps(
                        metrics.get("bulkformer_close_pairs", []), default=str
                    ),
                })
        pd.DataFrame(fold_rows).to_csv(out / "fold_reduction_summary.csv", index=False)
        selector = getattr(representation, "selector_", None)
        if selector is not None:
            selector.selection_report_.to_csv(out / "selected_embedding_dimensions.csv", index=False)
            selector.class_centroid_distances_.to_csv(out / "class_centroid_distances.csv")
            pd.DataFrame(selector.selected_close_pairs_, columns=["class_a", "class_b", "distance"]).to_csv(
                out / "selected_close_class_pairs.csv", index=False
            )
        pca = getattr(representation, "pca_", None)
        if pca is not None:
            ratios = np.asarray(pca.explained_variance_ratio_)
            pd.DataFrame({
                "component": representation.output_features_,
                "explained_variance_ratio": ratios,
                "cumulative_explained_variance_ratio": np.cumsum(ratios),
            }).to_csv(out / "pca_explained_variance.csv", index=False)
    

    def _model_keys_for_labels(self, y):
        model_imbalance_ = self.cfg.runtime.imbalance
        info = assess_imbalance(y, mild_threshold=model_imbalance_.mild_threshold, severe_threshold=model_imbalance_.severe_threshold)
        sev = info["severity"]
        model_keys_by_severity = {
            Severity.BALANCED: (
                "plain_random_forest",
                "light_gbm",
                "xgboost",
                "stacked_ensemble",
            ),
            Severity.MILD: (
                "plain_random_forest",
                "weighted_random_forest",
                "light_gbm",
                "weighted_light_gbm",
                "xgboost",
                "weighted_xgboost",
                "stacked_ensemble",
            ),
            Severity.SEVERE: (
                "plain_random_forest",
                "weighted_random_forest",
                "smote_random_forest",
                "adasyn_random_forest",
                "svm_smote_random_forest",
                "bl_smote_random_forest",
                "light_gbm",
                "weighted_light_gbm",
                "xgboost",
                "weighted_xgboost",
                "stacked_ensemble",
            ),
        }
        return model_keys_by_severity[sev]


    # ----- per-variant -----
    def run_rf_variant(self, key: str, X: pd.DataFrame, y: pd.Series, filename_for_artifact: str) -> tuple[dict, Path]:
        r = self.cfg.runtime
        rf_cfg = self.cfg.models.random_forest
        cal = rf_cfg.defaults.calibration
        self.class_name_map = _build_class_name_map(y, self.cfg)
        cvs = r.cv_folds


        base_builder, space, fit_kwargs, cal = self._candidate_setup(key, X, y)
        self.builders[key] = base_builder

        fs = getattr(r, "feature_selection", None)
        tun = getattr(r, "tuning", None)
        scoring = str(getattr(tun, "scoring_override", None) or r.scoring)
        eval_calibrated_in_cv = bool(cal.calibrate)

        cvres = nested_oof_cv(
            model_builder=base_builder,
            X=X,
            y=y,
            outer_splits=int(cvs),
            seed=int(r.random_state),
            tune_space=space,
            tune_iter=self._tune_iterations(),
            tune_scoring=scoring,
            tune_inner_cv=int(getattr(tun, "inner_cv", None) or 3),
            n_jobs=int(r.n_jobs),
            feature_cfg=fs,
            eval_calibrated_in_cv= eval_calibrated_in_cv,
            calib_method=str(cal.method),
            calib_cv=cvs,
            calibration_ensemble=bool(cal.ensemble),
            fit_kwargs=fit_kwargs,
        )

        if fs is not None and bool(getattr(fs, "enabled", False)):
            cvres, consensus_genes, full_info = self._save_genesets_from_cvres(
                model_key=key, cvres=cvres, feature_selection_cfg=fs
            )
            if consensus_genes and not self._partition_active():
                X = X.loc[:, consensus_genes]

        self.cv_results[key] = cvres
        df = pd.DataFrame(cvres.per_fold_metrics)
        row = {
            "model": key,
            "f1_macro_mean": df["f1_macro"].mean(),
            "f1_macro_std": df["f1_macro"].std(),
            "accuracy_mean": df["accuracy"].mean(),
            "accuracy_std": df["accuracy"].std(),
        }
        row.update(selection_summary(cvres.per_fold_metrics, scoring))
        if self._bulkformer_reduction_active():
            reduction = embedding_config(self.cfg).reduction
            row.update(
                bulkformer_reduction_strategy=reduction.strategy,
                bulkformer_requested_supervised_dimensions=(
                    reduction.supervised_dimensions
                    if "supervised" in reduction.strategy else np.nan
                ),
                bulkformer_requested_pca_components=(
                    reduction.pca_components
                    if "pca" in reduction.strategy else np.nan
                ),
            )
        # Final production model on ALL data (calibrated if YAML says so)
        best_params = self._tune_full(key, base_builder, X, y) 

        # report CV results
        report_cv(
            cvres=cvres,
            artifacts_dir=self.artifacts_dir,
            model_name=key,
            estimator_cls_name=base_builder().__class__.__name__,
            X_shape=X.shape,
            y_series=y,
            seed=int(r.random_state),
            cfg_short={
                "cv_folds": int(cvs),
                "scoring": str(getattr(getattr(r, "tuning", None), "scoring_override", None) or r.scoring),
                "calibrate": bool(getattr(cal, "calibrate", False)),
                "calibration_cv": int(cvs),
                "calibration_ensemble": bool(cal.ensemble),
                "tune_inner_cv": int(getattr(getattr(r, "tuning", None), "inner_cv", None) or 3),
                "bayes_iter": self._tune_iterations(),
            },
            final_params_full_data=best_params or {},
            class_name_map=self.class_name_map, 
        )


        def tuned_builder():
            est = base_builder()
            if best_params:
                est.set_params(**best_params)
            return est

        final_model = self._fit_final(
            tuned_builder, X, y,
            calibrate=bool(cal.calibrate),
            method=cal.method,
            cv=int(cvs),
            seed=int(r.random_state),
            fit_kwargs=fit_kwargs,
            ensemble=bool(cal.ensemble),
        )

        # Save final model
        saved_path = self._save_model(final_model, filename_for_artifact)
        self.final_models[key] = final_model  # optional in-memory handle
        self._write_bulkformer_reduction_artifacts(final_model, cvres, key)

        return row, saved_path
    
    # ----- bossting -----
    def run_boosting(self, key, X: pd.DataFrame, y: pd.Series, filename_for_artifact: str, model_conf: Config) -> tuple[dict, Path]:
        r = self.cfg.runtime

        if self.class_name_map is None:
            self.class_name_map = _build_class_name_map(y, self.cfg)
        base_builder, space, fit_kwargs, cal = self._candidate_setup(key, X, y, model_conf=model_conf)

        fs = getattr(r, "feature_selection", None)
        tun = getattr(r, "tuning", None)
        scoring = str(getattr(tun, "scoring_override", None) or r.scoring)

        eval_calibrated_in_cv = bool(getattr(cal, "calibrate", False))
        cal_method = str(getattr(cal, "method", "sigmoid")) if cal is not None else "sigmoid"
        cvs = r.cv_folds
        self.builders[key] = base_builder

        cvres = nested_oof_cv(
            model_builder=base_builder,
            X=X,
            y=y,
            outer_splits=int(cvs),
            seed=int(r.random_state),
            tune_space=space,
            tune_iter=self._tune_iterations(),
            tune_scoring=scoring,
            tune_inner_cv=int(getattr(tun, "inner_cv", None) or 3),
            n_jobs=int(r.n_jobs),
            feature_cfg=fs,
            eval_calibrated_in_cv=eval_calibrated_in_cv,
            calib_method=cal_method,
            calib_cv=cvs,
            calibration_ensemble=bool(getattr(cal, "ensemble", False)),
            fit_kwargs=fit_kwargs,
        )
        if fs is not None and bool(getattr(fs, "enabled", False)):
            cvres, consensus_genes, full_info = self._save_genesets_from_cvres(model_key=key, cvres=cvres, feature_selection_cfg=fs)
            if consensus_genes and len(consensus_genes) > 0 and not self._partition_active():
                X = X.loc[:, consensus_genes]

        self.cv_results[key] = cvres
        df = pd.DataFrame(cvres.per_fold_metrics)
        row = {
            "model": key,
            "f1_macro_mean": df["f1_macro"].mean(),
            "f1_macro_std": df["f1_macro"].std(),
            "accuracy_mean": df["accuracy"].mean(),
            "accuracy_std": df["accuracy"].std(),
        }
        row.update(selection_summary(cvres.per_fold_metrics, scoring))
        if self._bulkformer_reduction_active():
            reduction = embedding_config(self.cfg).reduction
            row.update(
                bulkformer_reduction_strategy=reduction.strategy,
                bulkformer_requested_supervised_dimensions=(
                    reduction.supervised_dimensions
                    if "supervised" in reduction.strategy else np.nan
                ),
                bulkformer_requested_pca_components=(
                    reduction.pca_components
                    if "pca" in reduction.strategy else np.nan
                ),
            )
        
        # Final production model on ALL data (calibrated if YAML says so)
        best_params = self._tune_full(key, base_builder, X, y, fit_kwargs=fit_kwargs)

        per_fold_best = [m.get("best_rounds") for m in cvres.per_fold_metrics]
        cap = None
        estimator_key = "classifier__n_estimators" if self._bulkformer_reduction_active() else "n_estimators"
        if self._partition_active():
            estimator_key = 'estimator__' + estimator_key
        if best_params and estimator_key in best_params:
            cap = int(best_params[estimator_key])
        
        elif hasattr(model_conf.param_grid, "n_estimators"):
            try:
                cap = max(model_conf.param_grid.n_estimators)
            except Exception:
                cap = None
        pfe = r.per_fold_estimators
        final_n_estimators = _aggregate_best_rounds(per_fold_best, method=str(pfe.method), pad=int(pfe.pad),min_rounds=int(pfe.min_rounds), max_rounds=cap)

        # report CV results
        report_cv(
            cvres=cvres,
            artifacts_dir=self.artifacts_dir,
            model_name=key,
            estimator_cls_name=base_builder().__class__.__name__,
            X_shape=X.shape,
            y_series=y,
            seed=int(r.random_state),
            cfg_short={
                "cv_folds": int(r.cv_folds),
                "scoring": scoring,
                "eval_calibrated_in_cv": bool(eval_calibrated_in_cv),
                "calib_method": cal_method if eval_calibrated_in_cv else "",
                "calib_cv": int(cvs) if eval_calibrated_in_cv else 0,
                "tune_inner_cv": int(getattr(tun, "inner_cv", None) or 3),
                "bayes_iter": self._tune_iterations(),
            },
            final_params_full_data=best_params or {},
            class_name_map=self.class_name_map,
        )
        
        def tuned_builder():
            est = base_builder()
            if best_params:
                est.set_params(**best_params)
            if final_n_estimators is not None:
                params = est.get_params(deep=True) if hasattr(est, "get_params") else {}
                if estimator_key in params:
                    est.set_params(**{estimator_key: int(final_n_estimators)})
            return est
        
        fit_kwargs_no_es = dict(fit_kwargs or {})
        fit_kwargs_no_es = sanitize_fit_kwargs_for_xgb(key, fit_kwargs_no_es)

        final_model = self._fit_final(
            tuned_builder, X, y,
            calibrate=bool(getattr(cal, "calibrate", False)) if cal is not None else False,
            method=cal_method,
            cv=cvs,
            seed=int(r.random_state),
            fit_kwargs=fit_kwargs_no_es,
            ensemble=bool(getattr(cal, "ensemble", False)),
        )

        # Save final model
        saved_path = self._save_model(final_model, filename_for_artifact)
        self.final_models[key] = final_model 
        self._write_bulkformer_reduction_artifacts(final_model, cvres, key)

        return row, saved_path

    def best_rf_key_from_cv(self) -> str:
        return self._best_completed_key(
            [key for key in self.final_models if key.endswith("random_forest")]
        )

    def _best_completed_key(self, keys):
        scoring = self.cfg.runtime.scoring
        rows = []
        for key in keys:
            # CV records can survive a subsequent final-fit/reporting failure.
            if key not in self.final_models or key in getattr(self, "_failed_keys", set()):
                continue
            result = self.cv_results.get(key)
            if result is None:
                continue
            try:
                rows.append(dict(model=key, **selection_summary(result.per_fold_metrics, scoring)))
            except ValueError:
                continue
        if not rows:
            raise ValueError("No completed base candidates with usable training-CV scores")
        ranked, _ = rank_candidates(pd.DataFrame(rows), keys, scoring)
        return str(ranked.iloc[0]["model"])

    # ----- stacked -----
    def run_stacked(self, key: str, X: pd.DataFrame, y: pd.Series, filename_for_artifact: str, model_conf: Config) -> tuple[dict, Path]:
        r = self.cfg.runtime
        stack_cfg = self.cfg.models.stacked_ensemble
        logr = stack_cfg.logistic_regression

        rf_key = self.best_rf_key_from_cv()
        xgb_key = self.best_boosting_key_from_cv(["xgboost", "weighted_xgboost"])
        lgbm_key = self.best_boosting_key_from_cv(["light_gbm", "weighted_light_gbm"])

        fs = getattr(r, "feature_selection", None)
        tun = getattr(r, "tuning", None)
        scoring = str(getattr(tun, "scoring_override", None) or r.scoring)

        # --- meta builder ---
        def meta_builder():
            return LogisticRegression(
                max_iter=int(logr.max_iter),
                solver=str(logr.solver),
                random_state=int(r.random_state),
            )

        stack_cvres, _Z_oof = train_oof_stacker(
            base_specs=[],
            X=X,
            y=y,
            outer_splits=int(r.cv_folds),
            seed=int(r.random_state),
            tune_iter=self._tune_iterations(),
            tune_scoring=scoring,
            tune_inner_cv=int(getattr(tun, "inner_cv", None) or 3),
            n_jobs=int(r.n_jobs),
            feature_cfg=fs,
            meta_builder=meta_builder,
            fit_bases=self._fit_stacking_bases,
            passthrough=bool(stack_cfg.passthrough),
            passthrough_preprocessor=getattr(self, 'preprocessor', None),
        )

        self.cv_results[key] = stack_cvres

        df = pd.DataFrame(stack_cvres.per_fold_metrics)
        row = {
            "model": key,
            "f1_macro_mean": df["f1_macro"].mean(),
            "f1_macro_std": df["f1_macro"].std(),
            "accuracy_mean": df["accuracy"].mean(),
            "accuracy_std": df["accuracy"].std(),
        }
        row.update(selection_summary(stack_cvres.per_fold_metrics, scoring))
        if self._bulkformer_reduction_active():
            row["bulkformer_reduction_strategy"] = embedding_config(self.cfg).reduction.strategy
        final_stack = get_builder(
            "stacked_ensemble",
            random_state=int(r.random_state),
            n_jobs=int(r.n_jobs),
            max_iter=int(logr.max_iter),
            solver=str(logr.solver),
            cv="prefit",
            passthrough=bool(getattr(stack_cfg, "passthrough", True)),
            rf_trained=self.final_models[rf_key],
            xgb_trained=self.final_models[xgb_key],
            lgbm_trained=self.final_models[lgbm_key],
            rf_key=rf_key,
            xgb_key=xgb_key,
            lgbm_key=lgbm_key,
        )()
        if getattr(self, 'preprocessor', None) is not None and final_stack.passthrough:
            from sklearn.base import clone
            final_stack.passthrough_preprocessor_ = clone(self.preprocessor).fit(X, y)
        # Each base receives its own fitted schema. RF consensus must never
        # truncate the common input needed by the other bases or passthrough.
        final_stack.fit_honest(X, y, Z_oof=pd.DataFrame(_Z_oof, index=X.index,
            columns=stack_cvres.stacking_probability_columns))

        saved_path = self._save_model(final_stack, filename_for_artifact)
        self.final_models[key] = final_stack
        scope = dict(outer_folds=stack_cvres.per_fold_metrics,
            final_oof_scopes=stack_cvres.stacking_final_oof_scopes,
            final_base_models=dict(rf=rf_key, xgb=xgb_key, lgbm=lgbm_key),
            probability_columns=final_stack.probability_columns_,
            meta_feature_columns=final_stack.meta_feature_columns_,
            input_features=list(final_stack.feature_names_in_))
        (self.artifacts_dir / 'stacking_fit_scope.json').write_text(
            json.dumps(scope, indent=2, default=str), encoding='utf-8')

        # Report
        class_name_map = _build_class_name_map(y, self.cfg)
        report_cv(
            cvres=stack_cvres,
            artifacts_dir=self.artifacts_dir,
            model_name=key,
            estimator_cls_name=final_stack.__class__.__name__,
            X_shape=X.shape,
            y_series=y,
            seed=int(r.random_state),
            cfg_short={
                "cv_folds": int(r.cv_folds),
                "scoring": scoring,
                "base_models": [rf_key, xgb_key, lgbm_key],
                "meta": f"logreg(solver={logr.solver}, max_iter={logr.max_iter})",
                "passthrough": bool(getattr(stack_cfg, "passthrough", True)),
            },
            class_name_map=class_name_map,
        )

        return row, saved_path
    
    def best_boosting_key_from_cv(self, keys: list[str]) -> str:
        return self._best_completed_key(keys)



    # ----- entry point -----
    def run(self, X: pd.DataFrame, y: pd.Series) -> TrainSummary:
        # Reusing a Trainer must not retain completed state from an earlier run.
        self.final_models.clear()
        self.cv_results.clear()
        self.builders.clear()
        self._failed_keys = set()
        rf_var = self.cfg.models.random_forest.variants
        selected_model_keys = self._model_keys_for_labels(y)
        lgbm_file_name = self.cfg.models.light_gbm.filename
        xgboost_file_name = self.cfg.models.xgboost.filename
        stacked_file_name = self.cfg.models.stacked_ensemble.filename
        lgbm_cw_file_name = self.cfg.models.light_gbm.weighted_filename
        xgb_cw_file_name = self.cfg.models.xgboost.weighted_filename


        if self.cfg.surface_markers.enabled and not embedding_enabled(self.cfg):
            sm_genes = apply_surface_marker_filter_with_min(X.columns.tolist(), min_features=self.cfg.runtime.min_features)
            X = X.loc[:, sm_genes]

        rf_filenames = {
            "plain_random_forest": rf_var.plain_random_forest.filename,
            "weighted_random_forest": rf_var.weighted_random_forest.filename,
            "smote_random_forest": rf_var.smote_random_forest.filename,
            "adasyn_random_forest": rf_var.adasyn_random_forest.filename,
            "svm_smote_random_forest": rf_var.svm_smote_random_forest.filename,
            "bl_smote_random_forest": rf_var.bl_smote_random_forest.filename,
        }
        variants = [
            (key, rf_filenames[key])
            for key in selected_model_keys
            if key in rf_filenames
        ]
        rows: list[dict] = []
        paths: dict[str, Path] = {}
        i = 1
        for key, fname in variants:
            try:
                row, p = self.run_rf_variant(key, X, y, filename_for_artifact=fname)
                i += 1
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths[key] = p
            except Exception as e:
                self._failed_keys.add(key)
                rows.append(_failed_row(key, e))

        #--- LightGBM ---
        if "light_gbm" in selected_model_keys:
            try:
                row, p = self.run_boosting("light_gbm", X, y, filename_for_artifact=lgbm_file_name, model_conf=self.cfg.models.light_gbm)
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths["light_gbm"] = p
            except Exception as e:
                self._failed_keys.add("light_gbm")
                print(traceback.format_exc())
                rows.append(_failed_row("light_gbm", e))
        if "weighted_light_gbm" in selected_model_keys:
            try:
                row, p = self.run_boosting(
                    "weighted_light_gbm",
                    X,
                    y,
                    filename_for_artifact=lgbm_cw_file_name,
                    model_conf=self.cfg.models.light_gbm,
                )
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths["weighted_light_gbm"] = p
            except Exception as e:
                self._failed_keys.add("weighted_light_gbm")
                print(traceback.format_exc())
                rows.append(_failed_row("weighted_light_gbm", e))

        # --- XGBoost ---
        if "xgboost" in selected_model_keys:
            try:
                row, p = self.run_boosting("xgboost", X, y, filename_for_artifact=xgboost_file_name, model_conf=self.cfg.models.xgboost)
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths["xgboost"] = p
            except Exception as e:
                self._failed_keys.add("xgboost")
                print(traceback.format_exc())
                rows.append(_failed_row("xgboost", e))

        if "weighted_xgboost" in selected_model_keys:
            try:
                row, p = self.run_boosting(
                    "weighted_xgboost",
                    X,
                    y,
                    filename_for_artifact=xgb_cw_file_name,
                    model_conf=self.cfg.models.xgboost,
                )
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths["weighted_xgboost"] = p
            except Exception as e:
                self._failed_keys.add("weighted_xgboost")
                print(traceback.format_exc())
                rows.append(_failed_row("weighted_xgboost", e))

        # --- Stacked ensemble (only if prerequisites exist) ---
        have_rf = any(k.endswith("random_forest") and k in paths for k, _ in variants)
        have_xgb = any(k in paths for k in ["xgboost", "weighted_xgboost"])
        have_lgbm = any(k in paths for k in ["light_gbm", "weighted_light_gbm"])


        if "stacked_ensemble" in selected_model_keys and have_rf and have_xgb and have_lgbm:
            try:
                row, p = self.run_stacked("stacked_ensemble", X, y, filename_for_artifact=stacked_file_name, model_conf=self.cfg.models.stacked_ensemble)
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths["stacked_ensemble"] = p
            except Exception as e:
                rows.append(_failed_row("stacked_ensemble", e))
        elif "stacked_ensemble" in selected_model_keys:
            missing = []
            if not have_rf: missing.append("rf")
            if not have_lgbm: missing.append("light_gbm")
            if not have_xgb: missing.append("xgboost")
            rows.append({
                "model": "stacked_ensemble",
                "f1_macro_mean": np.nan,
                "f1_macro_std": np.nan,
                "accuracy_mean": np.nan,
                "accuracy_std": np.nan,
                "status": "skipped",
                "error": f"Missing prerequisites: {', '.join(missing)}",
            })
        metrics_df = pd.DataFrame(rows)
        sort_metric = "selection_score_mean" if "selection_score_mean" in metrics_df else "f1_macro_mean"
        metrics_df = metrics_df.sort_values(
            by=[sort_metric, "model"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
        return TrainSummary(metrics=metrics_df, artifact_paths=paths)
