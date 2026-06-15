# generic-classifier/train/trainer.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import traceback
from skopt import BayesSearchCV
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from ..config.schema import Config
from ..models.registry import get_builder
from .cv_engine import nested_oof_cv, CVResult
from .stacking_engine import train_oof_stacker
from .tune import build_rf_search_space, build_lgbm_param_grid, build_xgb_param_grid
from ..utils.imbalance import assess_imbalance, Severity
from ..utils.toolbox import _to_dict, _aggregate_best_rounds, sanitize_fit_kwargs_for_xgb, _build_class_name_map, unwrap_for_export
from ..reporting.per_model_report import report_cv
from ..utils.param_router import split_init_and_fit_kwargs
from .auto_config import apply_auto_config
from omiclass.errors.errors import TuningError 
from omiclass.errors.faild_model import _failed_row
from sklearn.pipeline import Pipeline as SkPipeline
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
    def __init__(self, cfg: Config, artifacts_dir: Union[str, Path]):
        self.cfg = cfg
        self.class_name_map = None
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.final_models: dict[str, object] = {} 
        self.cv_results: dict[str, CVResult] = {}
        self.builders: dict[str, Any] = {}

    # Internal helpers
    def _safe_inner_cv(self, y: pd.Series, cv: int, seed: int):
        min_count = int(pd.Series(y).value_counts().min())
        if min_count < 2:
            return 0  # signal: cannot calibrate safely
        return StratifiedKFold(n_splits=max(2, min(cv, min_count)), shuffle=True, random_state=seed)
    
    def _tune_full(self, key: str, base_builder, X: pd.DataFrame, y: pd.Series, fit_kwargs: dict | None = None) -> dict:
        """Run BayesSearchCV on ALL data to get best_params_ for the FINAL model only."""
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

        tun = self.cfg.runtime.tuning
        runt = self.cfg.runtime

        inner = self._safe_inner_cv(y, cv=int(tun.inner_cv), seed=int(runt.random_state))
        if inner == 0:
            # Cannot split safely; skip tuning
            return {}
            
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

    def _fit_final(self, builder, X: pd.DataFrame, y: pd.Series, *, calibrate: bool, method: str, cv: int, seed: int, fit_kwargs: dict | None = None,):
        base = builder()
        if not calibrate:
            return self._final_fit_with_kwargs(base, X, y, fit_kwargs=fit_kwargs)
        inner = self._safe_inner_cv(y, cv=cv, seed=seed)
        if inner == 0:
            base.fit(X, y)
            return base
        wrapped = CalibratedClassifierCV(base, method=method, cv=inner)
        return self._final_fit_with_kwargs(wrapped, X, y, fit_kwargs=fit_kwargs)

    def _save_model(self, model, filename: str) -> Path:
        export_model = unwrap_for_export(model)
        path = (self.artifacts_dir / filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(export_model, path)
        return path
    

    # ----- per-variant -----
    def run_rf_variant(self, key: str, X: pd.DataFrame, y: pd.Series, filename_for_artifact: str) -> tuple[dict, Path]:
        r = self.cfg.runtime
        rf_cfg = self.cfg.models.random_forest
        cal = rf_cfg.defaults.calibration
        self.class_name_map = _build_class_name_map(y, self.cfg)
        cvs = r.cv_folds


        raw = self._compose_raw_params(model_key=key, model_conf=rf_cfg.defaults)

        variant_cfg = getattr(rf_cfg.variants, key)
        if getattr(variant_cfg, "class_weight", None) is not None:
            raw["class_weight"] = variant_cfg.class_weight

        init_params, fit_kwargs = split_init_and_fit_kwargs(key, raw)
        init_params["n_jobs"] = 1
        print(
            f"[Trainer] {key}: RandomForest estimator-level n_jobs=1 "
            "for reproducibility; runtime n_jobs is reserved for safe outer-level orchestration."
        )

        base_builder = get_builder(key, random_state=int(r.random_state), n_jobs=1, override_params=init_params,)
        self.builders[key] = base_builder

        prefix = "" if key in ("plain_random_forest", "weighted_random_forest") else "rf__"
        space = build_rf_search_space(X, rf_cfg.defaults.search_space, prefix=prefix)


        fs = getattr(r, "feature_selection", None)
        tun = getattr(r, "tuning", None)
        scoring = str(getattr(tun, "scoring_override", None) or r.scoring)
        eval_calibrated_in_cv = bool(getattr(self.cfg.runtime, "eval_calibrated_in_cv", False))

        cvres = nested_oof_cv(
            model_builder=base_builder,
            X=X,
            y=y,
            outer_splits=int(cvs),
            seed=int(r.random_state),
            tune_space=space,
            tune_iter=int(r.bayesian_iterations),
            tune_scoring=scoring,
            tune_inner_cv=int(getattr(tun, "inner_cv", 3) if tun is not None else 3),
            n_jobs=int(r.n_jobs),
            feature_cfg=fs,
            eval_calibrated_in_cv= eval_calibrated_in_cv,
            calib_method="sigmoid",
            calib_cv=cvs,
        )

        if fs is not None and bool(getattr(fs, "enabled", False)):
            cvres, consensus_genes, full_info = self._save_genesets_from_cvres(
                model_key=key, cvres=cvres, feature_selection_cfg=fs
            )
            if consensus_genes:
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
                "tune_inner_cv": int(getattr(getattr(r, "tuning", None), "inner_cv", 3)),
                "bayes_iter": int(getattr(r, "bayesian_iterations", 0)),
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
        )

        # Save final model
        saved_path = self._save_model(final_model, filename_for_artifact)
        self.final_models[key] = final_model  # optional in-memory handle

        return row, saved_path
    
    # ----- bossting -----
    def run_boosting(self, key, X: pd.DataFrame, y: pd.Series, filename_for_artifact: str, model_conf: Config) -> tuple[dict, Path]:
        r = self.cfg.runtime

        raw = self._compose_raw_params(model_key=key, model_conf=model_conf)
        raw = apply_auto_config(key, raw, y)

        weighted_xgb_multiclass = False
        if self.class_name_map is None:
            self.class_name_map = _build_class_name_map(y, self.cfg)


        if key == "weighted_light_gbm":
            raw["class_weight"] = "balanced"

        if key == "weighted_xgboost":
            counts = pd.Series(y).value_counts()
            if len(counts) == 2:
                raw["scale_pos_weight"] = float(counts.max() / counts.min())
            else:
                weighted_xgb_multiclass = True

        init_params, fit_kwargs = split_init_and_fit_kwargs(key, raw)
        if "xgboost" in key:
            init_params.update(
                random_state=int(r.random_state),
                seed=int(r.random_state),
                n_jobs=1,
                nthread=1,
            )
        elif "light_gbm" in key:
            init_params.update(
                random_state=int(r.random_state),
                seed=int(r.random_state),
                feature_fraction_seed=int(r.random_state),
                bagging_seed=int(r.random_state),
                data_random_seed=int(r.random_state),
                drop_seed=int(r.random_state),
                deterministic=True,
                force_col_wise=True,
                n_jobs=1,
            )
        if key == "weighted_light_gbm":
            tmp_builder = get_builder(
                key,
                random_state=int(r.random_state),
                n_jobs=int(r.n_jobs),
                override_params=init_params,
            )
            tmp_est = tmp_builder()

        fit_kwargs.pop("early_stopping_rounds", None)
        fit_kwargs.pop("early_stopping", None)

        if weighted_xgb_multiclass:
            sw = compute_sample_weight("balanced", y)
            fit_kwargs["sample_weight"] = sw
            
        def base_builder():
            est_builder = get_builder(
                key,
                random_state=int(r.random_state),
                n_jobs=int(r.n_jobs),
                override_params=init_params,
            )
            est = est_builder()
            if hasattr(est, "get_params") and "n_jobs" in est.get_params(deep=False):
                est.set_params(n_jobs=1)
            return est
            
        if "xgboost" in key:
            space = build_xgb_param_grid(X, model_conf.param_grid)
        else:
            space = build_lgbm_param_grid(X, model_conf.param_grid)

        fs = getattr(r, "feature_selection", None)
        tun = getattr(r, "tuning", None)
        scoring = str(getattr(tun, "scoring_override", None) or r.scoring)

        eval_calibrated_in_cv = bool(getattr(self.cfg.runtime, "eval_calibrated_in_cv", False))

        cal = getattr(model_conf, "calibration", None)
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
            tune_iter=int(r.bayesian_iterations),
            tune_scoring=scoring,
            tune_inner_cv=int(getattr(tun, "inner_cv", 3) if tun is not None else 3),
            n_jobs=int(r.n_jobs),
            feature_cfg=fs,
            eval_calibrated_in_cv=eval_calibrated_in_cv,
            calib_method=cal_method,
            calib_cv=cvs,
            fit_kwargs=fit_kwargs,
        )
        if fs is not None and bool(getattr(fs, "enabled", False)):
            cvres, consensus_genes, full_info = self._save_genesets_from_cvres(model_key=key, cvres=cvres, feature_selection_cfg=fs)
            if consensus_genes and len(consensus_genes) > 0:
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
        
        # Final production model on ALL data (calibrated if YAML says so)
        best_params = self._tune_full(key, base_builder, X, y, fit_kwargs=fit_kwargs)

        per_fold_best = [m.get("best_rounds") for m in cvres.per_fold_metrics]
        cap = None
        if best_params and "n_estimators" in best_params:
            cap = int(best_params["n_estimators"])
        
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
                "tune_inner_cv": int(getattr(tun, "inner_cv", 3) if tun is not None else 3),
                "bayes_iter": int(r.bayesian_iterations),
            },
            final_params_full_data=best_params or {},
            class_name_map=self.class_name_map,
        )
        
        def tuned_builder():
            est = base_builder()
            if best_params:
                est.set_params(**best_params)
            if final_n_estimators is not None:
                if hasattr(est, "get_params") and "n_estimators" in est.get_params(deep=False):
                    est.set_params(n_estimators=int(final_n_estimators))
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
        )

        # Save final model
        saved_path = self._save_model(final_model, filename_for_artifact)
        self.final_models[key] = final_model 

        return row, saved_path

    def best_rf_key_from_cv(self) -> str:
        metric = str(self.cfg.runtime.scoring)

        rf_keys = [k for k in self.cv_results.keys() if k.endswith("random_forest")]
        rows = []
        for k in rf_keys:
            cvres = self.cv_results.get(k)
            if cvres is None or not getattr(cvres, "per_fold_metrics", None):
                continue
            df = pd.DataFrame(cvres.per_fold_metrics)
            if metric not in df.columns:
                continue
            rows.append({
                "model": k,
                f"{metric}_mean": float(df[metric].mean()),
                f"{metric}_std":  float(df[metric].std(ddof=0)),
            })

        out = pd.DataFrame(rows).sort_values(
            by=[f"{metric}_mean", "model"],
            ascending=[False, True],
            kind="mergesort",
        )
        return out.iloc[0]["model"]

    # ----- stacked -----
    def run_stacked(self, key: str, X: pd.DataFrame, y: pd.Series, filename_for_artifact: str, model_conf: Config) -> tuple[dict, Path]:
        r = self.cfg.runtime
        stack_cfg = self.cfg.models.stacked_ensemble
        logr = stack_cfg.logistic_regression

        rf_key = self.best_rf_key_from_cv()
        xgb_key = self.best_boosting_key_from_cv(["xgboost", "weighted_xgboost"])
        lgbm_key = self.best_boosting_key_from_cv(["light_gbm", "weighted_light_gbm"])

        rf_cfg = self.cfg.models.random_forest
        prefix = "" if rf_key in ("plain_random_forest", "weighted_random_forest") else "rf__"
        rf_space = build_rf_search_space(X, rf_cfg.defaults.search_space, prefix=prefix)
        xgb_space = build_xgb_param_grid(X, self.cfg.models.xgboost.param_grid)
        lgbm_space = build_lgbm_param_grid(X, self.cfg.models.light_gbm.param_grid)

        fs = getattr(r, "feature_selection", None)
        tun = getattr(r, "tuning", None)
        scoring = str(getattr(tun, "scoring_override", None) or r.scoring)

        base_specs = [
            (rf_key, self.builders[rf_key], rf_space),
            (xgb_key,  self.builders[xgb_key], xgb_space),
            (lgbm_key,  self.builders[lgbm_key], lgbm_space),
        ]
        # --- meta builder ---
        def meta_builder():
            return LogisticRegression(
                max_iter=int(logr.max_iter),
                solver=str(logr.solver),
                random_state=int(r.random_state),
            )

        stack_cvres, _Z_oof = train_oof_stacker(
            base_specs=base_specs,
            X=X,
            y=y,
            outer_splits=int(r.cv_folds),
            seed=int(r.random_state),
            tune_iter=int(r.bayesian_iterations),
            tune_scoring=scoring,
            tune_inner_cv=int(getattr(tun, "inner_cv", 3) if tun is not None else 3),
            n_jobs=int(r.n_jobs),
            feature_cfg=fs,
            meta_builder=meta_builder,
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
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.final_models[rf_key])
        except Exception as e:
            pass
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
        if fs is not None and bool(getattr(fs, "enabled", False)):
            rf_cvres = self.cv_results.get(rf_key)
            consensus = getattr(rf_cvres, "geneset_consensus_genes", None) if rf_cvres is not None else None
            if consensus:
                X = X.loc[:, list(consensus)]


        if hasattr(final_stack, "fit_honest"):
            final_stack.fit_honest(X, y, Z_oof=_Z_oof)         # preferred (OOF meta)
        else:
            final_stack.fit(X, y)                              # fallback (non-OOF meta)

        saved_path = self._save_model(final_stack, filename_for_artifact)
        self.final_models[key] = final_stack

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
                "base_models": [rf_key, "xgboost", "light_gbm"],
                "meta": f"logreg(solver={logr.solver}, max_iter={logr.max_iter})",
                "passthrough": bool(getattr(stack_cfg, "passthrough", True)),
            },
            class_name_map=class_name_map,
        )

        return row, saved_path
    
    def best_boosting_key_from_cv(self, keys: list[str]) -> str:
        metric = str(self.cfg.runtime.scoring)
        rows = []

        for k in keys:
            cvres = self.cv_results.get(k)
            if cvres is None or not getattr(cvres, "per_fold_metrics", None):
                continue
            df = pd.DataFrame(cvres.per_fold_metrics)
            if metric not in df.columns:
                continue
            rows.append({
                "model": k,
                f"{metric}_mean": float(df[metric].mean()),
            })

        if not rows:
            return keys[0]

        return pd.DataFrame(rows).sort_values(
            by=[f"{metric}_mean", "model"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]["model"]



    # ----- entry point -----
    def run(self, X: pd.DataFrame, y: pd.Series) -> TrainSummary:
        rf_var = self.cfg.models.random_forest.variants
        model_imbalance_ = self.cfg.runtime.imbalance
        info = assess_imbalance(y, mild_threshold=model_imbalance_.mild_threshold, severe_threshold=model_imbalance_.severe_threshold)
        sev = info["severity"]
        lgbm_file_name = self.cfg.models.light_gbm.filename
        xgboost_file_name = self.cfg.models.xgboost.filename
        stacked_file_name = self.cfg.models.stacked_ensemble.filename
        lgbm_cw_file_name = self.cfg.models.light_gbm.weighted_filename
        xgb_cw_file_name = self.cfg.models.xgboost.weighted_filename

        if self.cfg.surface_markers.enabled:
            sm_genes = apply_surface_marker_filter_with_min(X.columns.tolist(), min_features=self.cfg.runtime.min_features)
            X = X.loc[:, sm_genes]

        if sev == Severity.BALANCED:
            # Skip resampling models entirely
            variants = [
                ("plain_random_forest",    rf_var.plain_random_forest.filename),
                ("weighted_random_forest", rf_var.weighted_random_forest.filename),
            ]
        else:  # Severity.SEVERE or Severity.MILD
            variants = [
                ("plain_random_forest",       rf_var.plain_random_forest.filename),
                ("weighted_random_forest",    rf_var.weighted_random_forest.filename),
                ("smote_random_forest",       rf_var.smote_random_forest.filename),
                ("adasyn_random_forest",      rf_var.adasyn_random_forest.filename),
                ("svm_smote_random_forest",   rf_var.svm_smote_random_forest.filename),
                ("bl_smote_random_forest",    rf_var.bl_smote_random_forest.filename),
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
                rows.append(_failed_row(key, e))

        #--- LightGBM ---
        try:
            row, p = self.run_boosting("light_gbm", X, y, filename_for_artifact=lgbm_file_name, model_conf=self.cfg.models.light_gbm)
            row["status"] = "ok"
            row["error"] = ""
            rows.append(row)
            paths["light_gbm"] = p
        except Exception as e:
            print(traceback.format_exc())
            rows.append(_failed_row("light_gbm", e))
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
            print(traceback.format_exc())
            rows.append(_failed_row("weighted_light_gbm", e))

        # --- XGBoost ---
        try:
            row, p = self.run_boosting("xgboost", X, y, filename_for_artifact=xgboost_file_name, model_conf=self.cfg.models.xgboost)
            row["status"] = "ok"
            row["error"] = ""
            rows.append(row)
            paths["xgboost"] = p
        except Exception as e:
            print(traceback.format_exc())
            rows.append(_failed_row("xgboost", e))

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
            print(traceback.format_exc())
            rows.append(_failed_row("weighted_xgboost", e))

        # --- Stacked ensemble (only if prerequisites exist) ---
        have_rf = any(k.endswith("random_forest") and (k in self.final_models) for k, _ in variants)
        have_xgb = any(k in self.final_models for k in ["xgboost", "weighted_xgboost"])
        have_lgbm = any(k in self.final_models for k in ["light_gbm", "weighted_light_gbm"])


        if have_rf and have_xgb and have_lgbm:
            try:
                row, p = self.run_stacked("stacked_ensemble", X, y, filename_for_artifact=stacked_file_name, model_conf=self.cfg.models.stacked_ensemble)
                row["status"] = "ok"
                row["error"] = ""
                rows.append(row)
                paths["stacked_ensemble"] = p
            except Exception as e:
                rows.append(_failed_row("stacked_ensemble", e))
        else:
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
        metrics_df = pd.DataFrame(rows).sort_values(
            by=["f1_macro_mean", "model"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
        return TrainSummary(metrics=metrics_df, artifact_paths=paths)
