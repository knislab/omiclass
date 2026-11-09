# omiclass/config/overrides.py
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Type, get_args

from pydantic import BaseModel

from .schema import Config
from omiclass.errors.errors import ConfigError


@dataclass(frozen=True)
class OverrideSpec:
    aliases: Dict[str, str]  
    frozen_prefixes: Tuple[Tuple[str, ...], ...]  
    allowed_frozen_paths: Tuple[Tuple[str, ...], ...] = ()


SPEC = OverrideSpec(
    aliases={
        "foundation": "foundation",
        "foundation_enabled": "foundation.enabled",
        "foundation_model": "foundation.model",
        "foundation_classifier": "foundation.classifier",
        "foundation_representation": "foundation.representation",
        "foundation_pooling": "foundation.pooling",
        "foundation_correction": "foundation.correction",
        "foundation_mode": "foundation.mode",
        "foundation_run_dir": "foundation.run_dir",
        "foundation_split_path": "foundation.split_path",
        "PDF": "PDF",
        "pdf": "PDF",

        # project
        "project_name": "project.name",
        "output_dir": "project.out_dir",
        "split_output_dir": "project.split_data_dir",
        "classes_names": "project.classes_names",
        "classification_column": "project.cluster_column",
        "class_col": "project.cluster_column",
        "meta_path": "project.meta_path",
        "modality": "project.modality",
        "project_modality": "project.modality",
        "test_holdout_path": "project.hold_out_path",
        "markers_path": "project.geneset_path",

        # BulkFormer feature extraction
        "bulkformer_enabled": "bulkformer.enabled",
        "bulkformer_root_dir": "bulkformer.root_dir",
        "bulkformer_device": "bulkformer.device",
        "bulkformer_batch_size": "bulkformer.batch_size",
        "bulkformer_aggregate_type": "bulkformer.aggregate_type",
        "bulkformer_graph_path": "bulkformer.graph_path",
        "bulkformer_graph_weights_path": "bulkformer.graph_weights_path",
        "bulkformer_gene_embedding_path": "bulkformer.gene_embedding_path",
        "bulkformer_checkpoint_path": "bulkformer.checkpoint_path",
        "bulkformer_gene_info_path": "bulkformer.gene_info_path",
        "bulkformer_gene_lengths_path": "bulkformer.gene_lengths_path",
        "bulkformer_interested_gene_path": "bulkformer.interested_gene_path",
        "bulkformer_reduction_strategy": "bulkformer.reduction.strategy",
        "bulkformer_supervised_dimensions": "bulkformer.reduction.supervised_dimensions",
        "bulkformer_pca_components": "bulkformer.reduction.pca_components",
        "bulkformer_one_vs_rest_fraction": "bulkformer.reduction.one_vs_rest_fraction",
        "bulkformer_close_pair_fraction": "bulkformer.reduction.close_pair_fraction",
        "bulkformer_close_pair_count": "bulkformer.reduction.close_pair_count",
        "bulkformer_close_pair_pca_components": "bulkformer.reduction.close_pair_pca_components",
        "bulkformer_reduction_random_state": "bulkformer.reduction.random_state",

        # CellFM feature extraction (shares the representation strategies above)
        "cellfm_enabled": "cellfm.enabled",
        "cellfm_root_dir": "cellfm.root_dir",
        "cellfm_checkpoint_path": "cellfm.checkpoint_path",
        "cellfm_gene_info_path": "cellfm.gene_info_path",
        "cellfm_vendor_dir": "cellfm.vendor_dir",
        "cellfm_device": "cellfm.device",
        "cellfm_batch_size": "cellfm.batch_size",
        "cellfm_random_state": "cellfm.random_state",
        "cellfm_num_views": "cellfm.num_views",
        "cellfm_reduction_strategy": "cellfm.reduction.strategy",
        "cellfm_supervised_dimensions": "cellfm.reduction.supervised_dimensions",
        "cellfm_pca_components": "cellfm.reduction.pca_components",
        "cellfm_one_vs_rest_fraction": "cellfm.reduction.one_vs_rest_fraction",
        "cellfm_close_pair_fraction": "cellfm.reduction.close_pair_fraction",
        "cellfm_close_pair_count": "cellfm.reduction.close_pair_count",
        "cellfm_close_pair_pca_components": "cellfm.reduction.close_pair_pca_components",
        "cellfm_reduction_random_state": "cellfm.reduction.random_state",

        # BulkRNABert isolated feature extraction
        "bulkrnabert_enabled": "bulkrnabert.enabled",
        "bulkrnabert_root_dir": "bulkrnabert.root_dir",
        "bulkrnabert_model_name": "bulkrnabert.model_name",
        "bulkrnabert_checkpoint_directory": "bulkrnabert.checkpoint_directory",
        "bulkrnabert_vocabulary_path": "bulkrnabert.vocabulary_path",
        "bulkrnabert_python_executable": "bulkrnabert.python_executable",
        "bulkrnabert_tpm_input_path": "bulkrnabert.tpm_input_path",
        "bulkrnabert_cache_dir": "bulkrnabert.cache_dir",
        "bulkrnabert_device": "bulkrnabert.device",
        "bulkrnabert_batch_size": "bulkrnabert.batch_size",
        "bulkrnabert_reduction_strategy": "bulkrnabert.reduction.strategy",
        "bulkrnabert_supervised_dimensions": "bulkrnabert.reduction.supervised_dimensions",
        "bulkrnabert_pca_components": "bulkrnabert.reduction.pca_components",
        "bulkrnabert_one_vs_rest_fraction": "bulkrnabert.reduction.one_vs_rest_fraction",
        "bulkrnabert_close_pair_fraction": "bulkrnabert.reduction.close_pair_fraction",
        "bulkrnabert_close_pair_count": "bulkrnabert.reduction.close_pair_count",
        "bulkrnabert_close_pair_pca_components": "bulkrnabert.reduction.close_pair_pca_components",
        "bulkrnabert_reduction_random_state": "bulkrnabert.reduction.random_state",

        # batch_correction
        "batch_column": "batch_correction.batch_column",
        "design_columns": "batch_correction.design_columns",
        "batch_covariates_path": "batch_correction.covariates_path",
        "batch_n_pcs": "batch_correction.n_pcs",
        "batch_fill_na": "batch_correction.fill_na",

        # runtime
        "random_state": "runtime.random_state",
        "n_jobs": "runtime.n_jobs",
        "cv_folds": "runtime.cv_folds",
        "tuning_inner_cv": "runtime.tuning.inner_cv",
        "min_features": "runtime.min_features",
        "bayesian_iterations" : "runtime.bayesian_iterations",
        "log_transform": "runtime.log_transform",
        "split_ratio": "runtime.split.train_ratio",
        "split_method": "runtime.split.method",

        # runtime.feature selection
        "feature_selection_enabled": "runtime.feature_selection.enabled",
        "feature_selection_reduce_sampler_pipelines": "runtime.feature_selection.reduce_sampler_pipelines",
        "feature_selection_min_features": "runtime.feature_selection.min_features",
        "feature_selection_max_features": "runtime.feature_selection.max_features",
        "feature_selection_subset_fracs" : "runtime.feature_selection.subset_fracs",
        "feature_selection_consensus_top_n": "runtime.feature_selection.consensus_top_n",
        "feature_selection_consensus_min_fraction": "runtime.feature_selection.consensus_min_frac",
        "feature_selection_consensus_min_folds": "runtime.feature_selection.consensus_min_folds",
        "feature_selection_consensus_min_genes": "runtime.feature_selection.consensus_min_genes",

        # runtime.survival_prediction
        "survival_prediction_enabled": "runtime.survival_prediction.enabled",
        "survival_time_column": "runtime.survival_prediction.time_col",
        "survival_event_column": "runtime.survival_prediction.event_col",
        "survival_prediction_expression_path": "runtime.survival_prediction.X_path", 
        "survival_prediction_time_scale": "runtime.survival_prediction.time_scale",

        # runtime.feature reduction
        "feature_reduction_enabled": "runtime.feature_reduction.enabled",
        "feature_reduction_ratios": "runtime.feature_reduction.k_ratios",
        "feature_reduction_repeats": "runtime.feature_reduction.n_repeats",
        "feature_reduction_min_features": "runtime.feature_reduction.min_features",

        # DE (Differential Expression)
        "de_covariates": "de.covariates",
        "de_include_batch_correction": "de.include_batch_correction",
        "include_batch_correction_in_de": "de.include_batch_correction",
        "de_include_batch_column": "de.include_batch_correction",
        "de_min_samples": "de.min_samples_frac",
        "de_k_genes": "de.k_genes",
        "de_logfc": "de.logfc_thresh",
        "de_fdr": "de.fdr_thresh",
        "de_gene_selection_min_genes": "de_gene_selection.min_genes",
        "de_gene_selection_genes_per_class": "de_gene_selection.max_genes_per_class",
        "de_gene_selection_min_genes_per_class": "de_gene_selection.min_genes_per_class",
        "de_gene_selection_max_genes_per_class": "de_gene_selection.max_genes_per_class",
        "de_gene_selection_max_genes": "de_gene_selection.max_genes",
        "de_gene_selection_one_vs_rest_fraction": "de_gene_selection.one_vs_rest_fraction",
        "de_gene_selection_close_pair_fraction": "de_gene_selection.close_pair_fraction",
        "de_gene_selection_close_pair_count": "de_gene_selection.close_pair_count",
        "de_gene_selection_close_pair_distance_method": "de_gene_selection.close_pair_distance_method",
        "de_gene_selection_close_pair_knn_k": "de_gene_selection.close_pair_knn_k",
        "de_gene_selection_pca_components": "de_gene_selection.pca_components",

        # Normalization & Surface Markers
        "gene_lengths_path": "normalization.gene_lengths_path",
        "gene_lengths_version": "project.version",
        "gencode_version": "project.version",
        "gtf_path": "normalization.gtf_path",
        "surface_markers_enabled": "surface_markers.enabled",
        "cptac_max_missing_fraction": "cptac_ms_preprocessing.max_missing_fraction",
        "cptac_log2_transform": "cptac_ms_preprocessing.log2_transform",
        "cptac_pseudocount": "cptac_ms_preprocessing.pseudocount",
        "cptac_imputation": "cptac_ms_preprocessing.imputation",
        "pc_pvalue_heatmap_max": "plots.pc_pvalue_heatmap_max",

        # models.random_forest (Defaults & Search Space)
        "rf_oob_score": "models.random_forest.defaults.oob_score",
        "rf_calibrate": "models.random_forest.defaults.calibration.calibrate",
        "rf_calibration_method": "models.random_forest.defaults.calibration.method",
        "rf_calibration_ensemble": "models.random_forest.defaults.calibration.ensemble",
        "rf_search_n_estimators": "models.random_forest.defaults.search_space.n_estimators",
        "rf_search_max_depth": "models.random_forest.defaults.search_space.max_depth",
        "rf_search_max_features": "models.random_forest.defaults.search_space.max_features",
        "rf_search_min_samples_leaf": "models.random_forest.defaults.search_space.min_samples_leaf",
        "rf_search_min_samples_split": "models.random_forest.defaults.search_space.min_samples_split",

        # models.light_gbm
        "lgbm_early_stopping": "models.light_gbm.init_params.early_stopping_rounds",
        "lgbm_metric": "models.light_gbm.eval_metric",
        "lgbm_grid_learning_rate": "models.light_gbm.param_grid.learning_rate", 
        "lgbm_grid_num_leaves": "models.light_gbm.param_grid.num_leaves",      
        "lgbm_grid_max_depth": "models.light_gbm.param_grid.max_depth",      
        "lgbm_grid_feature_fraction": "models.light_gbm.param_grid.feature_fraction", 
        "lgbm_grid_bagging_fraction": "models.light_gbm.param_grid.bagging_fraction",
        "lgbm_grid_n_estimators": "models.light_gbm.param_grid.n_estimators",   
        "lgbm_grid_min_child_samples": "models.light_gbm.param_grid.min_child_samples", 
        "lgbm_grid_bagging_freq": "models.light_gbm.param_grid.bagging_freq",  
        "lgbm_grid_lambda_l1": "models.light_gbm.param_grid.lambda_l1",      
        "lgbm_grid_lambda_l2": "models.light_gbm.param_grid.lambda_l2",

        # models.xgboost
        "xgb_filename": "models.xgboost.filename",
        "xgb_early_stopping": "models.xgboost.init_params.early_stopping_rounds",
        "xgb_grid_learning_rate": "models.xgboost.param_grid.learning_rate",
        "xgb_grid_n_estimators": "models.xgboost.param_grid.n_estimators",
        "xgb_grid_max_depth": "models.xgboost.param_grid.max_depth",
        "xgb_grid_min_child_weight": "models.xgboost.param_grid.min_child_weight",
        "xgb_grid_gamma": "models.xgboost.param_grid.gamma",
        "xgb_grid_reg_alpha": "models.xgboost.param_grid.reg_alpha",
        "xgb_grid_reg_lambda": "models.xgboost.param_grid.reg_lambda",
        "xgb_grid_subsample": "models.xgboost.param_grid.subsample",
        "xgb_grid_colsample_bytree": "models.xgboost.param_grid.colsample_bytree",
        "xgb_grid_colsample_bylevel": "models.xgboost.param_grid.colsample_bylevel",
        "xgb_grid_colsample_bynode": "models.xgboost.param_grid.colsample_bynode",
        "xgb_metric": "models.xgboost.eval_metric",


        # models.stacked_ensemble
        "ensemble_filename": "models.stacked_ensemble.filename",
        "ensemble_max_iter": "models.stacked_ensemble.logistic_regression.max_iter",

        # run option 
        "raw_count": "raw_count",
    },
    frozen_prefixes=(
        ("runtime", "imbalance"),
        ("runtime", "per_fold_estimators"),
        ("runtime", "tuning"),
    ),
    # Existing tuning controls must work in nested and flat forms alike.
    allowed_frozen_paths=(
        ("runtime", "tuning", "inner_cv"),
        ("runtime", "tuning", "enabled"),
        ("runtime", "tuning", "method"),
    ),
)


# ---------------- internal helpers ----------------

def _deep_set(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = d
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value

def _deep_get(d: Dict[str, Any], path: List[str]) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _has_value(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, str) and x.strip() == "":
        return False
    if isinstance(x, (list, tuple, set, dict)) and len(x) == 0:
        return False
    return True

def _is_model_type(tp: Any) -> bool:
    try:
        return isinstance(tp, type) and issubclass(tp, BaseModel)
    except TypeError:
        return False


def _model_type(tp):
    if _is_model_type(tp):
        return tp
    return next((arg for arg in get_args(tp) if _is_model_type(arg)), None)


def _validate_override_paths(values, model_cls=Config, prefix=()):
    """Validate user input recursively, including optional schema blocks.

    Saved configurations are deliberately not passed through this check: older
    bundles may contain historical fields which do not affect prediction.
    """
    leaves = []
    for key, value in values.items():
        path = prefix + (key,)
        if path == ("raw_count",):
            continue  # Existing CLI runtime option, not a Config field.
        field = model_cls.model_fields.get(key)
        if field is None:
            raise ConfigError(f"Unknown override field: {'.'.join(path)}")
        nested = _model_type(field.annotation)
        if isinstance(value, BaseModel):
            value = value.model_dump(exclude_unset=True)
            values[key] = value
        if nested is not None and isinstance(value, dict):
            # An explicit empty optional block instantiates its schema defaults;
            # omitting it could instead select data-dependent search defaults.
            leaves.extend(_validate_override_paths(value, nested, path) if value else [path])
        else:
            leaves.append(path)
    return leaves


def _collect_leaf_paths(model_cls: Type[BaseModel], prefix: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    """
    Leaf path = path ending in a non-BaseModel field.
    Used so `inner_cv` can map to a unique full path automatically.
    """
    out: List[Tuple[str, ...]] = []
    for fname, finfo in model_cls.model_fields.items():
        ann = finfo.annotation
        if _is_model_type(ann):
            out.extend(_collect_leaf_paths(ann, prefix + (fname,)))
        else:
            out.append(prefix + (fname,))
    return out


def _short_index(leaf_paths: List[Tuple[str, ...]]) -> Dict[str, List[Tuple[str, ...]]]:
    idx: Dict[str, List[Tuple[str, ...]]] = {}
    for p in leaf_paths:
        idx.setdefault(p[-1], []).append(p)
    return idx


def _is_under_frozen(path: Tuple[str, ...], frozen_prefixes: Tuple[Tuple[str, ...], ...]) -> bool:
    return any(path[: len(fp)] == fp for fp in frozen_prefixes)


# ---------------- public API ----------------

def normalize_user_overrides(user_kwargs: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
    """
    Convert a flat/aliased user override dict into the nested override dict
    expected by loaders._deep_update.

    Supports:
      - dotted keys: "runtime.split.train_ratio"
      - aliases: "split_ratio" -> "runtime.split.train_ratio"
      - short keys: "inner_cv" -> resolved if unique in schema

    Blocks overrides that target SPEC.frozen_prefixes.
    """
    leaf_paths = _collect_leaf_paths(Config)
    short_idx = _short_index(leaf_paths)

    out: Dict[str, Any] = {}
    errors: List[str] = []

    for key, val in user_kwargs.items():
        # 1) dotted path
        if "." in key:
            path = tuple(key.split("."))
        # 2) alias
        elif key in SPEC.aliases:
            path = tuple(SPEC.aliases[key].split("."))
        elif key in Config.model_fields:
            path = (key,)
        # 3) short key 
        else:
            matches = short_idx.get(key, [])
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) == 0:
                errors.append(f"Unknown override key '{key}'. Use dotted path or add alias in overrides.py")
                continue
            else:
                errors.append(
                    f"Ambiguous override key '{key}' matches: "
                    + ", ".join(".".join(p) for p in matches)
                    + ". Use a dotted path or add alias in overrides.py"
                )
                continue

        candidate = {}
        _deep_set(candidate, list(path), deepcopy(val))
        try:
            leaves = _validate_override_paths(candidate)
            for leaf in leaves:
                if (_deep_get(candidate, list(leaf)) != {}
                        and _is_under_frozen(leaf, SPEC.frozen_prefixes)
                        and leaf not in SPEC.allowed_frozen_paths):
                    raise ConfigError(f"Override blocked (frozen): {'.'.join(leaf)}")
                if leaf[-1] == "early_stopping_rounds" and _deep_get(candidate, list(leaf)) not in (None, 0, False, "0"):
                    raise ConfigError(
                        f"Unsupported override {'.'.join(leaf)}: early stopping is disabled by the existing trainer; use the n_estimators search space"
                    )
            # Merge leaves so mixed nested/dotted input cannot silently erase
            # another setting. Conflicting values are errors, not order-dependent.
            for leaf in leaves + ([("raw_count",)] if path == ("raw_count",) else []):
                for length in range(1, len(leaf)):
                    parent = _deep_get(out, list(leaf[:length]))
                    if parent is not None and not isinstance(parent, dict):
                        raise ConfigError(f"Conflicting overrides for {'.'.join(leaf[:length])}")
                value = _deep_get(candidate, list(leaf))
                previous = _deep_get(out, list(leaf))
                if previous is not None and previous != value:
                    raise ConfigError(f"Conflicting overrides for {'.'.join(leaf)}")
                _deep_set(out, list(leaf), value)
        except ConfigError as exc:
            errors.append(str(exc))

    # sync batch_correction.design_columns <-> de.covariates
    design_path = ["batch_correction", "design_columns"]
    de_path = ["de", "covariates"]

    design_val = _deep_get(out, design_path)
    de_val = _deep_get(out, de_path)

    design_has = _has_value(design_val)
    de_has = _has_value(de_val)

    if design_has and not de_has:
        _deep_set(out, de_path, design_val)
    elif de_has and not design_has:
        _deep_set(out, design_path, de_val)

    if errors and strict:
        raise ConfigError("Bad user overrides:\n- " + "\n- ".join(errors))

    return out
