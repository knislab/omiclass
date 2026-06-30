# generic-classifier/config/schema.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

BatchMethod = Literal["batch", "design", "covariates", "pcs"]
Modality = Literal["rnaseq", "cptac_ms"]

# ---------- feature selection ----------
class FeatureSelectionCfg(BaseModel):
    enabled: bool = True
    subset_fracs: List[float] = Field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    min_features: int = 20
    max_features: Optional[int] = None 
    consensus_top_n: Optional[int] = None
    consensus_min_frac: float = 0.6
    consensus_min_folds: Optional[int] = 2
    consensus_min_genes: int = 20

# ---------- feature reduction ----------
class FeatureReductionCfg(BaseModel):
    enabled: bool = False
    k_ratios: List[float] = Field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    n_repeats: int = 3
    min_features: int = 10

# ---------- survival_prediction ----------
class SurvivalPredictionCfg(BaseModel):
    enabled: bool = False
    X_path: Optional[Path] = None
    time_col: Optional[str] = None
    event_col: Optional[str] = None
    time_scale: float = 365.25

# ---------- DE ----------
class DECfg(BaseModel):
    covariates: Optional[List[str]] = None
    include_batch_correction: bool = False
    min_samples_frac: float = 0.5
    geneset_selection: bool = True
    k_genes: int = 100
    logfc_thresh: float = 1.0
    fdr_thresh: float = 0.05
    min_count: Optional[int] = 10


class DEGeneSelectionCfg(BaseModel):
    min_genes: int = 200
    genes_per_class: Optional[int] = None
    min_genes_per_class: int = 20
    max_genes_per_class: int = 75
    max_genes: int = 600
    one_vs_rest_fraction: float = 0.5
    close_pair_fraction: float = 0.5
    close_pair_count: Optional[int] = None
    pca_components: int = 10

    @model_validator(mode="after")
    def _validate_gene_selection(self):
        if self.min_genes <= 0:
            raise ValueError("de_gene_selection.min_genes must be positive.")
        if self.genes_per_class is not None:
            if self.genes_per_class <= 0:
                raise ValueError("de_gene_selection.genes_per_class must be positive when set.")
            self.max_genes_per_class = self.genes_per_class
        if self.min_genes_per_class <= 0:
            raise ValueError("de_gene_selection.min_genes_per_class must be positive.")
        if self.max_genes_per_class <= 0:
            raise ValueError("de_gene_selection.max_genes_per_class must be positive.")
        if self.max_genes_per_class < self.min_genes_per_class:
            raise ValueError("de_gene_selection.max_genes_per_class must be >= min_genes_per_class.")
        if self.max_genes <= 0:
            raise ValueError("de_gene_selection.max_genes must be positive.")
        if self.max_genes < self.min_genes:
            raise ValueError("de_gene_selection.max_genes must be >= min_genes.")
        if self.one_vs_rest_fraction < 0 or self.close_pair_fraction < 0:
            raise ValueError("DE gene-selection fractions must be non-negative.")
        if self.one_vs_rest_fraction + self.close_pair_fraction <= 0:
            raise ValueError("At least one DE gene-selection fraction must be positive.")
        if self.close_pair_count is not None and self.close_pair_count <= 0:
            raise ValueError("de_gene_selection.close_pair_count must be positive when set.")
        if self.pca_components <= 0:
            raise ValueError("de_gene_selection.pca_components must be positive.")
        return self

# ---------- Normalization ----------
class NormalizationCfg(BaseModel):
    gene_lengths_path: Optional[Path] = None
    gtf_path: Optional[Path] = None


class CPTACMSPreprocessingCfg(BaseModel):
    max_missing_fraction: Optional[float] = None
    log2_transform: bool = False
    pseudocount: float = 1e-6
    imputation: Literal["none", "median_by_feature"] = "none"

    @field_validator("max_missing_fraction")
    @classmethod
    def _validate_missing_fraction(cls, v):
        if v is None:
            return v
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("max_missing_fraction must be between 0 and 1.")
        return float(v)

    @field_validator("pseudocount")
    @classmethod
    def _validate_pseudocount(cls, v):
        if float(v) <= 0:
            raise ValueError("pseudocount must be > 0.")
        return float(v)

# ---------- Surface markers ----------
class SurfaceMarkerCfg(BaseModel):
    enabled: bool = False


# ---------- plots ----------
class PlotsCfg(BaseModel):
    pc_pvalue_heatmap_max: Optional[float] = None

    @field_validator("pc_pvalue_heatmap_max")
    @classmethod
    def _validate_pc_pvalue_heatmap_max(cls, v):
        if v is None:
            return v
        if float(v) <= 0:
            raise ValueError("plots.pc_pvalue_heatmap_max must be positive when set.")
        return float(v)
    

# ---------- Project ----------
class ProjectCfg(BaseModel):
    name: str = "Omiclass"                     
    modality: Modality = "rnaseq"
    out_dir: Path = Field(default_factory=Path.cwd)
    split_data_dir: Optional[Path] = None
    classes_names: Optional[List[str]] = None 
    hold_out_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    cluster_column: Optional[str] = None
    geneset_path: Optional[Path] = None
    version: Optional[int] = 19


    # Validator to ensure paths are absolute/resolvable if needed 
    @model_validator(mode='after')
    def _validate_paths(self):
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        if self.split_data_dir is None:
            self.split_data_dir = self.out_dir
        return self
    
    @model_validator(mode="after")
    def _require_fields_if_enabled(self):
        if self.meta_path is None:
            raise ValueError("meta_path is required.")
        if self.cluster_column is None:
            raise ValueError("cluster_column from meta is required.")
        return self

    @field_validator("modality", mode="before")
    @classmethod
    def _validate_modality(cls, v):
        if v is None:
            return "rnaseq"
        if not isinstance(v, str):
            raise ValueError(
                "Unsupported project.modality value. Supported modalities: 'rnaseq', 'cptac_ms'."
            )
        normalized = v.strip().lower()
        allowed = {"rnaseq", "cptac_ms"}
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported project.modality '{v}'. Supported modalities: 'rnaseq', 'cptac_ms'."
            )
        return normalized
    

    

# ---------- batch correction ----------
class BatchCorrectionCfg(BaseModel):
    method: List[BatchMethod] = Field(default_factory=list)
    batch_column: Optional[str] = None
    design_columns: Optional[List[str]] = None
    covariates_path: Optional[Path] = None
    n_pcs: Optional[int] = None
    fill_na: Optional[bool] = None

    @field_validator("method", mode="before")
    @classmethod
    def _normalize_method(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("method")
    @classmethod
    def _validate_method_combo(cls, v: List[BatchMethod]) -> List[BatchMethod]:
        # de-duplicate while preserving order
        seen = {}
        v = [seen.setdefault(m, m) for m in v if m not in seen]

        allowed_sets = [
            set(), 
            {"batch"},
            {"batch", "design"},
            {"covariates"},
            {"covariates", "design"},
            {"pcs"},
            {"pcs", "design"},
        ]
        if set(v) not in allowed_sets:
            raise ValueError(
                "Invalid method combination. Allowed: "
                "{'batch'}, {'batch','design'}, {'covariates'}, "
                "{'covariates','design'}, {'pcs'} or {'pcs','design'}."
            )
        return v

    @field_validator("design_columns", mode="before")
    @classmethod
    def _normalize_design_columns(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("n_pcs")
    @classmethod
    def _check_n_pcs(cls, v):
        if v in (0, "0"):
            return None
        if v is not None and v <= 0:
            raise ValueError("n_pcs must be a positive integer.")
        return v
        
    @model_validator(mode="after")
    def _enforce_required_fields(self):
        # Normalize method to a list
        method = self.method or []

        if "batch" in method and not self.batch_column:
            raise ValueError("batch_column is required when method includes 'batch'.")
        if "design" in method and not self.design_columns:
            raise ValueError("design_columns is required when method includes 'design'.")
        if "covariates" in method and not self.covariates_path:
            raise ValueError("covariates_path is required when method includes 'covariates'.")
        if "pcs" in method and self.n_pcs is None:
            raise ValueError("n_pcs is required when method includes 'pcs'.")

        return self
# ---------- split ----------
class SplitCfg(BaseModel):
    method: Literal["stratified","random"] = "stratified"
    train_ratio: float = 0.8

    @model_validator(mode='after')
    def _validate_ratio(self):
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1.")
        return self

# ---------- tuning ----------
class TuningCfg(BaseModel):
    enabled: bool = True
    method: Literal["bayes", "random", "grid"] = "bayes"
    inner_cv: Optional[int] = None

# ---------- imbalance ----------
class ImbalanceCfg(BaseModel):
    model_config = ConfigDict(frozen=True)
    mild_threshold: float = 1.5  
    severe_threshold: float = 3.0  

# ---------- per_fold_estimators ----------
class PerFoldEstimatorsCfg(BaseModel):
    model_config = ConfigDict(frozen=True)
    method: Literal["median","mean"] = "median"
    pad: float = 0.05
    min_rounds: int = 50

# ---------- runtime ----------
class RuntimeCfg(BaseModel):
    random_state: int = 80
    n_jobs: int = 1
    scoring: str = "f1_macro"
    cv_folds: Optional[int] = None
    learning_rate: float = 0.1
    min_features: int = 20
    bayesian_iterations: int = 40
    log_transform: bool = True
    activation_method: Literal["sigmoid","softmax","auto","none"] = "sigmoid"
    split: SplitCfg = Field(default_factory=SplitCfg)
    imbalance: ImbalanceCfg = Field(default_factory=ImbalanceCfg)
    tuning: TuningCfg = Field(default_factory=TuningCfg)
    per_fold_estimators: PerFoldEstimatorsCfg = Field(default_factory=PerFoldEstimatorsCfg)
    feature_selection: FeatureSelectionCfg = Field(default_factory=FeatureSelectionCfg)
    survival_prediction: SurvivalPredictionCfg = Field(default_factory=SurvivalPredictionCfg)
    feature_reduction: FeatureReductionCfg = Field(default_factory=FeatureReductionCfg)

# ---------- calibrate ----------
class CalibrateCfg(BaseModel):
    calibrate: bool = True
    method: Literal["sigmoid","isotonic"] = "sigmoid"

# ---------- rf search space ----------
class RFSearchSpace(BaseModel):
    n_estimators: List[int] = Field(default_factory=lambda: [200, 600])
    max_depth: List[int] = Field(default_factory=lambda: [4, 24])
    min_samples_split: List[int] = Field(default_factory=lambda: [2, 20])
    min_samples_leaf: List[int] = Field(default_factory=lambda: [1, 2])
    max_features: List[Union[str, float]] = Field(default_factory=lambda: ["sqrt", "log2", 0.3, 0.5, 0.7])

# ---------- defaults random forest ----------
class RFDefaultsCfg(BaseModel):
    oob_score: bool = True
    verbose: int = 0
    calibration: CalibrateCfg = Field(default_factory=CalibrateCfg)
    search_space: Optional[RFSearchSpace] = None

# ---------- rf models ----------
class RFModelsCfg(BaseModel):
    filename: Optional[str] = None
    class_weight: Optional[Union[str, Dict[str, float]]] = None 

# ---------- rf variants ----------
class RFVariantCfg(BaseModel):
    plain_random_forest: RFModelsCfg = Field(default_factory=RFModelsCfg)
    weighted_random_forest: RFModelsCfg = Field(default_factory=RFModelsCfg)
    smote_random_forest: RFModelsCfg = Field(default_factory=RFModelsCfg)
    adasyn_random_forest: RFModelsCfg = Field(default_factory=RFModelsCfg)
    svm_smote_random_forest: RFModelsCfg = Field(default_factory=RFModelsCfg)
    bl_smote_random_forest: RFModelsCfg = Field(default_factory=RFModelsCfg)

    @model_validator(mode="after")
    def _set_default_filenames(self):
        defaults = {
            "plain_random_forest": "plain_random_forest.joblib",
            "weighted_random_forest": "weighted_random_forest.joblib",
            "smote_random_forest": "smote_random_forest.joblib",
            "adasyn_random_forest": "adasyn_random_forest.joblib",
            "svm_smote_random_forest": "svm_smote_random_forest.joblib",
            "bl_smote_random_forest": "bl_smote_random_forest.joblib",
        }

        for field, fname in defaults.items():
            model_cfg = getattr(self, field)
            if model_cfg.filename is None:
                model_cfg.filename = fname

        return self

# ---------- random_forest ----------
class RandomForestCfg(BaseModel):
    defaults: RFDefaultsCfg = Field(default_factory=RFDefaultsCfg)
    variants: RFVariantCfg = Field(default_factory=RFVariantCfg)

# ---------- light_gbm parm_grid ----------
class LightGBMParamGrid(BaseModel):
    learning_rate: List[float]
    num_leaves: List[int]
    max_depth: List[int]
    feature_fraction: List[float]
    bagging_fraction: List[float]
    n_estimators: List[int] 
    min_child_samples: List[int]
    bagging_freq: List[int]
    lambda_l1: List[float]
    lambda_l2: List[float]

# ---------- boosting init_params ----------
class BoostInitParams(BaseModel):
    verbosity: int = -1
    early_stopping_rounds: int = 10

# ---------- light_gbm ----------
class LightGBMCfg(BaseModel):
    filename: Optional[str] = None
    weighted_filename: Optional[str] = None
    init_params: BoostInitParams = Field(default_factory=BoostInitParams)
    param_grid: Optional[LightGBMParamGrid] = None
    eval_metric: str = "multi_logloss"
    metric: str = "multi_logloss"

    @model_validator(mode="after")
    def _default_filename(self):
        if self.filename is None:
            self.filename = "light_gbm.joblib"
        if self.weighted_filename is None:
            self.weighted_filename = "weighted_light_gbm.joblib"
        return self

# ---------- xgboost parm_grid ----------
class xgboostParamGrid(BaseModel):
    learning_rate: List[float]
    n_estimators: List[int]
    max_depth: List[int]
    min_child_weight: List[float]
    gamma: List[float]
    reg_alpha: List[float]
    reg_lambda: List[float]
    subsample: List[float]
    colsample_bytree: List[float]
    colsample_bylevel: List[float]
    colsample_bynode: List[float]

# ---------- xgboost ----------
class XGBoostCfg(BaseModel):
    filename: Optional[str] = None
    weighted_filename: Optional[str] = None
    init_params: BoostInitParams =  Field(default_factory=lambda: BoostInitParams(verbosity=0))
    param_grid: Optional[xgboostParamGrid] = None
    eval_metric: str = "mlogloss"
    
    @model_validator(mode="after")
    def _default_filename(self):
        if self.filename is None:
            self.filename = "xgboost.joblib"
        if self.weighted_filename is None:
            self.weighted_filename = "weighted_xgboost.joblib"
        return self
    
# ---------- logistic regression ----------
class LogisticRegressionCfg(BaseModel):
    max_iter: int = 500
    solver: str = "lbfgs"

# ---------- stacked_ensemble ----------
class logisticRegressionCfg(BaseModel):
    filename: Optional[str] = None
    logistic_regression: LogisticRegressionCfg = Field(default_factory=LogisticRegressionCfg)

    @model_validator(mode="after")
    def _default_filename(self):
        if self.filename is None:
            self.filename = "stacked_ensemble.joblib"
        return self


# ---------- models ----------
class ModelsCfg(BaseModel):
    random_forest: RandomForestCfg = Field(default_factory=RandomForestCfg)
    light_gbm: LightGBMCfg = Field(default_factory=LightGBMCfg)
    xgboost: XGBoostCfg = Field(default_factory=XGBoostCfg)
    stacked_ensemble: logisticRegressionCfg = Field(default_factory=logisticRegressionCfg)

# ---------- top-level Config ----------
class Config(BaseModel):
    PDF: bool = False
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    batch_correction: BatchCorrectionCfg = Field(default_factory=BatchCorrectionCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    de: DECfg = Field(default_factory=DECfg)
    de_gene_selection: DEGeneSelectionCfg = Field(default_factory=DEGeneSelectionCfg)
    normalization: NormalizationCfg = Field(default_factory=NormalizationCfg)
    cptac_ms_preprocessing: CPTACMSPreprocessingCfg = Field(default_factory=CPTACMSPreprocessingCfg)
    surface_markers: SurfaceMarkerCfg = Field(default_factory=SurfaceMarkerCfg)
    plots: PlotsCfg = Field(default_factory=PlotsCfg)

    @model_validator(mode="after")
    def _final_checks(self):
        # Add any cross-field validations here if needed
        return self
