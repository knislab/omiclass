# generic-classifier/config/schema.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Literal, Dict
from pydantic import BaseModel, Field, model_validator, field_validator, model_serializer, ConfigDict

BatchMethod = Literal["batch", "design", "covariates", "pcs"]
Modality = Literal["rnaseq", "cptac_ms"]

# ---------- foundation-model feature extraction ----------
class EmbeddingReductionCfg(BaseModel):
    strategy: Literal["full", "supervised", "pca", "supervised_pca"] = "full"
    supervised_dimensions: int = 200
    pca_components: int = 50
    one_vs_rest_fraction: float = 0.5
    close_pair_fraction: float = 0.5
    close_pair_count: Optional[int] = None
    close_pair_pca_components: int = 10
    random_state: Optional[int] = None

    @model_validator(mode="after")
    def _validate_reduction(self):
        if self.supervised_dimensions <= 0:
            raise ValueError("Embedding supervised_dimensions must be positive.")
        if self.pca_components <= 0 or self.close_pair_pca_components <= 0:
            raise ValueError("Embedding PCA component counts must be positive.")
        if self.one_vs_rest_fraction < 0 or self.close_pair_fraction < 0:
            raise ValueError("Embedding reduction budget fractions must be non-negative.")
        if self.one_vs_rest_fraction + self.close_pair_fraction <= 0:
            raise ValueError("At least one supervised embedding budget fraction must be positive.")
        if self.close_pair_count is not None and self.close_pair_count <= 0:
            raise ValueError("Embedding close_pair_count must be positive when set.")
        return self


# Public compatibility name for existing configurations and imports.
BulkFormerReductionCfg = EmbeddingReductionCfg


class BulkFormerCfg(BaseModel):
    enabled: bool = False
    root_dir: Path = Path("BulkFormer")
    device: str = "auto"
    batch_size: int = 4
    aggregate_type: Literal["max", "mean", "median", "all"] = "mean"
    graph_path: Optional[Path] = None
    graph_weights_path: Optional[Path] = None
    gene_embedding_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    gene_info_path: Optional[Path] = None
    gene_lengths_path: Optional[Path] = None
    interested_gene_path: Optional[Path] = None
    reduction: BulkFormerReductionCfg = Field(default_factory=BulkFormerReductionCfg)

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v):
        if v is None:
            return "auto"
        if not isinstance(v, str) or not v.strip():
            raise ValueError("bulkformer.device must be a non-empty string.")
        return v.strip()

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v):
        if int(v) <= 0:
            raise ValueError("bulkformer.batch_size must be a positive integer.")
        return int(v)

    @model_validator(mode="after")
    def _set_default_paths(self):
        root = Path(self.root_dir)
        if self.graph_path is None:
            self.graph_path = root / "data" / "G_tcga.pt"
        if self.graph_weights_path is None:
            self.graph_weights_path = root / "data" / "G_tcga_weight.pt"
        if self.gene_embedding_path is None:
            self.gene_embedding_path = root / "data" / "esm2_feature_concat.pt"
        if self.checkpoint_path is None:
            self.checkpoint_path = root / "model" / "bulkformer_147M.pt"
        if self.gene_info_path is None:
            self.gene_info_path = root / "data" / "bulkformer_gene_info.csv"
        if self.gene_lengths_path is None:
            self.gene_lengths_path = root / "data" / "gene_length_df.csv"
        if self.interested_gene_path is None:
            self.interested_gene_path = root / "data" / "interested_gene_list.pt"
        return self


class CellFMCfg(BaseModel):
    """Frozen CellFM extraction settings for sample-level RNA-seq embeddings."""

    enabled: bool = False
    root_dir: Path = Path("CellFM/CellFM-torch")
    checkpoint_path: Optional[Path] = None
    gene_info_path: Optional[Path] = None
    vendor_dir: Optional[Path] = None
    device: str = "auto"
    batch_size: int = 1
    random_state: int = 0
    num_views: int = 20
    reduction: EmbeddingReductionCfg = Field(default_factory=EmbeddingReductionCfg)

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v):
        if v is None:
            return "auto"
        if not isinstance(v, str) or not v.strip():
            raise ValueError("cellfm.device must be a non-empty string.")
        return v.strip()

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v):
        if int(v) <= 0:
            raise ValueError("cellfm.batch_size must be a positive integer.")
        return int(v)

    @field_validator("num_views", mode="before")
    @classmethod
    def _validate_num_views(cls, v):
        if isinstance(v, bool):
            raise ValueError("cellfm.num_views must be a positive integer.")
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped.isdigit():
                raise ValueError("cellfm.num_views must be a positive integer.")
            v = int(stripped)
        if not isinstance(v, int) or v <= 0:
            raise ValueError("cellfm.num_views must be a positive integer.")
        return int(v)

    @model_validator(mode="after")
    def _set_default_paths(self):
        root = Path(self.root_dir)
        if self.checkpoint_path is None:
            self.checkpoint_path = root.parent / "weights" / "CellFM_80M_weight.ckpt"
        if self.gene_info_path is None:
            self.gene_info_path = root / "csv" / "expand_gene_info.csv"
        if self.vendor_dir is None:
            # The bundled checkout keeps binary dependencies next to CellFM-torch.
            self.vendor_dir = root.parent / "vendor"
        return self

class BulkRNABertCfg(BaseModel):
    """Isolated, frozen BulkRNABert extraction; input is explicitly unlogged TPM."""

    model_config = ConfigDict(protected_namespaces=())
    enabled: bool = False
    root_dir: Path = Path("BulkRNABert")
    model_name: str = "bulk_rna_bert_gtex_encode"
    checkpoint_directory: Optional[Path] = None
    vocabulary_path: Optional[Path] = None
    python_executable: Path = Field(
        default_factory=lambda: Path.home() / ".conda/envs/bulkrnabert/bin/python"
    )
    tpm_input_path: Optional[Path] = None
    cache_dir: Path = Path("final_results_runs/test_bulkrnabert/cache")
    device: Literal["auto", "cpu", "gpu"] = "auto"
    batch_size: int = Field(default=1, gt=0)
    reduction: EmbeddingReductionCfg = Field(default_factory=EmbeddingReductionCfg)

    @model_validator(mode="after")
    def _set_default_paths(self):
        if not self.model_name or Path(self.model_name).name != self.model_name:
            raise ValueError("bulkrnabert.model_name must be a checkpoint folder name.")
        if self.checkpoint_directory is None:
            self.checkpoint_directory = self.root_dir / "checkpoints"
        if self.vocabulary_path is None:
            self.vocabulary_path = self.root_dir / "data/bulkrnabert/common_gene_id.txt"
        return self


# ---------- feature selection ----------
class FeatureSelectionCfg(BaseModel):
    enabled: bool = True
    reduce_sampler_pipelines: bool = True
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
    close_pair_distance_method: str = "centroid"
    close_pair_knn_k: int = 5
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
        if self.close_pair_distance_method not in {"centroid", "knn"}:
            raise ValueError("de_gene_selection.close_pair_distance_method must be either 'centroid' or 'knn'.")
        if self.close_pair_knn_k <= 0:
            raise ValueError("de_gene_selection.close_pair_knn_k must be positive.")
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
    # `method` remains the legacy limma adjustment flags.
    algorithm: Literal["limma", "none", "frozen_quantile", "reference_ruv", "fsva_exact"] = "limma"
    input_scale: Literal["tpm", "log1p_tpm"] = "tpm"
    training_samples_path: Optional[Path] = None
    reference_path: Optional[Path] = None
    control_genes_path: Optional[Path] = None
    control_justification: Optional[str] = None
    n_factors: int = Field(default=1, gt=0)
    rscript_path: Optional[Path] = None
    r_library_path: Optional[Path] = None
    reference_random_state: int = 11
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
    ensemble: bool = False

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
    passthrough: bool = True
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

class ArticleHeadCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=8, gt=0)
    hidden_dims: List[int] = Field(default_factory=lambda: [64])
    token_dim: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.0001, ge=0)
    smote_k_neighbors: int = Field(default=5, gt=0)

    @field_validator("hidden_dims")
    @classmethod
    def _positive_widths(cls, values):
        if not values or any(v <= 0 for v in values):
            raise ValueError("hidden_dims must contain positive widths")
        return values


class BertHeadCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=32, gt=0)
    max_epochs: int = Field(default=200, gt=0)
    patience: int = Field(default=20, gt=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    min_delta: float = Field(default=0.0, ge=0)
    threads: int = Field(default=1, gt=0)
    smote_k_neighbors: int = Field(default=5, gt=0)


class FoundationCfg(BaseModel):
    """Opt-in manuscript protocols; legacy API calls remain on their original route."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    model: Literal["bulkformer", "cellfm", "cellfm_1view", "cellfm_20view", "bulkrnabert"] = "bulkformer"
    classifier: Literal["omiclass", "foundation_classifier", "lr", "svm", "mlp", "mlp_smote"] = "omiclass"
    representation: Literal["full", "supervised", "pca", "supervised_pca", "tokens"] = "full"
    pooling: Literal["mean", "max_mean_median", "max", "median"] = "mean"
    extraction_seed: int = 0
    input_scale: Literal["counts", "tpm"] = "counts"
    correction: Literal["none", "limma"] = "none"
    allow_transductive_correction: bool = False
    mode: Literal["train", "predict"] = "train"
    prediction_scope: Literal["external", "all"] = "external"
    run_dir: Optional[Path] = None
    split_path: Optional[Path] = None
    article_head: ArticleHeadCfg = Field(default_factory=ArticleHeadCfg)
    bert_head: BertHeadCfg = Field(default_factory=BertHeadCfg)
    lr_c_values: List[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])

    @field_validator("lr_c_values")
    @classmethod
    def _positive_c(cls, values):
        if not values or any(v <= 0 for v in values):
            raise ValueError("LR C values must be positive")
        return values

    @model_validator(mode="after")
    def _compatible(self):
        if not self.enabled:
            return self
        backend = "cellfm" if self.model.startswith("cellfm") else self.model
        if self.pooling != "mean" and backend != "bulkformer":
            raise ValueError("Only BulkFormer supports alternative pooling")
        if self.representation == "tokens" and self.classifier != "mlp":
            raise ValueError("Full tokens support the article MLP only; no token SVM/SMOTE/LR/Omiclass protocol")
        if self.representation == "tokens" and self.pooling != "mean":
            raise ValueError("Token representations do not take a pooled-vector aggregation")
        if self.input_scale == "tpm" and backend != "bulkrnabert":
            raise ValueError("TPM input is supported only for BulkRNABert; BulkFormer/CellFM require counts")
        if self.correction == "limma" and not self.allow_transductive_correction:
            raise ValueError("Historical limma fits the supplied cohort before splitting; explicitly set allow_transductive_correction=True")
        if self.mode == "predict" and self.run_dir is None:
            raise ValueError("Prediction requires foundation.run_dir")
        return self


# ---------- top-level Config ----------
class Config(BaseModel):
    PDF: bool = False
    foundation: FoundationCfg = Field(default_factory=FoundationCfg)
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    bulkformer: BulkFormerCfg = Field(default_factory=BulkFormerCfg)
    cellfm: CellFMCfg = Field(default_factory=CellFMCfg)
    bulkrnabert: BulkRNABertCfg = Field(default_factory=BulkRNABertCfg)
    batch_correction: BatchCorrectionCfg = Field(default_factory=BatchCorrectionCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    de: DECfg = Field(default_factory=DECfg)
    de_gene_selection: DEGeneSelectionCfg = Field(default_factory=DEGeneSelectionCfg)
    normalization: NormalizationCfg = Field(default_factory=NormalizationCfg)
    cptac_ms_preprocessing: CPTACMSPreprocessingCfg = Field(default_factory=CPTACMSPreprocessingCfg)
    surface_markers: SurfaceMarkerCfg = Field(default_factory=SurfaceMarkerCfg)
    plots: PlotsCfg = Field(default_factory=PlotsCfg)

    @model_validator(mode="before")
    @classmethod
    def _connect_foundation(cls, values):
        if not isinstance(values, dict):
            return values
        import copy
        data = copy.deepcopy(values)
        f = data.get("foundation", {})
        if isinstance(f, FoundationCfg):
            f = f.model_dump()
        if not isinstance(f, dict) or not f.get("enabled", False):
            return data
        settings = FoundationCfg(**f)
        if not settings.enabled:
            return data
        backend = "cellfm" if settings.model.startswith("cellfm") else settings.model
        enabled = [name for name in ("bulkformer", "cellfm", "bulkrnabert")
                   if data.get(name, {}).get("enabled", False)]
        if any(name != backend for name in enabled):
            raise ValueError("foundation.model conflicts with another enabled backend")
        encoder = data.setdefault(backend, {})
        encoder["enabled"] = True
        if backend == "cellfm":
            encoder.setdefault("batch_size", 4)
        if settings.model in {"cellfm_1view", "cellfm_20view"}:
            views = 1 if settings.model == "cellfm_1view" else 20
            if "num_views" in encoder and encoder["num_views"] != views:
                raise ValueError("CellFM model alias conflicts with num_views")
            encoder["num_views"] = views
        strategy = settings.representation if settings.representation != "tokens" else "full"
        reduction = encoder.setdefault("reduction", {})
        if "strategy" in reduction and reduction["strategy"] != strategy:
            raise ValueError("foundation.representation conflicts with backend reduction strategy")
        reduction["strategy"] = strategy
        if backend == "bulkformer":
            aggregate = "all" if settings.pooling == "max_mean_median" else settings.pooling
            if "aggregate_type" in encoder and encoder["aggregate_type"] != aggregate:
                raise ValueError("foundation.pooling conflicts with bulkformer.aggregate_type")
            encoder["aggregate_type"] = aggregate
        runtime = data.setdefault("runtime", {})
        fs = runtime.setdefault("feature_selection", {})
        if fs.get("enabled", False):
            raise ValueError("The manuscript foundation protocols require runtime.feature_selection.enabled=False")
        fs["enabled"] = False
        return data

    @model_serializer(mode="wrap")
    def _serialize_optional_bulkrnabert(self, handler):
        # Keep frozen configs/output structure identical for existing runs.
        data = handler(self)
        if not self.bulkrnabert.enabled:
            data.pop("bulkrnabert", None)
        if not self.foundation.enabled:
            data.pop("foundation", None)
        return data

    @model_validator(mode="after")
    def _final_checks(self):
        if self.foundation.enabled:
            if self.project.modality != "rnaseq":
                raise ValueError("Foundation protocols require RNA-seq")
            if self.foundation.correction == "limma":
                bc = self.batch_correction
                if bc.algorithm != "limma" or not bc.batch_column or bc.covariates_path or bc.n_pcs:
                    raise ValueError("Foundation correction requires observed cohort limma with a batch column, no covariate matrix/PC removal")
                if self.project.cluster_column in [bc.batch_column, *(bc.design_columns or [])]:
                    raise ValueError("Outcome labels cannot be correction covariates")
        if self.batch_correction.algorithm in {"frozen_quantile", "reference_ruv", "fsva_exact"}:
            if self.project.modality != "rnaseq" or self.bulkformer.enabled or self.cellfm.enabled or self.bulkrnabert.enabled:
                raise ValueError("Frozen expression references currently require rnaseq without embedding backends")
        if self.bulkrnabert.enabled:
            if self.bulkformer.enabled or self.cellfm.enabled:
                raise ValueError("BulkRNABert is an alternative embedding backend; enable only one.")
            if self.project.modality != "rnaseq":
                raise ValueError("bulkrnabert.enabled requires project.modality='rnaseq'.")
            if self.bulkrnabert.tpm_input_path is None and not self.foundation.enabled:
                raise ValueError("bulkrnabert.tpm_input_path must identify unlogged TPM input.")
            if self.bulkrnabert.reduction.strategy != "full" and self.runtime.feature_selection.enabled:
                raise ValueError("runtime.feature_selection.enabled must be false when using a non-full embedding reduction strategy.")
        if self.bulkformer.enabled and self.cellfm.enabled:
            raise ValueError("BulkFormer and CellFM are alternative embedding backends; enable only one.")
        if bool(getattr(self.bulkformer, "enabled", False)) and self.project.modality != "rnaseq":
            raise ValueError("bulkformer.enabled requires project.modality='rnaseq'.")
        if bool(getattr(self.cellfm, "enabled", False)) and self.project.modality != "rnaseq":
            raise ValueError("cellfm.enabled requires project.modality='rnaseq'.")
        if (
            self.de_gene_selection.close_pair_count is None
            and self.project.classes_names is not None
        ):
            self.de_gene_selection.close_pair_count = max(1, len(self.project.classes_names) - 1)
        if (
            (self.bulkformer.enabled or self.cellfm.enabled)
            and (self.bulkformer if self.bulkformer.enabled else self.cellfm).reduction.strategy != "full"
            and self.runtime.feature_selection.enabled
        ):
            raise ValueError(
                "runtime.feature_selection.enabled must be false when using a non-full "
                "embedding reduction strategy."
            )
        return self
