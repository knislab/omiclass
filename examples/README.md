
# Examples

This folder contains minimal examples for running OmiClass with either Python or Docker.

## Run with Python

```bash
python examples/run_python_example.py
```

## Run with Docker

```bash
run_docker_example.sh
```
# Parameters

### Main parameters

These parameters define the main input files, output location, label column, batch column, and basic runtime behavior.

| Parameter | Explanation | Values |
|---|---|---|
| `expression_file` | Path to the main expression matrix. This is passed as the first argument to `run()`, not inside `kwargs`. | Path to CSV/TSV file |
| `output_dir` | Directory where all results and reports will be saved. | Path |
| `meta_path` | Path to the metadata file. | Path to CSV/TSV file |
| `class_col` | Metadata column used as the classification label. | Column name |
| `classification_column` | Alias for `class_col`. Metadata column used as the classification label. | Column name |
| `batch_column` | Metadata column representing the batch/cohort variable. Used for batch correction and batch-effect assessment. | Column name |
| `de_covariates` | Covariates included in differential expression analysis. | Comma-separated column names, e.g. `age,sex` |
| `n_jobs` | Number of CPU jobs/threads to use. | Integer |
| `project_name` | Name of the project/run. Used in reports and output organization. | Text string |
| `classes_names` | Optional display names for the classes in plots and reports. | List of class names |
| `random_state` | Random seed for reproducibility. | Integer |
| `cv_folds` | Number of cross-validation folds. | Integer |
| `split_ratio` | Fraction of samples used for training when splitting the data. | Float between 0 and 1 |
| `split_method` | Method used to split the dataset. | Example: `stratified`, `random` |
| `log_transform` | Whether to apply log transformation to the expression matrix. | `true` / `false` |
| `raw_count` | Whether the input expression matrix contains raw counts. | `true` / `false` |
| `test_holdout_path` | Optional path to an external holdout/test dataset. | Path |
| `markers_path` | Path to predefined marker genes or gene sets. | Path |


### Advanced / optional parameters

These parameters control advanced behavior such as batch-correction details, feature selection, survival prediction, differential expression thresholds, normalization, and model-specific search spaces.

| Parameter | Explanation | Values |
|---|---|---|
| `split_output_dir` | Directory for saving split train/test data. | Path |
| `design_columns` | Metadata columns to preserve during batch correction. | Column name or list of columns |
| `batch_covariates_path` | Path to additional covariates used for batch correction. | Path |
| `batch_n_pcs` | Number of principal components to evaluate during batch assessment. | Integer |
| `batch_fill_na` | Whether/how to fill missing values in batch covariates. | Boolean or fill strategy |
| `min_features` | Minimum number of features allowed in feature-based steps. | Integer |
| `bayesian_iterations` | Number of Bayesian optimization iterations. | Integer |
| `feature_selection_enabled` | Enable or disable feature selection. | `true` / `false` |
| `feature_selection_min_features` | Minimum number of features to keep during feature selection. | Integer |
| `feature_selection_max_features` | Maximum number of features to keep during feature selection. | Integer |
| `feature_selection_subset_fracs` | Fractions of features to test during feature selection. | List of floats |
| `feature_selection_consensus_top_n` | Number of top features considered for consensus selection. | Integer |
| `feature_selection_consensus_min_fraction` | Minimum fraction required for a feature to be included in the consensus set. | Float between 0 and 1 |
| `feature_selection_consensus_min_folds` | Minimum number of folds in which a feature must appear. | Integer |
| `feature_selection_consensus_min_genes` | Minimum number of genes required in the final consensus set. | Integer |
| `survival_prediction_enabled` | Enable or disable survival prediction analysis. | `true` / `false` |
| `survival_time_column` | Metadata column containing survival time. | Column name |
| `survival_event_column` | Metadata column containing event status. | Column name |
| `survival_prediction_expression_path` | Path to the expression matrix used for survival prediction. | Path |
| `survival_prediction_time_scale` | Time scale used for survival analysis. | Example: `days`, `months`, `years` |
| `feature_reduction_enabled` | Enable or disable feature reduction analysis. | `true` / `false` |
| `feature_reduction_ratios` | Ratios of selected features to test during feature reduction. | List of floats |
| `feature_reduction_repeats` | Number of repeats for feature reduction analysis. | Integer |
| `feature_reduction_min_features` | Minimum number of features allowed during feature reduction. | Integer |
| `de_min_samples` | Minimum sample fraction required for differential expression analysis. | Float between 0 and 1 |
| `de_k_genes` | Number of top differentially expressed genes to report. | Integer |
| `de_logfc` | Log fold-change threshold for differential expression. | Float |
| `de_fdr` | FDR threshold for differential expression. | Float, e.g. `0.05` |
| `gene_lengths_path` | Path to gene length annotation file used for normalization. | Path |
| `gene_lengths_version` | Gene annotation version used for gene lengths. | Example: `v19`, `v34` |
| `gencode_version` | GENCODE annotation version. | Example: `v19`, `v34` |
| `gtf_path` | Path to a GTF annotation file. | Path |
| `surface_markers_enabled` | Enable or disable surface marker annotation. | `true` / `false` |
| `rf_oob_score` | Enable out-of-bag scoring for Random Forest. | `true` / `false` |
| `rf_calibrate` | Enable probability calibration for Random Forest. | `true` / `false` |
| `rf_calibration_method` | Calibration method for Random Forest probabilities. | Example: `sigmoid`, `isotonic` |
| `rf_search_n_estimators` | Search space for the number of trees in Random Forest. | List or range of integers |
| `rf_search_max_depth` | Search space for maximum tree depth in Random Forest. | List of integers or `None` |
| `rf_search_max_features` | Search space for the number of features considered at each split. | Example: `sqrt`, `log2`, float |
| `rf_search_min_samples_leaf` | Search space for minimum samples per leaf. | List of integers |
| `rf_search_min_samples_split` | Search space for minimum samples required to split a node. | List of integers |
| `lgbm_early_stopping` | Early stopping rounds for LightGBM. | Integer |
| `lgbm_metric` | Evaluation metric for LightGBM. | Example: `multi_logloss`, `binary_logloss` |
| `lgbm_grid_learning_rate` | LightGBM learning-rate search values. | List of floats |
| `lgbm_grid_num_leaves` | LightGBM number-of-leaves search values. | List of integers |
| `lgbm_grid_max_depth` | LightGBM maximum-depth search values. | List of integers |
| `lgbm_grid_feature_fraction` | Fraction of features used by LightGBM. | List of floats between 0 and 1 |
| `lgbm_grid_bagging_fraction` | Fraction of samples used for LightGBM bagging. | List of floats between 0 and 1 |
| `lgbm_grid_n_estimators` | Number of LightGBM trees/boosting rounds. | List of integers |
| `lgbm_grid_min_child_samples` | Minimum number of samples per LightGBM leaf. | List of integers |
| `lgbm_grid_bagging_freq` | Frequency of LightGBM bagging. | List of integers |
| `lgbm_grid_lambda_l1` | L1 regularization values for LightGBM. | List of floats |
| `lgbm_grid_lambda_l2` | L2 regularization values for LightGBM. | List of floats |
| `xgb_filename` | Filename used for saving/loading XGBoost model artifacts. | Filename |
| `xgb_early_stopping` | Early stopping rounds for XGBoost. | Integer |
| `xgb_grid_learning_rate` | XGBoost learning-rate search values. | List of floats |
| `xgb_grid_n_estimators` | Number of XGBoost trees/boosting rounds. | List of integers |
| `xgb_grid_max_depth` | XGBoost maximum-depth search values. | List of integers |
| `xgb_grid_min_child_weight` | XGBoost minimum child weight search values. | List of floats or integers |
| `xgb_grid_gamma` | XGBoost minimum loss reduction values. | List of floats |
| `xgb_grid_reg_alpha` | XGBoost L1 regularization values. | List of floats |
| `xgb_grid_reg_lambda` | XGBoost L2 regularization values. | List of floats |
| `xgb_grid_subsample` | Fraction of samples used by XGBoost. | List of floats between 0 and 1 |
| `xgb_grid_colsample_bytree` | Fraction of features sampled per tree. | List of floats between 0 and 1 |
| `xgb_grid_colsample_bylevel` | Fraction of features sampled per tree level. | List of floats between 0 and 1 |
| `xgb_grid_colsample_bynode` | Fraction of features sampled per split node. | List of floats between 0 and 1 |
| `xgb_metric` | Evaluation metric for XGBoost. | Example: `mlogloss`, `logloss`, `auc` |
| `ensemble_filename` | Filename used for saving/loading stacked ensemble artifacts. | Filename |
| `ensemble_max_iter` | Maximum iterations for the ensemble logistic regression model. | Integer |