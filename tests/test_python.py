from ..api import run
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
expression_file = f"{base_dir}/data/filtered_count.csv"
kwargs = {
    "output_dir": f"{base_dir}/results",
    "meta_path": f"{base_dir}/data/filtered_meta_ighv.csv",
    "class_col": "ighv_mut",
    "batch_column": "cohort",
    "de_covariates": "age,sex",
    "n_jobs": 5
}

run(expression_file, **kwargs)