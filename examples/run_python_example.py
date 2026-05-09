import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api import run

base_dir = Path(__file__).resolve().parent
expression_file = f"{base_dir}/data/raw_counts.csv"
kwargs = {
    "output_dir": f"{base_dir}/results",
    "meta_path": f"{base_dir}/data/patients_meta.csv",
    "class_col": "class",
    "batch_column": "batch",
    "de_covariates": ["age", "sex"],
    "n_jobs": 5,
    "gene_lengths_path": f"{base_dir}/data/gene_lengths.csv",
}

run(expression_file, **kwargs)