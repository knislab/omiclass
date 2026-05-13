
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api import run

base_dir = Path(__file__).resolve().parent
expression_file = f"{base_dir}/data/raw_counts_with_external.csv"
kwargs = {
    "output_dir": f"{base_dir}/results/Python_results",
    "meta_path": f"{base_dir}/data/patients_meta_with_external.csv",
    "raw_count": True,
}

run(expression_file, **kwargs)