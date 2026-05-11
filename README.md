# Omiclass

Omiclass is a configurable machine-learning framework for omics classification tasks. It trains and evaluates multiple classifiers on gene-expression style input data, supports imbalance handling methods, and writes model outputs and reports to a results directory.

The framework is designed to run either directly with Python or inside Docker, so the same analysis can be reproduced on a local machine, a lab server, or a clean container environment.

## Main features

- Expression-based classification from tabular input files
- Train/test splitting and cross-validation
- Bayesian hyperparameter optimization
- Random Forest, LightGBM, XGBoost and stacked classifer models
- Imbalance handling using class weights and oversampling strategies
- Output folders with trained models, predictions, metrics, and reports

## Schema
![Schema](examples/image/schema.png)

## Repository structure

```text
omiclass/
├── omiclass/              # Framework source code
├── examples/              # Example commands and small example config files
├── api.py.                # Framework entery point
├── Dockerfile             # Docker image definition
├── environment.yml.txt    # Dependencies
├── README.md              # This file
├── .gitignore
├── .dockerignore
└── LICENSE
```

## Input files

The framework expects at least two input tables:

1. **Expression file**
   - Rows should represent samples or patient according to the selected loader/configuration.
   - Columns should contain the features used for classification.
   - Example: `filtered_count.csv`

2. **Metadata file**
   - Contains sample-level labels and optional covariates.
   - Must include the classification label column.
   - Example: `filtered_meta.csv`

For example, for IGHV classification, the metadata file should contain a label column such as:

```text
ighv_mut
```

## Quick start with Docker

Build the Docker image from the repository root:

```bash
docker build -t omiclass:dev .
```

Create local folders for input and output:

```bash
mkdir -p data results
```

Place your input files in `data/`, for example:

```text
data/filtered_count.csv
data/filtered_meta_ighv.csv
```

Run the framework:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  omiclass:dev \
  --override expression_file /app/data/filtered_count.csv \
  --override meta_path /app/data/filtered_meta_ighv.csv \
  --override output_dir /app/results \
  --override class_col ighv_mut \
  --override de_covariates age,sex
```

The input files remain on your server/local machine. The results are written to the mounted `results/` folder.

### Rerun using Docker

When using Docker, mount the folder that contains the input data and the previous output directory, then pass the existing pipeline directory as `output_dir`.

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/results:/app/results" \
  omiclass:dev \
  --override expression_file /app/data/filtered_count.csv \
  --override output_dir /app/data/<Existing_pipeline> \
  --override meta_path /app/data/merged_meta_cll.csv
```

The mounted directory is shared with the server, so the updated reports and outputs will be written back to the existing pipeline folder.

## Quick start with Python

You can run Omiclass directly from Python by importing the `run` function from `api.py`.

Example file: `examples/python_example.py`

```python
from pathlib import Path

from api import run

base_dir = Path(__file__).resolve().parent.parent

expression_file = f"{base_dir}/data/filtered_counts.csv"

kwargs = {
    "output_dir": f"{base_dir}/results",
    "meta_path": f"{base_dir}/data/filtered_meta_ighv.csv",
    "class_col": "ighv_mut",
    "batch_column": "cohort",
    "de_covariates": ["age", "sex"],
    "n_jobs": 5,
}

run(expression_file, **kwargs)
```

From the project root:
```bash
cd examples
```
Create conda environment
```bash
conda env create -f ../environment.yml
conda activate omiclass_env
```
Run the example 
```bash
python python_example.py
```

The main input expression file is passed as the first argument to `run()`.  
Additional parameters are passed as keyword arguments through `**kwargs`.

## Quick start with Bash

You can run the `run` function from `api.py` from bash.
```bash
python api.py \
--override expression_file <data_path>/raw_counts.csv \
--override output_dir <data_path>/results \
--override meta_path <data_path>/patients_meta.csv \
--override class_col class \
--override batch_column batch \
--override de_covariates age,sex \
```

## Example folder

The `examples/` folder contains:

```text
examples/
├── README.md
├── run_python_example.py
├── run_docker_example.py
└── data/
    ├── hold_out_patients.txt
    ├── patients_meta.csv
    ├── raw_counts.csv
    └── README.md
```

Use the example script as a template and replace the input paths, label column, and covariates with the values for your project.

## Configuration
Parameters are set through the public alias names.

The same aliases can be used in two ways:

1. **Python API**: pass parameters as keyword arguments (`**kwargs`).
2. **Docker/CLI**: pass parameters with repeated `--override` arguments.

For example, the user-facing alias `output_dir` maps internally to `project.out_dir`, and `class_col` maps internally to `project.cluster_column`. The user should set the alias name, not the internal nested path.

Main fields includes:

```yaml
expression_file: path to the expression matrix
output_dir: path to the output directory
meta_path: path to the metadata file
class_col: metadata column used as the classification label
batch_column: metadata column used for batch correction
n_jobs: number of CPU jobs to use
```
