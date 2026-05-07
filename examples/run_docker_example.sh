# Build the Docker image for omiclass
docker build -t omiclass:dev ..

# Use the examples/data folder
data_path=$(realpath data)
results_path=$(realpath results)

mkdir -p "$results_path"

docker run --rm \
  -v "$data_path:/app/data" \
  -v "$results_path:/app/results" \
  omiclass:dev \
  --override expression_file /app/data/raw_counts.csv \
  --override output_dir /app/results \
  --override meta_path /app/data/patients_meta.csv \
  --override class_col class \
  --override batch_column batch \
  --override de_covariates age,sex \
  --override n_jobs 5 \
  --override gene_lengths_path /app/data/gene_lengths.csv
