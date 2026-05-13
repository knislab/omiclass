# Build the Docker image for omiclass
docker build -t omiclass:dev ..

# Use the examples/data folder
data_path=$(realpath data)
results_path=$(realpath results)

docker run --rm \
  -v "$data_path:/app/data" \
  -v "$results_path:/app/results" \
  omiclass:dev \
  --override expression_file /app/data/raw_counts_with_external.csv \
  --override output_dir /app/results/Docker_results \
  --override meta_path /app/data/patients_meta_with_external.csv \
  --override raw_count True