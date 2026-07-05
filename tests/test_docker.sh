docker build -t omiclass:dev ..

data_path=$(realpath /data/tmp_test)
results_path=$(realpath ../results)

mkdir -p "$results_path"

docker run --rm \
  -v "$data_path:/app/data" \
  -v "$results_path:/app/results" \
  omiclass:dev \
  --override expression_file /app/data/filtered_count.csv \
  --override output_dir /app/results \
  --override meta_path /app/data/filtered_meta_ighv.csv \
  --override class_col ighv_mut \
  --override batch_column cohort \
  --override de_covariates age,sex