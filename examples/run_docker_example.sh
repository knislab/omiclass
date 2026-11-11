#!/usr/bin/env bash
set -euo pipefail
: "${INPUT_DIR:?Set INPUT_DIR to permitted expression.csv, metadata.csv, markers.txt inputs}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a writable output parent directory}"
mkdir -p "$OUTPUT_DIR"
docker run --rm -v "$(realpath "$INPUT_DIR"):/app/data:ro" -v "$(realpath "$OUTPUT_DIR"):/app/results" \
  omiclass:dev \
  --override expression_file /app/data/expression.csv \
  --override meta_path /app/data/metadata.csv \
  --override classification_column class \
  --override output_dir /app/results \
  --override raw_count false \
  --override log_transform false \
  --override feature_selection_enabled false \
  --override markers_path /app/data/markers.txt \
  --override run_de false \
  --override batch_correction.algorithm none
