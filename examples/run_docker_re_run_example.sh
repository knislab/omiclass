#!/usr/bin/env bash
set -euo pipefail
: "${INPUT_DIR:?Set INPUT_DIR to the permitted prediction expression.csv and metadata.csv}"
: "${RUN_DIR:?Set RUN_DIR to the actual completed training run printed by training}"
test -f "$RUN_DIR/leaderboard.csv"
test -f "$RUN_DIR/config.freeze.yaml"
if test -f "$RUN_DIR/foundation.json"; then
  printf 'Use api.predict for foundation runs.\n' >&2
  exit 2
fi
docker run --rm -v "$(realpath "$INPUT_DIR"):/app/data:ro" -v "$(realpath "$RUN_DIR"):/app/run" \
  -e OMICLASS_RUNTIME_REPORT=/app/run/external_runtime_provenance.txt \
  omiclass:dev \
  --override expression_file /app/data/expression.csv \
  --override meta_path /app/data/metadata.csv \
  --override output_dir /app/run
