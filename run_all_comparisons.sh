#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${COMPARISON_PYTHON:-python3}" "$ROOT/comparison_tools/shared/run.py" "$@"
