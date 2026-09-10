#!/usr/bin/env bash
set -euo pipefail

python_bin=/opt/conda/envs/omiclass_env/bin/python
report_path=${OMICLASS_RUNTIME_REPORT:-/app/runtime_provenance.txt}

"$python_bin" /opt/omiclass-docker/runtime_provenance.py "$report_path"
exec "$python_bin" /app/api.py "$@"
