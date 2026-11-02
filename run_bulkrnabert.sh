#!/usr/bin/env bash
# Run task/representation combinations through a bounded parallel worker pool.
set -euo pipefail
repo=$(realpath "$(dirname "$0")")
if [[ ${1:-} == --help ]]; then
    cat <<'EOF'
Usage: bash run_bulkrnabert.sh [--dry-run] [api.py arguments...]
Environment:
  MAX_PARALLEL=4       Maximum simultaneous task/variant runs
  N_JOBS=4             Classifier workers per run
  TASKS="ball cll brca apolo"
  VARIANTS="full pca supervised supervised_pca"
  LOG_DIR=path         Default: a new timestamped directory under test_bulkrnabert/logs
  OMICLASS_PYTHON=path Python in the Omiclass environment (default: python)
  BULKRNABERT_PYTHON=path  Python in the extraction environment
Example: MAX_PARALLEL=2 N_JOBS=8 TASKS="brca cll" bash run_bulkrnabert.sh
EOF
    exit 0
fi
dry_run=0
if [[ ${1:-} == --dry-run ]]; then dry_run=1; shift; fi
MAX_PARALLEL=${MAX_PARALLEL:-4}
N_JOBS=${N_JOBS:-4}
for value in "$MAX_PARALLEL" "$N_JOBS"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_PARALLEL and N_JOBS must be positive integers.' >&2; exit 2; }
done
read -r -a tasks <<< "${TASKS-ball cll brca apolo}"
read -r -a variants <<< "${VARIANTS-full pca supervised supervised_pca}"
(( ${#tasks[@]} && ${#variants[@]} )) || { echo 'Select at least one task and variant.' >&2; exit 2; }
scripts=()
declare -A seen=()
for task in "${tasks[@]}"; do
    case $task in ball|cll|brca|apolo) ;; *) echo "Unknown task: $task" >&2; exit 2;; esac
    for variant in "${variants[@]}"; do
        case $variant in full|pca|supervised|supervised_pca) ;; *) echo "Unknown variant: $variant" >&2; exit 2;; esac
        key=$task/$variant
        [[ -z ${seen[$key]:-} ]] || continue
        seen[$key]=1
        suffix=_$variant
        [[ $variant != full ]] || suffix=
        scripts+=("$repo/final_results_runs/test_bulkrnabert/$task/run_${task}${suffix}.sh")
    done
done
# A single explicit replacement/prepared directory cannot describe several tasks.
if (( ${#tasks[@]} > 1 )) && [[ -n ${BULKRNABERT_PREPARED_DIR:-}${BULKRNABERT_TPM_INPUT:-} ]]; then
    echo 'Use TASKS with one task when overriding TPM input or prepared directory.' >&2
    exit 2
fi
if (( dry_run )); then
    for script in "${scripts[@]}"; do
        printf '%q ' bash "$script" --override n_jobs "$N_JOBS" "$@"
        printf '\n'
    done
    exit 0
fi
command -v flock >/dev/null
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
if [[ -z ${LOG_DIR:-} ]]; then
    mkdir -p "$repo/final_results_runs/test_bulkrnabert/logs"
    LOG_DIR=$(mktemp -d "$repo/final_results_runs/test_bulkrnabert/logs/$(date +%Y%m%d-%H%M%S)-XXXXXX")
fi
mkdir -p "$LOG_DIR"
export LOG_DIR
echo "Running ${#scripts[@]} combinations; concurrency=$MAX_PARALLEL, n_jobs=$N_JOBS; logs=$LOG_DIR"
# xargs returns nonzero if any worker fails, while allowing other jobs to finish.
printf '%s\0' "${scripts[@]}" | xargs -0 -n 1 -P "$MAX_PARALLEL" bash -c '
    script=${!#}
    args=("${@:1:$#-1}")
    name=$(basename "$script" .sh)
    echo "START $name"
    if bash "$script" "${args[@]}" >"$LOG_DIR/$name.log" 2>&1; then
        echo "PASS  $name"
    else
        echo "FAIL  $name (see $LOG_DIR/$name.log)" >&2
        exit 1
    fi
' _ --override n_jobs "$N_JOBS" "$@"
