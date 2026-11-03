#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash experiments/reference_preprocessing/run_later.sh METHOD [options]
METHOD: frozen_quantile | fsva_exact
Options:
  --screen NAME       Run detached in a named GNU screen session
  --workers N         Concurrent classifier seeds, 1-5 (default: 2)
  --cores-per-seed N  CPU cores per seed (default: 5)
  --cpus LIST         Method CPU set (quantile: 0-9; exact-fSVA: 10-19)
  --out DIR           New output directory (default: unique timestamped directory)
  --prepared DIR      Existing prepared input snapshot
  --python PATH       Compatible Omiclass Python executable
  --r-library DIR     R library containing sva (for fsva_exact)
  --help              Print help without starting anything

All five manuscript seeds run. Existing output directories are never reused.
Attach: screen -r NAME    Detach: Ctrl-A then D    Stop: Ctrl-C while attached
USAGE
}

if [[ $# -eq 0 || ${1-} == --help || ${1-} == -h ]]; then usage; exit 0; fi
method=$1
shift
case "$method" in frozen_quantile|fsva_exact) ;; *) usage >&2; exit 2 ;; esac
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd -- "$project_root"
prepared="$project_root/experiments/reference_preprocessing/outputs/strict_20261102_v2/prepared"
r_library="$project_root/experiments/reference_preprocessing/outputs/runtime_r_library"
python_executable=${OMICLASS_PYTHON:-/data01/home/ls/katzm/.conda/envs/viz_env/bin/python}
workers=2
cores_per_seed=5
cpu_list=""
output_dir=""
screen_name=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --screen|--workers|--cores-per-seed|--cpus|--out|--prepared|--python|--r-library)
      if [[ $# -lt 2 ]]; then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      case "$1" in
        --screen) screen_name=$2 ;;
        --workers) workers=$2 ;;
        --cores-per-seed) cores_per_seed=$2 ;;
        --cpus) cpu_list=$2 ;;
        --out) output_dir=$2 ;;
        --prepared) prepared=$2 ;;
        --python) python_executable=$2 ;;
        --r-library) r_library=$2 ;;
      esac
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
case "$workers" in [1-5]) ;; *) printf 'workers must be 1-5\n' >&2; exit 2 ;; esac
if [[ ! $cores_per_seed =~ ^[1-9][0-9]*$ ]]; then
  printf 'cores-per-seed must be a positive integer\n' >&2; exit 2
fi
if [[ ! -x "$python_executable" ]]; then printf 'Python executable unavailable: %s\n' "$python_executable" >&2; exit 2; fi
if [[ ! -f "$prepared/manifest.json" ]]; then printf 'Prepared input snapshot unavailable: %s\n' "$prepared" >&2; exit 2; fi
if [[ "$method" == fsva_exact && ! -d "$r_library/sva" ]]; then printf 'sva R library unavailable: %s\n' "$r_library" >&2; exit 2; fi
if [[ -z "$output_dir" ]]; then
  output_dir="$project_root/experiments/reference_preprocessing/outputs/manual_${method}_$(date +%Y%m%d_%H%M%S)_$$"
fi
if [[ -e "$output_dir" ]]; then printf 'Output directory must be new: %s\n' "$output_dir" >&2; exit 2; fi
command=("$python_executable" -u -m experiments.reference_preprocessing.launch_comparison
  --prepared "$prepared" --method "$method" --workers "$workers"
  --cores-per-seed "$cores_per_seed" --out "$output_dir")
if [[ -n "$cpu_list" ]]; then command+=(--cpus "$cpu_list"); fi
if [[ "$method" == fsva_exact ]]; then command+=(--r-library "$r_library"); fi
printf 'Output directory: %s\n' "$output_dir"
if [[ -n "$screen_name" ]]; then
  command -v screen >/dev/null || { printf 'GNU screen is unavailable\n' >&2; exit 2; }
  exec screen -dmS "$screen_name" "${command[@]}"
fi
exec "${command[@]}"
