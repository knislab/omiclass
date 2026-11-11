# Examples using your own permitted inputs

Install Omiclass first; no research-server checkout, bundled dataset or pretrained model is assumed. Prepared expression matrices have features in rows and samples in columns; metadata has one row per sample with a `class` column. This minimal example performs no raw-count normalization; prepare the expression scale consistently first. Markers are one feature ID per line. Identifiers must follow the [input contracts](../README.md).

Train from the repository root:

```bash
python examples/run_python_example.py \
  --expression inputs/expression.csv --metadata inputs/metadata.csv \
  --markers inputs/markers.txt \
  --output outputs
```

The script prints the actual timestamped run directory. Save that value; do not substitute an old folder of HTML reports. Apply the core model using compatible prepared-expression inputs and metadata:

```bash
python examples/run_python_re_run_example.py \
  --expression inputs/prediction_expression.csv --metadata inputs/prediction_metadata.csv \
  --run-dir outputs/ACTUAL_COMPLETED_RUN
```

The rerun script reuses the saved input mode without overriding it, and rejects foundation runs (use `api.predict` for those). It requires `leaderboard.csv` and `config.freeze.yaml` before invoking the existing prediction workflow. Fitted model files are also required by the framework. It writes the existing core prediction/report outputs under that run. The Bash wrappers accept the same arguments and quote paths safely.

For Docker, build `omiclass:dev` from the repository root first. Set `INPUT_DIR` and `OUTPUT_DIR` and run `bash examples/run_docker_example.sh`. Supply `expression.csv`, `metadata.csv` and `markers.txt` in the input directory. For core prediction, set `INPUT_DIR` and `RUN_DIR` to the actual completed run and invoke `run_docker_re_run_example.sh`. Inputs are mounted read-only; the saved core run is writable for prediction reports. Docker provenance is kept in the mounted output/run.

These examples use the existing marker workflow without DE/batch correction, without additional log transformation or marker reranking. Older runs trained with `raw_count=True`, DE disabled and batch correction disabled could record ready-expression mode; that historical behavior is preserved for saved models. New raw-count training now honors the flag using the existing normalization, with verified gene lengths required. The full DE workflow requires native R, and explicitly transductive correction/frozen references have their own [preprocessing requirements](../docs/reference_preprocessing.md). Foundation training and `api.predict` use the separate [foundation guide](../docs/foundation-api.md) and authorized external assets. No architecture, search space or trained results are changed by these wrappers.

Historical example tables, images and generated HTML are excluded from publication pending permission/provenance review. No replacement annotation is invented. For raw-count DE workflows, supply verified lengths explicitly; automatic bundled annotation fallback is unavailable in the candidate until resource permissions are resolved.
