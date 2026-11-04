# Reproduce the manuscript exports and results map

For the current Figure 5 revision only, regenerate both heatmap layouts with:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ls/katzm/.conda/envs/viz_env/bin/python \
  /data01/private/projects/omiclass/manuscript_figures/scripts/separate_legends_figures_3_5.py --figures 5

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ls/katzm/.conda/envs/viz_env/bin/python \
  /data01/private/projects/omiclass/manuscript_figures/scripts/plot_figure_5_horizontal.py
```

Both layouts hide `Mean pooled` for corrected and uncorrected inputs at display
time. `Full` and `Full tokens` retain their original values. The saved source
tables and experiments are read only. See [../Figure_5/README.md](../Figure_5/README.md)
for the exact pooling distinction, especially BulkFormer's max+mean+median
aggregation outside B-ALL. The merged collective remains legend-free.
The `--figures` option leaves other manuscript figures and their per-figure
validation records untouched; the shared validation catalogue retains them.

Use the existing Python 3.9.19 visualization environment. Its package versions
are recorded in `requirements.txt`; Poppler's `pdffonts` must also be on PATH.
The plotting script additionally uses PyMuPDF and Pillow to validate its exports.
Run from any working directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ls/katzm/.conda/envs/viz_env/bin/python \
  /data01/private/projects/omiclass/manuscript_figures/scripts/plot_figure_2.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ls/katzm/.conda/envs/viz_env/bin/python \
  /data01/private/projects/omiclass/manuscript_figures/scripts/separate_legends_figures_3_5.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ls/katzm/.conda/envs/viz_env/bin/python \
  /data01/private/projects/omiclass/manuscript_figures/scripts/generate_results_map.py
```

Export the plots before generating the map so that it includes the final files,
standalone legend pieces, and validation records. The scripts accept `--project-root`; their default
root is derived from their own location, not the current working directory.
The plotting scripts default to 400 dpi and reject values below 300 dpi.
Use `--no-individual-panels` to export only the assembled Figure 2 in a fresh
output directory; this flag does not remove any panels from a previous run.

`plot_figure_2.py` imports the source-reading functions from
`generate_results_map.py`. It reads original saved predictions and configuration
files, recomputes accuracy and macro F1/recall/precision, verifies the pinned
five seeds and CV-selected model, checks evaluation IDs and paired conditions,
and verifies the original saved means and sample SDs before plotting.
It writes only to `manuscript_figures/Figure_2/`. The map generator writes only
`manuscript_figures/results_map.json`. Neither script calls original plotting
scripts, loads fitted estimators, performs inference, trains a model, acquires
experiment locks, or controls any experiment processes.

The generator fails on an inconsistent pinned manuscript result. Incomplete
alternative runs are retained as unresolved records with missing metrics set
to JSON `null`. Additional foundation jobs use finalized completion pointers;
identical transfer copies are deduplicated by job ID, completion fingerprint,
and result CSV hash, with all equivalent paths retained. Distinct retries are
not silently collapsed or averaged into the published figure.

The map is a snapshot: rerunning it updates the catalogue when additional
experiments finish. It preserves original figure contributions separately from
additional repetitions and unfinalized jobs. Source file hashes make each
snapshot traceable. Byte-identical PDF files are not promised across reruns
because PDF creation timestamps can differ; plotted data and geometry are
determined by the verified source results and script.

`separate_legends_figures_3_5.py` overwrites only the manuscript copies of
Figures 3–5, including the separate-metric Figure 3 alternatives and the existing
three-page Figure 4 collection. It writes standalone shared legend/notes pieces
once per figure directory. Original result files and original plotting scripts
are read only. Figure 4 uses saved UMAP coordinates and preserved original vector
panels, without fitting anything. Its ticks and KP annotations use embedded
TrueType fonts. Figure 5 uses the exact saved heatmap table and retains the side
colorbars. Only specifically named pure drawing helpers are loaded from the
original scripts; their import blocks and exporting entrypoints are not executed.
The generator records the current export manifest separately from the initial
copied-file inventory so that historical file hashes remain identifiable.

Example navigation:

```python
import json
from pathlib import Path

root = Path('/data01/private/projects/omiclass/manuscript_figures')
data = json.loads((root / 'results_map.json').read_text())
group = (data['figures']['figure_2']['panels']['b']['experiments']['omiclass']
         ['conditions']['corrected']['configurations']['marker_selected'])
seed_11 = group['repetitions']['seed_11']
print(seed_11['prediction_file'], seed_11['metrics'])
print(group['summary'])
```

Map metric values are fractions from 0 to 1. Figure 2's summary CSV is explicitly
in percentage points; its per-seed CSV uses fractions. Error bars use sample SD
across seeds, with `ddof=1`, not standard error or pooled prediction scores.
