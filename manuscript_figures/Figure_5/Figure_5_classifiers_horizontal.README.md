# Figure 5 — horizontal classifiers

Compact layout revision: this script replaces its own horizontal exports only. The earlier panels, legends, notes, and manifests are unchanged.

Each `Figure_5[a-d]_*_classifiers_horizontal.pdf/png` has one Classifiers heading, with Omiclass, Logistic Regression, MLP, MLP + SMOTE, and SVM side by side in both the Accuracy and Macro-F1 tables. The collective `Figure_5_collective_classifiers_horizontal.pdf/png` stacks a–d vertically and has no legends, colorbars, or explanatory footer. Individual panels retain a 0–100 numeric colorbar; all exports use the same original viridis mapping.

The metric gap is 0.12 inches; each metric table is 5.35 inches wide. Numeric cells retain 9-point type on 0.16-inch rows. Model headings use 0.15-inch rows, and the collective panel gap is 0.10 inches. Row and column labels, numeric annotation sizes, and their order are unchanged.

All 588 retained evaluated configurations and their metrics are preserved exactly. 63 mean-pooled configurations are intentionally hidden in both correction conditions; the original source table still contains all 651 results. All 49 original status cells are retained. The merged grid exposes 63 additional unconfigured intersections; these are explicitly N/A, never zero or invented missing results. Original CLS exclusions remain in force. DE means and SDs keep their existing formatting.

Pooling distinction: Full is the unreduced pooled sample vector. For CellFM (both view settings) and BulkFormer B-ALL, it is mean pooled. For BulkFormer TCGA-BRCA, TCGA-BRCA to APOLLO, and CLL-map (corrected and uncorrected), Full is the elementwise sum of max, mean, and median pooling. The removed Mean pooled rows use only mean pooling and are distinct for those BulkFormer datasets. Full tokens remains unchanged: retained contextual gene vectors are projected and flattened into the MLP without gene pooling.

Both the individual panels and the collective are drawn from the original saved table. Heatmap cells are vector rectangles, text uses embedded TrueType fonts, and PNGs are 300 dpi by default. The validation JSON records hashes, exact metric/status comparisons, font checks, and output paths.

Reproduce from any directory:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ls/katzm/.conda/envs/viz_env/bin/python \
  /data01/private/projects/omiclass/manuscript_figures/scripts/plot_figure_5_horizontal.py
```
