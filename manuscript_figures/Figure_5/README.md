# Figure 5 — shared status key and notes

All heatmaps are rendered from the original saved audited table. The `Mean pooled` rows, including their corrected versions, are intentionally hidden in both the older individual layout and the merged-classifier layout. `Full`, its corrected versions, and `Full tokens` remain unchanged. All 588 retained evaluated configurations, both metrics, SDs, and retained N/A cells match the source exactly. The original source table still contains all 651 results; the 63 hidden results were not deleted or replaced.

`Full` denotes the unreduced pooled sample vector. CellFM (both view settings) and BulkFormer B-ALL use mean pooling, so their historical mean-pooled entries can duplicate existing fits. For BulkFormer TCGA-BRCA, TCGA-BRCA to APOLLO, and CLL-map, both corrected and uncorrected `Full` vectors use the **elementwise sum of max, mean, and median pooling**. Their hidden `Mean pooled` rows use only the mean and therefore represent distinct comparisons. `Full tokens` retains contextual gene-token vectors, followed by learned token projection and flattening into the MLP without gene pooling. This display change does not merge, substitute, or average any results.

The merged-classifier panels keep all five classifiers together, with Accuracy and Macro-F1 side by side. The collective stacks a–d and has no legend or colorbar. Individual panels retain the original viridis 0–100 scale and a numeric colorbar. PNGs are at least 300 dpi; PDFs preserve live embedded TrueType text (`pdf.fonttype = 42`).

| Piece | Purpose |
| --- | --- |
| [Figure_5_legend_configuration_status.pdf](Figure_5_legend_configuration_status.pdf) | Shared MISS, FAIL and N/A definitions |
| [Figure_5_shared_notes.pdf](Figure_5_shared_notes.pdf) | Shared seed/SD, correction and evaluation-population notes |

Each standalone piece is exported once and can be reused for all four dataset panels. Existing panel PDF/PNG/SVG paths are retained. Current PDFs preserve live text with embedded TrueType fonts.

All legend pieces have PDF and 400-dpi PNG versions. Figure 5 also has SVG versions. The panels have no inline legend or repeated bottom notes. Original source results and original plotting scripts are unchanged.

Reproduce the older layout with `../scripts/separate_legends_figures_3_5.py --figures 5`, then run `../scripts/plot_figure_5_horizontal.py` for the merged panels and collective. Exact commands are in [../scripts/README.md](../scripts/README.md). Both scripts require the saved local result tables and source drawing helpers; they execute no experiments. Current scientific-data checks, output hashes and font audits are in `legend_separation_manifest.json`, `PDF_FONT_AUDIT.txt`, `Figure_5_classifiers_horizontal.validation.json`, and [../LEGEND_SEPARATION_VALIDATION.json](../LEGEND_SEPARATION_VALIDATION.json).
