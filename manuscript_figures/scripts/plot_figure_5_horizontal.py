#!/usr/bin/env python3
"""Rebuild compact horizontal-classifier Figure 5 panels and collective.

Both the individual and collective figures are rendered from the saved audited
table, never from images or existing PDF pages. Only this script's horizontal
exports are replaced; earlier Figure 5 versions and experimental results remain
untouched. Color limits stay at 0–100 (viridis).
Individual panels have numeric colorbars; the collective has no legends or
colorbars. Every heatmap cell, including the colorbars, remains vector in PDF.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', '/tmp/omiclass-manuscript-matplotlib')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                            'font.family': 'DejaVu Sans', 'font.size': 9,
                            'axes.spines.top': False, 'axes.spines.right': False})
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASETS = [('a', 'BALL', 'B-ALL'), ('b', 'BRCA-TCGA', 'TCGA-BRCA'),
            ('c', 'BRCA-TCGA-APOLLO', 'TCGA-BRCA_to_APOLLO'), ('d', 'CLLmap', 'CLL-map')]
CLASSIFIERS = ['Omiclass', 'Logistic Regression', 'MLP', 'MLP + SMOTE', 'SVM']
KEYS = ['dataset', 'group', 'row_representation', 'correction', 'column']
EXCLUDED_REPRESENTATIONS = ('CLS', 'Mean pooled')
# Dimensions are inches, so narrowing the canvas never shrinks the text.
ROW_HEIGHT = .16
MODEL_HEADER_HEIGHT = .15
HEADER_HEIGHT = .92
BOTTOM_MARGIN = .08
LEFT = 2.00
METRIC_WIDTH = 5.35
METRIC_GAP = .12
TABLE_RIGHT = LEFT + 2 * METRIC_WIDTH + METRIC_GAP
COLORBAR_LEFT = TABLE_RIGHT + .16
INDIVIDUAL_WIDTH = TABLE_RIGHT + .96
COLLECTIVE_WIDTH = TABLE_RIGHT + .08
PANEL_GAP = .10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def row_geometry(rows):
    """Use slimmer model headings without changing cell or label order."""
    heights = np.array([MODEL_HEADER_HEIGHT if rep is None else ROW_HEIGHT
                        for _, rep, _ in rows])
    edges = np.r_[0., np.cumsum(heights)]
    return edges, (edges[:-1] + edges[1:]) / 2


def pure_helpers(reference, compact):
    """Load only the existing grid/row definitions and literal constants."""
    namespace = {'pd': pd}
    tree = ast.parse(reference.read_text())
    constants = {'MODELS', 'REPS', 'CLASSIFIERS', 'STATUS'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_grid'
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reference), 'exec'), namespace)
    tree = ast.parse(compact.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compact_rows']
    row_namespace = {'REFERENCE': namespace}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(compact), 'exec'), row_namespace)
    return namespace, row_namespace['compact_rows']


def prepare_panels(audited, original, definitions, compact_rows):
    if audited.duplicated(KEYS).any() or original.duplicated(KEYS).any():
        raise ValueError('Source table contains duplicate heatmap coordinates')
    panels, displayed = [], []
    for letter, dataset, filename in DATASETS:
        specs, cells = definitions['make_grid'](audited, dataset)
        specs = [spec for spec in specs if spec[1] not in EXCLUDED_REPRESENTATIONS]
        cells = cells[~cells.row_representation.isin(EXCLUDED_REPRESENTATIONS)]
        rows = compact_rows(specs, cells, CLASSIFIERS)
        parts = [cells[(cells.group == group) & (cells.row_representation == rep) & (cells.correction == condition)]
                 for group, rep, condition in rows if rep is not None]
        new = pd.concat(parts, ignore_index=True)
        if new.duplicated(KEYS).any():
            raise ValueError(f'New layout duplicates a configuration: {dataset}')
        indexed = new.set_index(KEYS).sort_index()
        old = original[original.dataset.eq(dataset) &
                       ~original.row_representation.isin(EXCLUDED_REPRESENTATIONS)].set_index(KEYS).sort_index()
        if not old.index.isin(indexed.index).all():
            raise ValueError(f'An existing displayed cell was omitted: {dataset}')
        columns = ['accuracy', 'f1', 'accuracy_sd', 'f1_sd', 'status']
        pd.testing.assert_frame_equal(old[columns], indexed.loc[old.index, columns], check_dtype=False, check_exact=True)
        present = indexed[indexed.status.eq('present')]
        source = audited[audited.dataset.eq(dataset) &
                         ~audited.row_representation.isin(EXCLUDED_REPRESENTATIONS)].set_index(KEYS).sort_index()
        pd.testing.assert_index_equal(present.index, source.index)
        pd.testing.assert_frame_equal(present[columns], source[columns], check_dtype=False, check_exact=True)
        additional = indexed[~indexed.index.isin(old.index)]
        if len(additional) and not additional.status.eq('not_applicable').all():
            raise ValueError('A newly exposed layout cell was treated as an invented or missing result')
        new['added_structural_na_for_merged_layout'] = ~new.set_index(KEYS).index.isin(old.index)
        displayed.append(new)
        panels.append({'letter': letter, 'dataset': dataset, 'filename': filename,
                       'rows': rows, 'cells': new,
                       'height': row_geometry(rows)[0][-1] + HEADER_HEIGHT + BOTTOM_MARGIN,
                       'present_configurations': int(new.status.eq('present').sum()),
                       'original_status_cells': int(old.status.ne('present').sum()),
                       'additional_structural_na_cells': len(additional)})
    return panels, pd.concat(displayed, ignore_index=True)


def draw_metric(ax, panel, metric, definitions, cmap, norm, show_labels):
    rows, cells = panel['rows'], panel['cells']
    edges, centers = row_geometry(rows)
    values = np.full((len(rows), len(CLASSIFIERS)), np.nan)
    annotations, labels = [], []
    lookup = cells.set_index(['group', 'row_representation', 'correction', 'column'])
    for i, (group, rep, condition) in enumerate(rows):
        if rep is None:
            labels.append('DE baseline' if group == 'de' else definitions['MODELS'][group])
            continue
        label = 'DE' if group == 'de' else rep
        if condition.startswith('Corrected'):
            label += ' · corrected'
        labels.append(label)
        for j, classifier in enumerate(CLASSIFIERS):
            entry = lookup.loc[(group, rep, condition, classifier)]
            if entry.status == 'present':
                values[i, j] = entry[metric]
    mesh = ax.pcolormesh(np.arange(6) - .5, edges,
                         np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                         shading='flat', edgecolors='none', antialiased=False, rasterized=False)
    ax.set_xlim(-.5, 4.5); ax.set_ylim(edges[-1], 0)
    for i, (group, rep, condition) in enumerate(rows):
        if rep is None:
            ax.add_patch(Rectangle((-.5, edges[i]), 5, edges[i + 1] - edges[i],
                                   facecolor='white', edgecolor='none'))
            continue
        for j, classifier in enumerate(CLASSIFIERS):
            entry = lookup.loc[(group, rep, condition, classifier)]
            if entry.status == 'present':
                value = float(entry[metric])
                text = f'{value:.1f}'
                if group == 'de':
                    text += f' ±{entry[metric + "_sd"]:.1f}'
                rgba = cmap(norm(value))
                luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
                color, size = ('black' if luminance > .5 else 'white'), 9
            else:
                fill, hatch, text = definitions['STATUS'][entry.status]
                ax.add_patch(Rectangle((j - .5, edges[i]), 1, edges[i + 1] - edges[i], facecolor=fill, hatch=hatch,
                                       edgecolor='#bdbdbd', linewidth=.3))
                color, size = '#777777', 8
            ax.text(j, centers[i], text, ha='center', va='center', fontsize=size, color=color)
            annotations.append(text)
        ax.axhline(edges[i + 1], color='white', linewidth=.4)
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASSIFIERS, fontsize=10)
    ax.xaxis.tick_top(); ax.tick_params(axis='both', length=0, pad=4)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels if show_labels else [''] * len(rows), fontsize=9)
    if show_labels:
        for label, (_, rep, _) in zip(ax.get_yticklabels(), rows):
            if rep is None:
                label.set_fontweight('bold'); label.set_color('#222222')
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Check the vector grid's actual scalar data before export.
    np.testing.assert_allclose(np.ma.asarray(mesh.get_array()).filled(np.nan), values, rtol=0, atol=0, equal_nan=True)
    return {'mesh': mesh, 'values': values, 'annotations': annotations}


def draw_dataset(fig, panel, top_in, definitions, with_colorbar):
    width, height = fig.get_size_inches()
    heat_height = row_geometry(panel['rows'])[0][-1]
    heat_top = top_in + HEADER_HEIGHT
    bottom = height - heat_top - heat_height
    cmap, norm = plt.get_cmap('viridis'), Normalize(0, 100)
    title = panel['dataset'] + ' — foundation-model performance'
    fig.text(.05 / width, 1 - (top_in + .05) / height, panel['letter'], fontsize=16,
             fontweight='bold', ha='left', va='top')
    center = LEFT + METRIC_WIDTH + METRIC_GAP / 2
    fig.text(center / width, 1 - (top_in + .05) / height, title, fontsize=16,
             fontweight='bold', ha='center', va='top')
    fig.text(center / width, 1 - (top_in + .32) / height, 'Classifiers', fontsize=13,
             fontweight='bold', ha='center', va='top')
    axes, checks = [], []
    for index, (metric, heading) in enumerate([('accuracy', 'Accuracy'), ('f1', 'Macro-F1')]):
        x = LEFT + index * (METRIC_WIDTH + METRIC_GAP)
        ax = fig.add_axes([x / width, bottom / height, METRIC_WIDTH / width, heat_height / height])
        ax._figure5_rows = panel['rows']
        check = draw_metric(ax, panel, metric, definitions, cmap, norm, index == 0)
        fig.text((x + METRIC_WIDTH / 2) / width, 1 - (top_in + .53) / height, heading,
                 fontsize=12, fontweight='bold', ha='center', va='top')
        axes.append(ax); checks.append(check)
    if with_colorbar:
        cax = fig.add_axes([COLORBAR_LEFT / width, (bottom + .15 * heat_height) / height,
                            .12 / width, .70 * heat_height / height])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                           label='Performance (%)', ticks=[0, 20, 40, 60, 80, 100])
        if bar.solids is not None:
            bar.solids.set_rasterized(False)
    return axes, checks


def export(fig, stem, dpi, expected_annotations, collective, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        tick_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        if any(a.overlaps(b) for a, b in zip(tick_boxes, tick_boxes[1:])):
            raise ValueError('Classifier names overlap')
        if any((b.x0 - a.x1) * 72 / fig.dpi < 4 for a, b in zip(tick_boxes, tick_boxes[1:])):
            raise ValueError('Classifier names need at least four points of horizontal separation')
        row_boxes = [label.get_window_extent(renderer) for label in ax.get_yticklabels() if label.get_text()]
        if any(a.overlaps(b) for a, b in zip(row_boxes, row_boxes[1:])):
            raise ValueError('Model or representation labels overlap')
        for heading in fig.texts:
            if any(heading.get_window_extent(renderer).overlaps(box) for box in tick_boxes):
                raise ValueError('A heading overlaps a classifier label')
        edges, centers = row_geometry(ax._figure5_rows)
        for annotation in ax.texts:
            j, y = annotation.get_position()
            row = int(np.argmin(abs(centers - y)))
            bbox = annotation.get_window_extent(renderer)
            cell = ax.transData.transform([(j - .5, edges[row]), (j + .5, edges[row + 1])])
            if (bbox.x0 < cell[:, 0].min() or bbox.x1 > cell[:, 0].max()
                    or bbox.y0 < cell[:, 1].min() or bbox.y1 > cell[:, 1].max()):
                raise ValueError('A cell annotation no longer fits within its cell')
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend(list(ax.texts) + ax.get_xticklabels() + ax.get_yticklabels()
                     + [ax.xaxis.label, ax.yaxis.label])
    for label in texts:
        if not label.get_visible() or not label.get_text():
            continue
        bbox = label.get_window_extent(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1 + 1 or bbox.y1 > fig.bbox.y1 + 1:
            raise ValueError(f'Text outside the page: {label.get_text()!r}')
    assert not fig.legends and all(ax.get_legend() is None for ax in fig.axes)
    if collective:
        assert len(fig.axes) == 8, 'Collective must have eight metric axes and no colorbars'
    pdf_path, png_path = stem.with_suffix('.pdf'), stem.with_suffix('.png')
    with PdfPages(pdf_path, metadata={'Title': 'Figure 5 — Foundation-model performance across pipelines',
                                      'Subject': 'Five classifiers side by side; Accuracy and Macro-F1; original viridis 0–100 scale',
                                      'Creator': str(Path(__file__).resolve())}) as pdf:
        pdf.savefig(fig)
    fig.savefig(png_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    font_table = subprocess.check_output(['pdffonts', str(pdf_path)], text=True)
    rows = font_table.strip().splitlines()[2:]
    if not rows or not all('CID TrueType' in row and re.search(r'\s+yes\s+(?:yes|no)\s+yes\s+\d+\s+\d+\s*$', row) for row in rows):
        raise ValueError(f'PDF does not use embedded TrueType text:\n{font_table}')
    with fitz.open(pdf_path) as doc:
        assert len(doc) == 1 and not doc[0].get_images(), 'PDF must remain vector'
        text = doc[0].get_text()
        assert 'Main classifiers' not in text and 'Additional classifiers' not in text
        assert 'Mean pooled' not in text and 'mean pooled' not in text
        assert text.count('Classifiers') == (4 if collective else 1)
        assert 'MISS: missing' not in text and 'DE baseline: five seeds' not in text
        if collective:
            assert 'Performance (%)' not in text
        # Test exact cell annotations in reading order, using normalized spaces.
        spans = [span for block in doc[0].get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']]
        for span in spans:
            x0, y0, x1, y1 = span['bbox']
            if x0 < -.1 or y0 < -.1 or x1 > doc[0].rect.width + .1 or y1 > doc[0].rect.height + .1:
                raise ValueError(f'Exported PDF text outside page: {span["text"]!r}')
        actual = Counter(re.sub(r'\s+', ' ', s['text']).strip() for s in spans)
        for label, count in Counter(expected_annotations).items():
            if actual[label] < count:
                raise ValueError(f'An exported cell annotation is missing: {label!r} ({actual[label]} vs {count})')
    with Image.open(png_path) as png:
        pixels, resolution = list(png.size), list(png.info['dpi'])
        assert min(resolution) >= 299.9
    return {'pdf': {'path': str(pdf_path), 'sha256': digest(pdf_path)},
            'png': {'path': str(png_path), 'sha256': digest(png_path), 'pixels': pixels, 'dpi': resolution},
            'font_audit': font_table, 'pdf_raster_images': 0, 'categorical_legends': 0,
            'numeric_colorbars': 0 if collective else 1, 'all_cell_annotations_verified_in_pdf': True,
            'text_within_page_and_cells': True, 'row_and_column_labels_do_not_overlap': True,
            'page_inches': list(fig.get_size_inches())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error('--dpi must be at least 300')
    root = args.project_root.resolve()
    directory = root / 'manuscript_figures/Figure_5'
    base = root / 'final_results_runs/plots/foundation_comparison_extended'
    output_stems = [directory / f'Figure_5{letter}_{name}_classifiers_horizontal' for letter, _, name in DATASETS]
    collective_stem = directory / 'Figure_5_collective_classifiers_horizontal'
    report_path = directory / 'Figure_5_classifiers_horizontal.validation.json'
    cells_path = directory / 'Figure_5_classifiers_horizontal_displayed_cells.csv'
    notes_path = directory / 'Figure_5_classifiers_horizontal.README.md'
    owned = {p.with_suffix(ext) for p in output_stems + [collective_stem] for ext in ('.pdf', '.png')}
    owned.update([report_path, cells_path, notes_path])
    existing = {str(p): digest(p) for p in directory.rglob('*') if p.is_file() and p not in owned}
    sources = [base / 'heatmaps_compact_svm/audited_metrics.csv',
               base / 'heatmaps_compact_svm/displayed_configuration_status.csv',
               base / 'performance_heatmaps.py', base / 'performance_heatmaps_compact.py']
    source_hashes = {str(path): digest(path) for path in sources}
    audited, original = pd.read_csv(sources[0]), pd.read_csv(sources[1])
    definitions, compact_rows = pure_helpers(sources[2], sources[3])
    assert definitions['CLASSIFIERS'] == CLASSIFIERS
    panels, displayed = prepare_panels(audited, original, definitions, compact_rows)
    excluded = audited[audited.row_representation.isin(EXCLUDED_REPRESENTATIONS)]
    records, all_annotations = [], []
    for panel, stem in zip(panels, output_stems):
        print(f'Rebuilding {panel["letter"]}: {panel["dataset"]}, one five-classifier section...', flush=True)
        fig = plt.figure(figsize=(INDIVIDUAL_WIDTH, panel['height']))
        axes, checks = draw_dataset(fig, panel, 0, definitions, True)
        annotations = [text for check in checks for text in check['annotations']]
        all_annotations.extend(annotations)
        record = export(fig, stem, args.dpi, annotations, False, axes)
        record.update(panel=panel['letter'], dataset=panel['dataset'], rows_including_model_headings=len(panel['rows']),
                      present_configurations=panel['present_configurations'], original_status_cells=panel['original_status_cells'],
                      additional_structural_na_cells=panel['additional_structural_na_cells'])
        records.append(record)
    print('Rebuilding legend-free collective directly from the same data...', flush=True)
    gap = PANEL_GAP
    height = sum(panel['height'] for panel in panels) + gap * 3
    fig = plt.figure(figsize=(COLLECTIVE_WIDTH, height))
    top, all_axes = 0., []
    for panel in panels:
        axes, _ = draw_dataset(fig, panel, top, definitions, False)
        all_axes.extend(axes); top += panel['height'] + gap
    collective = export(fig, collective_stem, args.dpi, all_annotations, True, all_axes)
    displayed.to_csv(cells_path, index=False)
    for path, checksum in existing.items():
        assert digest(path) == checksum, f'Existing Figure 5 file changed: {path}'
    for path, checksum in source_hashes.items():
        assert digest(path) == checksum, f'Original source changed: {path}'
    report = {'generated_at': datetime.now(ZoneInfo('Asia/Jerusalem')).isoformat(timespec='seconds'),
              'script': str(Path(__file__).resolve()), 'script_sha256': digest(__file__),
              'classifier_order': CLASSIFIERS, 'metric_order': ['Accuracy', 'Macro-F1'],
              'individual_panels': records, 'collective': collective,
              'all_retained_values_sd_labels_and_statuses_preserved': True,
              'preservation_scope': 'All retained rows; Mean pooled and corrected Mean pooled intentionally hidden, not deleted from source results',
              'excluded_representations': EXCLUDED_REPRESENTATIONS,
              'intentionally_hidden_evaluated_configurations': len(excluded),
              'source_evaluated_configurations': len(audited),
              'all_original_status_cells_preserved': int(original.status.ne('present').sum()),
              'present_configurations': int(displayed.status.eq('present').sum()),
              'added_structural_na_cells': int(displayed.added_structural_na_for_merged_layout.sum()),
              'color_mapping': {'cmap': 'viridis', 'vmin': 0, 'vmax': 100},
              'layout_inches': {'metric_width': METRIC_WIDTH, 'metric_gap': METRIC_GAP,
                                'cell_row_height': ROW_HEIGHT, 'model_heading_height': MODEL_HEADER_HEIGHT,
                                'header_height': HEADER_HEIGHT, 'left_margin': LEFT,
                                'bottom_margin': BOTTOM_MARGIN, 'panel_gap': PANEL_GAP,
                                'individual_width': INDIVIDUAL_WIDTH, 'collective_width': COLLECTIVE_WIDTH},
              'underlying_source_table': str(sources[0]), 'derived_displayed_cells': str(cells_path),
              'existing_figure_5_files_unchanged': existing, 'source_sha256': source_hashes,
              'rebuild_method': 'Both individual and collective plots drawn directly from numerical tables; no image/PDF stitching; vector heatmap cells and editable TrueType text'}
    report_path.write_text(json.dumps(plain(report), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    notes_path.write_text('# Figure 5 — horizontal classifiers\n\n'
        'Compact layout revision: this script replaces its own horizontal exports only. '
        'The earlier panels, legends, notes, and manifests are unchanged.\n\n'
        'Each `Figure_5[a-d]_*_classifiers_horizontal.pdf/png` has one Classifiers heading, '
        'with Omiclass, Logistic Regression, MLP, MLP + SMOTE, and SVM side by side in both '
        'the Accuracy and Macro-F1 tables. The collective `Figure_5_collective_classifiers_horizontal.pdf/png` '
        'stacks a–d vertically and has no legends, colorbars, or explanatory footer. '
        'Individual panels retain a 0–100 numeric colorbar; all exports use the same original viridis mapping.\n\n'
        f'The metric gap is {METRIC_GAP:.2f} inches; each metric table is {METRIC_WIDTH:.2f} inches wide. '
        f'Numeric cells retain 9-point type on {ROW_HEIGHT:.2f}-inch rows. '
        f'Model headings use {MODEL_HEADER_HEIGHT:.2f}-inch rows, and the collective panel gap is {PANEL_GAP:.2f} inches. '
        'Row and column labels, numeric annotation sizes, and their order are unchanged.\n\n'
        f'All {int(displayed.status.eq("present").sum())} retained evaluated configurations and their metrics are preserved exactly. '
        f'{len(excluded)} mean-pooled configurations are intentionally hidden in both correction conditions; '
        f'the original source table still contains all {len(audited)} results. '
        f'All {int(original.status.ne("present").sum())} original status cells are retained. '
        f'The merged grid exposes {int(displayed.added_structural_na_for_merged_layout.sum())} additional unconfigured intersections; '
        'these are explicitly N/A, never zero or invented missing results. Original CLS exclusions remain in force. '
        'DE means and SDs keep their existing formatting.\n\n'
        'Pooling distinction: Full is the unreduced pooled sample vector. For CellFM (both view settings) '
        'and BulkFormer B-ALL, it is mean pooled. For BulkFormer TCGA-BRCA, TCGA-BRCA to APOLLO, and CLL-map '
        '(corrected and uncorrected), Full is the elementwise sum of max, mean, and median pooling. '
        'The removed Mean pooled rows use only mean pooling and are distinct for those BulkFormer datasets. '
        'Full tokens remains unchanged: retained contextual gene vectors are projected and flattened into the MLP without gene pooling.\n\n'
        'Both the individual panels and the collective are drawn from the original saved table. '
        'Heatmap cells are vector rectangles, text uses embedded TrueType fonts, and PNGs are 300 dpi by default. '
        'The validation JSON records hashes, exact metric/status comparisons, font checks, and output paths.\n\n'
        'Reproduce from any directory:\n\n```bash\n'
        'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\\n'
        '  /home/ls/katzm/.conda/envs/viz_env/bin/python \\\n'
        f'  {Path(__file__).resolve()}\n```\n')
    print(f'Created four updated panels and {collective_stem}.pdf/png.', flush=True)
    print(f'Verified {int(displayed.status.eq("present").sum())} retained results; {len(excluded)} configurations hidden only at display time; '
          f'{int(original.status.ne("present").sum())} original status cells, '
          f'and {len(existing)} unchanged existing Figure 5 files.', flush=True)


if __name__ == '__main__':
    main()
