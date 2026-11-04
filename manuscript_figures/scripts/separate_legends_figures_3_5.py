#!/usr/bin/env python3
"""Re-export Figures 3–5 with shared standalone legends, from saved data only.

Only manuscript_figures is written. No original exporter is executed, no model
is loaded, and no representation, UMAP or numerical result is recomputed.
Existing original source plots remain available outside manuscript_figures.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', '/tmp/omiclass-manuscript-matplotlib')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                            'font.family': 'DejaVu Sans'})
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
S3 = Path('final_results_runs/plots/non_foundation_framework_comparison')
S4 = Path('thesis_plots/final_article_figures/expanded_classification_3rows_4cols_clean/with_bulkrnabert')
S5 = Path('final_results_runs/plots/foundation_comparison_extended')
METHODS = ['multiclassPairs', 'ranktreeEnsemble', 'MINT', 'CPOP']
COLORS = {'multiclassPairs': '#777777', 'ranktreeEnsemble': '#4C78A8', 'MINT': '#59A14F', 'CPOP': '#B279A2'}
TASKS3 = [
    ('a', 'B-ALL', 'B-ALL_classification'),
    ('b', 'BRCA-TCGA → APOLLO', 'TCGA-BRCA_to_APOLLO_transfer'),
    ('c', 'BRCA proteomics', 'BRCA_proteomics'),
    ('d', 'CLL', 'CLL-map_classification'),
    ('e', 'CLL → DKFZ', 'CLL-map_to_DKFZ_transfer'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def extract(source, functions, namespace, constants=()):
    """Load only named pure plotting helpers, not a script's entrypoint/imports."""
    tree = ast.parse(Path(source).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions
             or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if set(functions) != found:
        raise ValueError(f'Missing source plotting helpers: {set(functions) - found}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class Exporter:
    def __init__(self, root, dpi):
        self.root, self.dpi = root, dpi
        self.destination = root / 'manuscript_figures'
        self.files = []
        self.inputs = {}
        self.checks = []

    def input(self, path):
        path = self.root / path if not Path(path).is_absolute() else Path(path)
        checksum = digest(path)
        if str(path) in self.inputs and self.inputs[str(path)] != checksum:
            raise ValueError(f'Source changed while reading: {path}')
        self.inputs[str(path)] = checksum
        return path

    def save(self, fig, relative, role='panel', legends=(), tight=True):
        stem = self.destination / relative
        stem.parent.mkdir(parents=True, exist_ok=True)
        options = {'bbox_inches': 'tight', 'pad_inches': .05} if tight else {}
        for extension in ('pdf', 'png'):
            path = stem.with_suffix('.' + extension)
            if extension == 'pdf':
                with PdfPages(path) as pdf:
                    pdf.savefig(fig, **options)
            else:
                fig.savefig(path, dpi=self.dpi, facecolor='white', **options)
            self.record(path, role, legends)
        if relative.startswith('Figure_5/'):
            path = stem.with_suffix('.svg')
            fig.savefig(path, **options)
            self.record(path, role, legends)

    def record(self, path, role, legends=()):
        self.files.append({'path': str(path), 'role': role, 'sha256': digest(path),
                           'standalone_legends_and_notes': list(legends)})

    def legend(self, number, name, handles, labels=None, ncol=None):
        labels = labels or [p.get_label() for p in handles]
        fig = plt.figure(figsize=(max(3, min(len(handles), ncol or len(handles))) * 1.35, .8))
        legend = fig.legend(handles, labels, loc='center', ncol=ncol or len(handles), frameon=False,
                            fontsize=10, handlelength=1.8, columnspacing=1.5)
        fig.canvas.draw()
        bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025, 1.25)
        relative = f'Figure_{number}/Figure_{number}_legend_{name}'
        stem = self.destination / relative
        for extension in ('pdf', 'png', 'svg') if number == 5 else ('pdf', 'png'):
            path = stem.with_suffix('.' + extension)
            if extension == 'pdf':
                with PdfPages(path) as pdf:
                    pdf.savefig(fig, bbox_inches=bbox, pad_inches=.03)
            else:
                fig.savefig(path, dpi=self.dpi, bbox_inches=bbox, pad_inches=.03, facecolor='white')
            self.record(path, 'standalone_legend')
        plt.close(fig)
        return relative

    def notes(self, number, name, text, width=10):
        fig = plt.figure(figsize=(width, .25 * (text.count('\n') + 1) + .12))
        fig.text(.5, .5, text, ha='center', va='center', fontsize=9)
        relative = f'Figure_{number}/Figure_{number}_{name}'
        self.save(fig, relative, 'standalone_notes')
        plt.close(fig)
        return relative


def figure3(exporter):
    base = exporter.root / S3
    metrics = pd.read_csv(exporter.input(base / 'figures_grouped/metrics.csv'))
    coverage = pd.read_csv(exporter.input(base / 'figures_grouped/coverage.csv'))
    original_metrics = pd.read_csv(exporter.input(base / 'figures/metrics.csv'))
    original_coverage = pd.read_csv(exporter.input(base / 'figures/coverage.csv'))
    exporter.input(base / 'grouped_comparison.py')
    exporter.input(base / 'comparison.py')
    metric_key = exporter.legend(3, 'paired_metrics', [
        Patch(facecolor='#DDDDDD', edgecolor='black', label='Accuracy'),
        Patch(facecolor='#DDDDDD', edgecolor='black', hatch='//', label='Macro F1')])
    color_key = exporter.legend(3, 'omiclass_colors', [
        Patch(facecolor='#E02020', edgecolor='black', label='Omiclass original/corrected'),
        Patch(facecolor='#F4A582', edgecolor='black', label='Omiclass uncorrected')])
    native_key = exporter.legend(3, 'separate_metrics_hatching', [
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Native comparator (separate-metric alternatives only)')])
    notes = exporter.notes(3, 'shared_notes',
        'Seeds 11, 73, 89, 61, 80; mean ± sample SD.\nMatched evaluation IDs/labels within each seed; MINT excluded.')
    for letter, task, dataset in TASKS3:
        complete = coverage.loc[coverage.status.eq('Complete'), ['task', 'method']]
        data = metrics.merge(complete, on=['task', 'method'], how='inner', validate='one_to_one')
        data = data[data.task.eq(task)]
        order = data.loc[data.framework.eq('Omiclass'), 'method'].tolist() + [m for m in METHODS if m in set(data.method)]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        values = []
        for x, method in enumerate(order):
            row = data[data.method.eq(method)].iloc[0]
            color = ('#F4A582' if row.condition == 'Uncorrected' else '#E02020') if row.framework == 'Omiclass' else COLORS[row.framework]
            for offset, metric, hatch in [(-.18, 'accuracy', ''), (.18, 'f1', '//')]:
                value, sd = row[metric], row[metric + '_sd']
                ax.bar(x + offset, value, yerr=sd, capsize=2.5, width=.34, color=color,
                       edgecolor='black', linewidth=.6, hatch=hatch, error_kw={'elinewidth': 1})
                ax.text(x + offset, value + sd + 1.1, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
                values.append(float(value))
        upper = max(float((data[m] + data[m + '_sd']).max()) for m in ('accuracy', 'f1'))
        ax.set_ylim(0, max(105, upper + 5)); ax.set_xlim(-.6, len(order) - .4)
        ax.set_yticks(np.arange(0, 101, 20)); ax.set_ylabel('Score (%)', fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([m.replace(' — ', '\n').replace('Corrected (limma)', 'Corrected\n(limma)') for m in order], fontsize=8.5)
        ax.tick_params(axis='x', length=0, pad=6); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(f'{task} — Accuracy and Macro F1', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        assert np.allclose([p.get_height() for p in ax.patches], values, rtol=0, atol=0)
        exporter.save(fig, f'Figure_3/Figure_3{letter}_{dataset}_accuracy_f1', legends=[metric_key, color_key, notes])
        exporter.checks.append({'figure': f'3{letter}', 'bars_and_saved_means_match': True, 'n_bars': len(values), 'no_inline_legend_or_notes': True})
        plt.close(fig)
        # Preserve the separate-metric layouts, including their historical N/A slots.
        data = original_metrics[original_metrics.task.eq(task)]
        order = data.loc[data.framework.eq('Omiclass'), 'method'].tolist() + METHODS
        for metric, label in [('accuracy', 'Accuracy'), ('f1', 'Macro F1')]:
            fig, ax = plt.subplots(figsize=(14, 5))
            for x, method in enumerate(order):
                selected = data[data.method.eq(method)]
                if selected.empty:
                    state = original_coverage[original_coverage.task.eq(task) & original_coverage.method.eq(method)].iloc[0]
                    text = 'N/A' if state.status == 'Not applicable' else f'MINT deferred\n{int(state.n_seeds)}/5 complete'
                    ax.text(x, 3, text, ha='center', va='bottom', fontsize=9, color='#666666')
                else:
                    row = selected.iloc[0]
                    color = ('#F4A582' if row.condition == 'Uncorrected' else '#E02020') if row.framework == 'Omiclass' else COLORS[row.framework]
                    value, sd = row[metric], row[metric + '_sd']
                    ax.bar(x, value, yerr=sd, capsize=3, width=.7, color=color, edgecolor='black', linewidth=.5,
                           hatch='//' if row.framework != 'Omiclass' else '')
                    ax.text(x, value + sd + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=14)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([m.replace(' — ', '\n') for m in order], rotation=40, ha='right', fontsize=10)
            ax.set_ylim(0, max(105, float((data[metric] + data[metric + '_sd']).max()) + 4))
            ax.set_ylabel(label + ' (%)'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(f'{task} — {label} — five-seed mean ± SD', pad=24, fontweight='bold', fontsize=18)
            fig.tight_layout()
            exporter.save(fig, f'Figure_3/alternatives_separate_metrics/Figure_3{letter}_{dataset}_{metric}', legends=[color_key, native_key, notes])
            plt.close(fig)


def font_tools():
    regular_path = findfont(FontProperties(family='DejaVu Sans'))
    bold_path = findfont(FontProperties(family='DejaVu Sans', weight='bold'))
    return (regular_path, fitz.Font(fontfile=regular_path)), (bold_path, fitz.Font(fontfile=bold_path))


def insert_text(page, point, text, size, font, bold=False, color=(0, 0, 0)):
    path, _ = font
    name = 'ManuscriptDejaVuBold' if bold else 'ManuscriptDejaVu'
    page.insert_font(fontname=name, fontfile=path)
    page.insert_text(point, text, fontsize=size, fontname=name, color=color, overlay=True)


def kp_box(page, dest, text, bold):
    width = bold[1].text_length('KP: 0.000', fontsize=16) + 10
    box = fitz.Rect(dest.x1 - width - 4, dest.y0 + 4, dest.x1 - 4, dest.y0 + 29)
    page.draw_rect(box, fill=(1, 1, 1), color=None, fill_opacity=.8, overlay=True)
    insert_text(page, (box.x0 + 5, box.y0 + 18), text, 16, bold, bold=True, color=(.20, .25, .33))


def insert_frozen(page, graphics, frames, spans, index, dest, regular, bold):
    # Same saved geometry/ticks as the original helper, with embedded TrueType.
    frame = frames[index]
    page.show_pdf_page(dest, graphics, 0, clip=frame, keep_proportion=False)
    oldrow, oldcol = divmod(index, 2)
    xsplit = (frames[0].x1 + frames[1].x0) / 2
    scope = fitz.Rect(0 if oldcol == 0 else xsplit,
                     21 if oldrow == 0 else (frames[2 * (oldrow - 1)].y1 + frames[2 * oldrow].y0) / 2,
                     xsplit if oldcol == 0 else max(f.x1 for f in frames) + 6,
                     736 if oldrow == 2 else (frames[2 * oldrow].y1 + frames[2 * (oldrow + 1)].y0) / 2)
    purity = []
    for span in spans:
        rect = fitz.Rect(span['bbox']); center = (rect.tl + rect.br) / 2
        text = span['text']
        if not scope.contains(center):
            continue
        if text.startswith('kNN purity:'):
            purity.append(text)
        if not re.fullmatch(r'[−+\-\d.eE]+', text):
            continue
        text = text.replace('−', '-')
        width = regular[1].text_length(text, fontsize=8)
        if center.y > frame.y1 and frame.x0 - 3 < center.x < frame.x1 + 3:
            x = dest.x0 + (center.x - frame.x0) / frame.width * dest.width
            insert_text(page, (x - width / 2, dest.y1 + 11), text, 8, regular)
        elif center.x < frame.x0 and frame.y0 - 3 < center.y < frame.y1 + 3:
            y = dest.y0 + (center.y - frame.y0) / frame.height * dest.height
            insert_text(page, (dest.x0 - width - 4, y + 3), text, 8, regular)
    assert len(purity) == 1
    text = 'KP: ' + purity[0].split(':', 1)[1].strip()
    kp_box(page, dest, text, bold)
    return text


def figure4(exporter):
    base = exporter.root / S4
    manifest = json.loads(exporter.input(base / 'manifest.json').read_text())
    source_helper = exporter.input(base.parent / 'create_layout.py')
    viz = exporter.input(exporter.root / 'final_results_runs/plots/viz_data.py')
    # Import only the label mapping and PDF vector extraction helpers.
    names = extract(viz, ['display_class_label'], {'Any': object}, ['CLASS_DISPLAY_NAMES'])
    namespace = extract(source_helper, ['pdf_spans', 'pdf_frames', 'source_parts'],
                        {'fitz': fitz, 're': re, 'display_class_label': names['display_class_label']})
    regular, bold = font_tools()
    kp_note = exporter.notes(4, 'shared_KP_definition', 'KP = k-Nearest Neighbor Purity', width=4)
    combined = fitz.open()
    for letter, layout, stem in zip('abc', manifest['layouts'], ('B_ALL', 'BRCA', 'CLL')):
        task = layout['task']
        frozen_source = Path(next(item['source'] for item in layout['panels'] if item['kind'] == 'frozen_original'))
        exporter.input(frozen_source)
        source, graphics, frames, spans, colors = namespace['source_parts'](frozen_source)
        color_key = exporter.legend(4, {'a': 'B-ALL_classes', 'b': 'BRCA_classes', 'c': 'CLL_classes'}[letter],
                                    [Patch(facecolor=color, label=label) for label, color in colors.items()],
                                    ncol=6 if letter == 'a' else len(colors))
        panel_w, panel_h = manifest['panel_size_pt']
        left, right, top, bottom, gap_x, gap_y = 32., 10., 45., 20., 30., 32.
        width = left + 5 * panel_w + 4 * gap_x + right
        height = top + 3 * panel_h + 2 * gap_y + bottom
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                             'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8}):
            fig, axes = plt.subplots(3, 5, figsize=(width / 72, height / 72))
            axes = axes.ravel()
            fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height,
                                bottom=bottom / height, wspace=gap_x / panel_w, hspace=gap_y / panel_h)
            expected_kp = []
            for item in layout['panels']:
                slot = item['slot']; ax = axes[slot]
                if item['kind'] == 'frozen_original':
                    ax.axis('off')
                    expected_kp.append(item['purity'])
                else:
                    rep = next(r for r in manifest['representations'] if r['task'] == task and r['slot'] == slot)
                    path = exporter.input(base / f'{stem}_{rep["backend"]}_{rep["representation"]}_umap.csv')
                    coords = pd.read_csv(path, dtype={'sample_id': str})
                    assert len(coords) == rep['n_samples'] and not coords.sample_id.duplicated().any()
                    assert set(coords.label) <= set(colors)
                    for label, color in colors.items():
                        part = coords[coords.label.eq(label)]
                        ax.scatter(part.UMAP1, part.UMAP2, s=5, color=color, alpha=.82,
                                   edgecolors='#555555', linewidths=.15)
                    assert sum(len(collection.get_offsets()) for collection in ax.collections) == len(coords)
                    expected_kp.append(f'KP: {rep["knn_purity"]:.3f}')
                ax.set_title(item['title'])
            fig.suptitle(task.replace('BRCA-TCGA-APOLO', 'BRCA TCGA / APOLLO') + ' — representation comparison', fontsize=15, y=.984)
            fig.canvas.draw(); renderer = fig.canvas.get_renderer()
            for ax in axes:
                while ax.title.get_window_extent(renderer).width > ax.get_window_extent(renderer).width and ax.title.get_fontsize() > 6:
                    ax.title.set_fontsize(ax.title.get_fontsize() - .25)
            stream = io.BytesIO()
            with PdfPages(stream) as pdf:
                pdf.savefig(fig)
            document = fitz.open(stream=stream.getvalue(), filetype='pdf'); page = document[0]
            actual_kp = []
            for item in layout['panels']:
                slot = item['slot']; pos = axes[slot].get_position()
                dest = fitz.Rect(pos.x0 * page.rect.width, (1 - pos.y1) * page.rect.height,
                                 pos.x1 * page.rect.width, (1 - pos.y0) * page.rect.height)
                assert abs(dest.width - panel_w) < .01 and abs(dest.height - panel_h) < .01
                if item['kind'] == 'frozen_original':
                    actual_kp.append(insert_frozen(page, graphics, frames, spans, item['source_index'], dest, regular, bold))
                else:
                    rep = next(r for r in manifest['representations'] if r['task'] == task and r['slot'] == slot)
                    text = f'KP: {rep["knn_purity"]:.3f}'
                    kp_box(page, dest, text, bold); actual_kp.append(text)
            assert Counter(actual_kp) == Counter(expected_kp)
            text = page.get_text()
            assert text.count('KP:') == 15 and 'KP = k-Nearest' not in text
            dataset = {'a': 'B-ALL', 'b': 'TCGA-BRCA_and_APOLLO', 'c': 'CLL-map'}[letter]
            output = exporter.destination / f'Figure_4/Figure_4{letter}_{dataset}_class_structure.pdf'
            document.subset_fonts()
            document.save(output, garbage=4, deflate=True)
            exporter.record(output, 'panel', [color_key, kp_note])
            with fitz.open(output) as saved:
                png = output.with_suffix('.png')
                saved[0].get_pixmap(dpi=exporter.dpi, alpha=False).save(png)
                exporter.record(png, 'panel', [color_key, kp_note])
                combined.insert_pdf(saved)
            document.close(); plt.close(fig); graphics.close(); source.close()
        exporter.checks.append({'figure': f'4{letter}', 'saved_coordinates_reused': True, 'original_frozen_vectors_reused': True,
                                'panel_physical_dimensions_preserved': True, 'n_panels': 15,
                                'kp_annotations_identical': True, 'no_inline_legend_or_notes': True})
    output = exporter.destination / 'Figure_4/Figure_4_abc_all_datasets_existing_3page_collection.pdf'
    combined.save(output, garbage=4, deflate=True); combined.close()
    exporter.record(output, 'existing_collection_reexport', [kp_note])


def figure5(exporter):
    base = exporter.root / S5
    data = pd.read_csv(exporter.input(base / 'heatmaps_compact_svm/audited_metrics.csv'))
    original_status = pd.read_csv(exporter.input(base / 'heatmaps_compact_svm/displayed_configuration_status.csv'))
    excluded_representations = ('CLS', 'Mean pooled')
    # Display policy only: keep every source table and fitted experiment intact.
    retained_status = original_status[~original_status.row_representation.isin(excluded_representations)]
    namespace = extract(exporter.input(base / 'performance_heatmaps.py'), ['make_grid'],
                        {'pd': pd}, ['MODELS', 'REPS', 'CLASSIFIERS', 'STATUS'])
    helper = extract(exporter.input(base / 'performance_heatmaps_compact.py'), ['compact_rows', 'plot_panel'],
                     {'np': np, 'Rectangle': Rectangle, 'REFERENCE': namespace})
    status_key = exporter.legend(5, 'configuration_status', [
        Patch(facecolor=value[0], edgecolor='#bdbdbd', hatch=value[1],
              label=value[2] + ': ' + name.replace('_', ' ')) for name, value in namespace['STATUS'].items()])
    notes = exporter.notes(5, 'shared_notes',
        'DE baseline: five seeds, mean ± SD. Foundation results: seed 80.\n'
        'Corrected = limma before encoding. Test populations may differ; consult the split audit.\n'
        'N/A is unconfigured, not zero. Full: unreduced pooled sample vector. Full tokens: unpooled gene tokens.')
    groups = {'Main classifiers': ['Omiclass', 'Logistic Regression'], 'Additional classifiers': ['MLP', 'MLP + SMOTE', 'SVM']}
    for letter, dataset, name in [('a', 'BALL', 'B-ALL'), ('b', 'BRCA-TCGA', 'TCGA-BRCA'),
                                  ('c', 'BRCA-TCGA-APOLLO', 'TCGA-BRCA_to_APOLLO'), ('d', 'CLLmap', 'CLL-map')]:
        specs, cells = namespace['make_grid'](data, dataset)
        specs = [spec for spec in specs if spec[1] not in excluded_representations]
        cells = cells[~cells.row_representation.isin(excluded_representations)]
        sets = {title: helper['compact_rows'](specs, cells, columns) for title, columns in groups.items()}
        heights = [len(rows) for rows in sets.values()]
        old_height = sum(heights) * .225 + 2.8
        removed_height = .8
        height = old_height - removed_height
        fig, axes = plt.subplots(2, 2, figsize=(12.5, height), gridspec_kw={'height_ratios': heights})
        cmap, norm = plt.get_cmap('viridis'), Normalize(0, 100)
        shown = []
        for row, (title, classifiers) in enumerate(groups.items()):
            rows = sets[title]
            for column, (metric, label) in enumerate([('accuracy', 'Accuracy'), ('f1', 'Macro-F1')]):
                ax = axes[row, column]
                helper['plot_panel'](ax, rows, cells, classifiers, metric, cmap, norm, column == 0)
                ax.set_title(title + ' — ' + label, fontsize=12, fontweight='bold', pad=32)
            for group, rep, condition in rows:
                if rep is not None:
                    selected = cells[(cells.group == group) & (cells.row_representation == rep) & (cells.correction == condition) & cells.column.isin(classifiers)]
                    shown.append(selected)
        displayed = pd.concat(shown, ignore_index=True)
        keys = ['dataset', 'group', 'row_representation', 'correction', 'column']
        current = displayed.set_index(keys).sort_index()
        original = retained_status[retained_status.dataset.eq(dataset)].set_index(keys).sort_index()
        columns = ['accuracy', 'f1', 'accuracy_sd', 'f1_sd', 'status']
        pd.testing.assert_frame_equal(current[columns], original[columns], check_dtype=False, check_exact=True)
        fig.suptitle(dataset + ' — foundation-model performance', fontsize=16, fontweight='bold', y=.985)
        # Keep the original plot area's physical height; remove only footer space.
        old_top_in = .09 * old_height
        old_bottom_in = .095 * old_height
        fig.subplots_adjust(left=.235, right=.91, top=1 - old_top_in / height,
                            bottom=(old_bottom_in - removed_height) / height, wspace=.12, hspace=.30)
        cax = fig.add_axes([.935, .22, .014, .54])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                     label='Performance (%)', ticks=[0, 20, 40, 60, 80, 100])
        exporter.save(fig, f'Figure_5/Figure_5{letter}_{name}_foundation_performance', legends=[status_key, notes])
        exporter.checks.append({'figure': f'5{letter}', 'all_retained_coordinates_status_and_both_metrics_exactly_match_original': True,
                                'excluded_representations': list(excluded_representations),
                                'n_displayed_results': int(displayed.status.eq('present').sum()),
                                'no_inline_legend_or_notes': True, 'side_colorbar_retained': True})
        plt.close(fig)


def validate(exporter):
    import subprocess
    audit = []
    for entry in exporter.files:
        path = Path(entry['path'])
        if path.suffix == '.pdf':
            fonts = subprocess.check_output(['pdffonts', str(path)], text=True)
            rows = fonts.strip().splitlines()[2:]
            # Fully embedded fonts are as valid as subsets; require embedding
            # and Unicode mapping, without requiring the `sub` column to be yes.
            assert rows and all('CID TrueType' in row and re.search(r'\s+yes\s+(?:yes|no)\s+yes\s+\d+\s+\d+\s*$', row) for row in rows), (path, fonts)
            with fitz.open(path) as document:
                text = '\n'.join(page.get_text() for page in document)
                assert text.strip()
                if entry['role'] in ('panel', 'existing_collection_reexport'):
                    assert 'Seeds 11, 73' not in text and 'KP = k-Nearest' not in text
                    assert 'DE baseline: five seeds' not in text and 'MISS: missing' not in text
                    if path.parent.name == 'Figure_5':
                        assert 'Mean pooled' not in text and 'mean pooled' not in text
            entry['pdf_font_compatibility'] = 'PASS: embedded CID TrueType, live text'
            entry['pages'] = len(document) if not document.is_closed else (3 if entry['role'] == 'existing_collection_reexport' else 1)
            audit.append(f"FILE: {path}\n{fonts}")
        elif path.suffix == '.png':
            with Image.open(path) as image:
                assert min(image.info['dpi']) >= 299.9
                entry.update(pixels=list(image.size), dpi=list(image.info['dpi']))
    for path, checksum in exporter.inputs.items():
        assert digest(path) == checksum, f'Original input changed: {path}'
    for number in (3, 4, 5):
        directory = exporter.destination / f'Figure_{number}'
        entries = [entry for entry in exporter.files if Path(entry['path']).is_relative_to(directory)]
        if not entries:
            continue
        json_write(directory / 'legend_separation_manifest.json', {
            'figure': number, 'plotting_script': str(Path(__file__).resolve()), 'files': entries,
            'original_sources_unchanged': True, 'all_pdf_fonts_embedded_truetype': True,
            'retained_scientific_values_and_coordinates_preserved': True,
            'excluded_representations': ['CLS', 'Mean pooled'] if number == 5 else [],
            'notes': 'Legends and repeated footnotes are standalone files. Physical original panel sizes are retained in Figure 4. Heatmap side colorbars remain attached.'})
        (directory / 'PDF_FONT_AUDIT.txt').write_text('\n'.join(part for part in audit if f'/Figure_{number}/' in part))
    validation_path = exporter.destination / 'LEGEND_SEPARATION_VALIDATION.json'
    previous = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    updated_directories = {str(Path(entry['path']).parent) for entry in exporter.files}
    updated_figures = {str(check['figure'])[0] for check in exporter.checks}
    previous_files = [entry for entry in previous.get('files', [])
                      if str(Path(entry['path']).parent) not in updated_directories]
    previous_checks = [check for check in previous.get('checks', [])
                       if str(check.get('figure', ''))[:1] not in updated_figures]
    json_write(validation_path, {
        'generated_at': datetime.now(ZoneInfo('Asia/Jerusalem')).isoformat(timespec='seconds'),
        'plotting_script': str(Path(__file__).resolve()),
        'source_sha256': {**previous.get('source_sha256', {}), **exporter.inputs},
        'files': previous_files + exporter.files, 'checks': previous_checks + exporter.checks,
        'all_original_source_hashes_unchanged': True, 'dpi': exporter.dpi})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--dpi', type=int, default=400)
    parser.add_argument('--figures', type=int, nargs='+', choices=(3, 4, 5), default=[3, 4, 5],
                        help='Select manuscript figures; use --figures 5 to leave Figures 3 and 4 untouched')
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error('--dpi must be at least 300')
    exporter = Exporter(args.project_root.resolve(), args.dpi)
    exporter.input(Path(__file__).resolve())
    if 3 in args.figures:
        print('Re-exporting Figure 3 and its separate-metric alternatives...', flush=True)
        figure3(exporter)
    if 4 in args.figures:
        print('Re-exporting Figure 4 from saved coordinates and frozen vectors...', flush=True)
        figure4(exporter)
    if 5 in args.figures:
        print('Re-exporting Figure 5 from the exact saved heatmap table...', flush=True)
        figure5(exporter)
    print('Validating embedded fonts, standalone legends, and unchanged scientific inputs...', flush=True)
    validate(exporter)
    print(f'Finished: {len(exporter.files)} plot/legend/note exports; originals unchanged.', flush=True)


if __name__ == '__main__':
    main()
