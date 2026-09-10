"""Packaging precision and safety, independent of the full manuscript archive."""
import importlib.util
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

PATH = Path(__file__).resolve().parents[1] / 'final_results_runs/build_supplementary_submission.py'
SPEC = importlib.util.spec_from_file_location('submission_packager', PATH)
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def test_exact_numeric_literals_and_literal_identifiers(tmp_path):
    package = object.__new__(mod.Package)
    package.output, package.catalog = tmp_path, []
    sheet = mod.Sheet('Precision', ['metric', 'P.Value', 'gene', 'seed'], [
        {'metric': '0.8764705882352942', 'P.Value': '1.234567890123456789e-301', 'gene': '=SUM(A1:A2)', 'seed': '80'},
        {'metric': '0.12345678901234567890123456789', 'P.Value': '1e-320', 'gene': '00123', 'seed': '11'},
        {'metric': '', 'P.Value': '0', 'gene': 'SEPT1', 'seed': '61'},
    ], note='Values are preserved.')
    package.write_workbook('test.xlsx', [sheet])
    with ZipFile(tmp_path / 'test.xlsx') as z:
        xml = z.read('xl/worksheets/sheet1.xml').decode()
    assert '<v>0.12345678901234567890123456789</v>' in xml
    assert '<f>' not in xml
    assert '1e-320' in xml
    assert package.catalog[0]['rows'] == 3
    sheet.rows[0]['metric'] = '0.5'
    with pytest.raises(ValueError, match='numeric precision mismatch'):
        package.validate_workbook(tmp_path / 'test.xlsx', [(sheet, 3)])


def test_overwrite_requires_owned_destination(tmp_path):
    destination = tmp_path / 'package'
    destination.mkdir()
    with pytest.raises(ValueError, match='Destination exists'):
        mod.validate_destination(destination, False)
    with pytest.raises(ValueError, match='not owned'):
        mod.validate_destination(destination, True)
    (destination / 'README.md').write_text(f'Package builder: {mod.OWNER}\n')
    mod.validate_destination(destination, True)
    link = tmp_path / 'linked'
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        mod.validate_destination(link, True)


def test_identifier_and_missing_conventions():
    assert mod.cell_value('00123', 'gene_id') == ('s', '00123')
    assert mod.cell_value('', 'ap') == ('s', 'NA')
    assert mod.cell_value('1e-320', 'P.Value') == ('s', '1e-320')
    assert mod.cell_value('0123', 'value', {'value'}) == ('s', '0123')


def test_long_provenance_is_split_without_truncation(tmp_path):
    import json
    package = object.__new__(mod.Package)
    original = ['/original/' + str(i) + '/' + 'x' * 200 for i in range(200)]
    package.sources = {'table.csv': 'hash'}
    package.provenance = {'table.csv': json.dumps(original)}
    source = package.source_sheet([mod.Sheet('Table', ['value'], [], source_counts={'table.csv': 96})])
    assert [r['original_source'] for r in source.rows] == original
    assert all(r['included_rows'] == 96 for r in source.rows)


def test_marker_membership_stays_seed_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'SEEDS', {'11', '80'})
    monkeypatch.setattr(mod, 'TASK_NAMES', {'task': 'Task'})
    package = object.__new__(mod.Package)
    package.archive = tmp_path
    package.sources, package.provenance, package.omissions = {}, {}, []
    package.context = {('task', seed): {'dataset': 'Dataset', 'selected_model': 'RF'} for seed in mod.SEEDS}
    for seed in sorted(mod.SEEDS):
        root = tmp_path / 'data/markers/task' / ('seed_' + seed)
        (root / 'de_results').mkdir(parents=True)
        (root / 'de_results/limma_selected_features.txt').write_text('gene' + seed + '\n')
        (root / 'de_results/limma_voom_de_results.csv').write_text(
            'gene_id,Contrast,logFC,P.Value,adj.P.Val\ngene11,A,0.1234567890123456789,1e-300,0.01\ngene80,A,2,0,0\n')
        (root / 'RF_optimized_features.csv').write_text('gene_id,model,seed,source_file\ngene' + seed + ',RF,' + seed + ',original\n')
    sheet = package.marker_sheets()[0]
    assert [(r['seed'], r['feature']) for r in sheet.rows] == [('11', 'gene11'), ('80', 'gene80')]
    assert all(r['selected_model_consensus'] == 'True' for r in sheet.rows)
    assert sheet.rows[0]['logFC'] == '0.1234567890123456789'


def test_s3_s7_guard_ignores_style_indices_but_detects_numeric_change(tmp_path):
    def make(path, value, style):
        from zipfile import ZipFile
        with ZipFile(path, 'w') as z:
            for n in range(3, 8):
                z.writestr(f'xl/worksheets/sheet{n}.xml',
                           f'<worksheet xmlns="{mod.NS[1:-1]}"><sheetData><row r="1"><c r="A1" s="{style}" t="n"><v>{value}</v></c></row></sheetData></worksheet>')
    before, after = tmp_path/'before.xlsx', tmp_path/'after.xlsx'
    make(before, '0.8764705882352942', 1)
    make(after, '0.8764705882352942', 9)
    mod.validate_unchanged_tables(before, after)
    make(after, '0.8764705882352943', 9)
    with pytest.raises(ValueError, match='S3–S7 changed'):
        mod.validate_unchanged_tables(before, after)


def test_s1_collapses_repeats_without_summing_different_training_cohorts():
    package = object.__new__(mod.Package)
    records=[]
    for dataset in ['B-ALL','TCGA-BRCA','APOLLO-BRCA','BRCA proteomics','CLL-Map','DKFZ']:
        for seed in ['11','80']:
            for correction in ['corrected','uncorrected']:
                for evaluation, n in ([('internal','569'),('external','802')] if dataset=='TCGA-BRCA' else [('internal','10')]):
                    records.append(dict(dataset=dataset,analysis=evaluation+correction,seed=seed,role='training',
                                        n_samples=n,class_name='Class A',class_count=n,preserved_covariates='["age"]',
                                        batch_variable='cohort' if correction=='corrected' else '',modality='rnaseq',evaluation=evaluation))
    package.read=lambda path: (list(records[0]), records)
    result=package.concise_datasets()
    assert len(result.rows)==6
    tcga=next(r for r in result.rows if r['Dataset']=='TCGA-BRCA')
    assert tcga['N']=='Training (internal evaluation): 569\nTraining (APOLLO evaluation): 802'
    assert result.source_counts['tables/dataset_characteristics.csv']==len(records)


def test_publication_projection_removes_internal_fields_and_setting_rows():
    sheets=[mod.Sheet('Sources',['source_path'],[{'source_path':'/data01/private/file.csv'}]),
            mod.Sheet('S2_Detailed_settings',['dataset','analysis','seed','parameter','value','source_file','evidence'],[
                dict(dataset='B-ALL',analysis='b_all_old_de',seed='80',parameter='project.meta_path',value='"/data01/private/meta.csv"',source_file='/tmp/source'),
                dict(dataset='B-ALL',analysis='b_all_old_de',seed='80',parameter='runtime.scoring',value='"f1_macro"',source_file='/tmp/source',evidence='frozen_config'),
                dict(dataset='B-ALL',analysis='b_all_old_de',seed='80',parameter='runtime.n_jobs',value='5'),
                dict(dataset='B-ALL',analysis='b_all_old_de',seed='80',parameter='runtime.random_state',value='80'),
            ])]
    result=mod.publication_sheets(sheets)
    assert len(result)==1 and len(result[0].rows)==2
    assert 'source_file' not in result[0].columns
    assert result[0].labels['seed']=='Seed'
    assert result[0].rows[0]['analysis']=='B-ALL'
    assert result[0].rows[0]['value']=='"f1_macro"'
    assert result[0].rows[1]['value']=='80'


@pytest.mark.parametrize('location',['/data01/private/file.csv',r'C:\\results\\file.csv',r'\\server\share\file.csv'])
def test_path_scan_detects_paths_in_nonworksheet_metadata(tmp_path,location):
    from xml.sax.saxutils import escape
    path=tmp_path/'metadata.xlsx'
    with ZipFile(path,'w') as z:
        z.writestr('docProps/core.xml','<properties><description>'+escape(location)+'</description></properties>')
    with pytest.raises(ValueError,match='Absolute filesystem path'):
        mod.validate_no_filesystem_information(path)


def test_journal_marker_projection_preserves_statistics_and_deduplicates():
    package = object.__new__(mod.Package)
    base = dict(analysis='b_all_old_de',seed='11',feature='00123',record_type='DE_test',
                contrast='A versus B',logFC='0.1234567890123456789',
                **{'P.Value':'1e-320','adj.P.Val':'0.001','score':'2.5'},selected_model_consensus='')
    other = dict(base, seed='80',selected_model_consensus='True')
    package.marker_sheets=lambda: [mod.Sheet('B-ALL',[],[base,dict(base),other]),mod.Sheet('Sources',[],[{}])]
    sheet=package.journal_markers()[0]
    assert len(sheet.rows)==2
    assert [r['Seed'] for r in sheet.rows]==['11','80']
    assert sheet.rows[0]['Used by Final Selected Model']==''
    assert sheet.rows[0]['Feature']=='00123'
    assert sheet.rows[0]['logFC']==base['logFC']
    assert sheet.rows[0]['P Value']=='1e-320'
    assert sheet.rows[0]['Selection Contrast / Source']=='Differential expression: A versus B'
    assert 'NA' in sheet.note
    assert len(sheet.columns)==14


def test_cll_convention_does_not_relabel_macro_f1_as_binary():
    result=mod.Package.convention(dict(f1_averaging='macro',precision_recall_averaging='macro',
                                      curve_averaging='binary',positive_class='unmutated'))
    assert 'F1: macro' in result
    assert 'AUROC/AP: binary' in result
    assert 'positive class: unmutated' in result


def test_archive_snapshot_detects_content_and_inventory_changes(tmp_path):
    f=tmp_path/'value.csv';f.write_text('a')
    before=mod.archive_snapshot(tmp_path)
    f.write_text('b')
    assert mod.archive_snapshot(tmp_path)!=before
    f.write_text('a');(tmp_path/'extra').write_text('')
    assert mod.archive_snapshot(tmp_path)!=before


def test_journal_settings_excludes_unverified_search_bounds():
    package=object.__new__(mod.Package)
    package.read=lambda _: ([],[dict(analysis='b_all_old_de',seed='80',
        parameter='runtime.cv_folds',value='999',evidence='current_code_reference_NOT_verified_used')])
    sheet=package.journal_settings()
    assert sheet.columns==['Section','Component / Model','Parameter','Value / Search Space','Notes']
    assert not any('999' in str(r) for r in sheet.rows)
    assert any('class-balanced' in str(r) for r in sheet.rows)


def test_new_destination_can_be_rebuilt_without_readme(tmp_path):
    from openpyxl import Workbook
    for name in ['Supplementary_Tables.xlsx','Supplementary_Data.xlsx']:
        book=Workbook();book.properties.creator='Omiclass';book.save(tmp_path/name)
    mod.validate_destination(tmp_path,True)
    (tmp_path/'unrelated.txt').write_text('preserve')
    with pytest.raises(ValueError,match='not owned'):
        mod.validate_destination(tmp_path,True)


def test_final_archive_journal_tables(tmp_path):
    archive=PATH.parent/'supplementary'
    if not archive.is_dir():
        pytest.skip('Final manuscript archive is not available')
    package=mod.Package(archive,tmp_path)
    sheets=package.journal_tables()
    assert [len(s.rows) for s in sheets][2:]==[50,16,40,96,24]
    assert len(sheets[0].rows)==6
    original=package.read('tables/classification_per_seed.csv')[1]
    for source,row in zip(original,sheets[2].rows):
        for dest,src in [('Accuracy','accuracy'),('F1','macro_f1'),('Recall','macro_recall'),
                         ('Precision','macro_precision'),('AUROC','auroc'),('Average Precision','ap')]:
            assert row[dest]==source[src]
        assert row['Selected Model']==mod.readable_identifier(source['selected_model'],'model')
    significant=[r for r in sheets[3].rows if r['Significant After Holm Correction']=='True']
    assert len(significant)==1
    assert significant[0]['Evaluation']=='TCGA→APOLLO'
    assert significant[0]['Metric']=='Accuracy'
    source=package.read('tables/Table_S6_foundation_model_single_run_results.csv')[1]
    assert [(r['Accuracy'],r['F1']) for r in sheets[5].rows]==[(r['accuracy'],r['f1']) for r in source]
    source=package.read('tables/Table_S7_knn_purity.csv')[1]
    assert [r['kNN Purity'] for r in sheets[6].rows]==[r['knn_purity'] for r in source]
    assert sheets[6].columns==['Task','Representation','Number of Samples','k','kNN Purity']
    assert all('Analysis Seed' not in r for r in sheets[6].rows)


def test_marker_worksheets_split_by_analysis_without_changing_rows():
    columns = ['Analysis', 'Seed', 'Feature']
    rows = [
        dict(zip(columns, ['CLL internal corrected', '80', '00123'])),
        dict(zip(columns, ['B-ALL internal', '11', 'GENE1'])),
        dict(zip(columns, ['B-ALL internal', '80', 'GENE2'])),
    ]
    sheets = mod.Package.split_marker_rows(columns, rows)
    assert [s.name for s in sheets] == ['B-ALL internal', 'CLL internal corrected']
    assert [s.rows for s in sheets] == [[rows[1], rows[2]], [rows[0]]]
    assert all(s.columns == columns and len(s.name) <= 31 for s in sheets)
    assert all(len({r['Analysis'] for r in s.rows}) == 1 for s in sheets)
    assert all('NA' in s.note for s in sheets)
    with pytest.raises(ValueError, match='Unknown marker training analysis'):
        mod.Package.split_marker_rows(columns, [dict(Analysis='unknown')])
