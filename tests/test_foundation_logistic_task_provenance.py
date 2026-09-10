"""Keep TCGA holdouts distinct from the external APOLLO evaluation."""
import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / 'final_results_runs/evaluate_foundation_logistic.py'
spec = importlib.util.spec_from_file_location('foundation_logistic_task_test', SCRIPT)
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


def test_tcga_frozen_split_does_not_require_apollo_cohort(tmp_path):
    config = {'project': {'meta_path': str(tmp_path / 'tcga_metadata.csv'),
                          'hold_out_path': None}}
    assert evaluator._source_provenance_issues(
        tmp_path, backend='bulkformer', task='brca', config=config,
        cellfm_num_views=20,
    ) == []


def test_apollo_still_rejects_tcga_metadata_and_holdout(tmp_path):
    config = {'project': {'meta_path': str(tmp_path / 'tcga_metadata.csv'),
                          'hold_out_path': None}}
    for task in ['apolo', 'apollo']:
        issues = evaluator._source_provenance_issues(
            tmp_path, backend='bulkformer', task=task, config=config,
            cellfm_num_views=20,
        )
        assert any(issue.startswith('brca_metadata:') for issue in issues)
        assert any(issue.startswith('brca_holdout:') for issue in issues)


def test_apollo_accepts_its_predefined_cohort(tmp_path):
    config = {'project': {'meta_path': str(evaluator.APOLLO_BRCA_COMBINED_METADATA),
                          'hold_out_path': str(evaluator.APOLLO_BRCA_HOLDOUT)}}
    assert evaluator._source_provenance_issues(
        tmp_path, backend='bulkformer', task='apolo', config=config,
        cellfm_num_views=20,
    ) == []
