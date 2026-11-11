"""Public packaging/discovery/example contracts; no build or model experiments."""
import ast
import configparser
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_preserves_imports_and_declares_direct_dependencies():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / 'setup.cfg')
    assert cfg['options']['py_modules'] == 'api'
    assert cfg['options']['packages'] == 'find_namespace:'
    assert not cfg['metadata'].get('license')
    assert not cfg['metadata'].get('license_files', '').strip()
    for dependency in ['PyYAML', 'requests', 'joblib', 'statsmodels', 'lifelines']:
        assert dependency in cfg['options']['install_requires']
    assert 'torch' in cfg['options.extras_require']['foundation']


def test_package_resources_include_r_and_exclude_unapproved_data():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / 'setup.cfg')
    assert cfg['options.package_data']['omiclass.batch'] == '*.R'
    assert cfg['options.package_data']['omiclass.de'] == '*.R'
    assert '*.csv' not in ''.join(cfg['options.package_data'].values())
    manifest = (ROOT / 'MANIFEST.in').read_text()
    assert 'prune examples/data' in manifest
    assert 'exclude LICENSE' in manifest
    assert 'exclude omiclass/data/filtered_surface_list.txt' in manifest


def test_public_test_collection_has_no_research_module_imports():
    forbidden = {'experiments', 'final_results_runs', 'comparison_tools', 'manuscript_figures'}
    for path in (ROOT / 'tests').glob('test_*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, str(path)
                modules = [node.module or '']
            else:
                continue
            assert not any(module.split('.')[0] in forbidden for module in modules), str(path)


@pytest.mark.parametrize('script', ['run_python_example.py', 'run_python_re_run_example.py'])
def test_example_argument_help_is_executable_without_data(script, monkeypatch):
    namespace = runpy.run_path(str(ROOT / 'examples' / script))
    monkeypatch.setattr(sys, 'argv', [script, '--help'])
    with pytest.raises(SystemExit) as stopped:
        namespace['main']()
    assert stopped.value.code == 0


def test_prediction_example_rejects_html_only_run(tmp_path, monkeypatch):
    (tmp_path / 'final_report.html').write_text('Illustration only')
    namespace = runpy.run_path(str(ROOT / 'examples/run_python_re_run_example.py'))
    monkeypatch.setattr(sys, 'argv', ['example', '--expression', 'expression.csv', '--metadata', 'meta.csv', '--run-dir', str(tmp_path)])
    monkeypatch.setitem(namespace['main'].__globals__, 'run', lambda *a, **kw: pytest.fail('Accidental training'))
    with pytest.raises(SystemExit) as stopped:
        namespace['main']()
    assert stopped.value.code == 2


def test_marker_example_explicitly_uses_prepared_expression(tmp_path, monkeypatch):
    namespace = runpy.run_path(str(ROOT / 'examples/run_python_example.py'))
    seen = []
    monkeypatch.setitem(namespace['main'].__globals__, 'run',
                        lambda *a, **kw: (seen.append(kw) or SimpleNamespace(out_dir=tmp_path)))
    monkeypatch.setattr(sys, 'argv', ['example', '--expression', 'expression.csv', '--metadata', 'meta.csv',
        '--markers', 'markers.txt', '--output', str(tmp_path)])
    namespace['main']()
    assert seen[0]['raw_count'] is False and seen[0]['log_transform'] is False
    assert seen[0]['run_de'] is False and seen[0]['feature_selection_enabled'] is False


def test_prediction_example_does_not_override_saved_input_scale(tmp_path, monkeypatch):
    for name in ['leaderboard.csv','config.freeze.yaml']:
        (tmp_path / name).write_text('fixture')
    namespace = runpy.run_path(str(ROOT / 'examples/run_python_re_run_example.py'))
    seen = []
    monkeypatch.setitem(namespace['main'].__globals__, 'run',
        lambda *a,**kw: (seen.append(kw) or SimpleNamespace(out_dir=tmp_path)))
    monkeypatch.setattr(sys, 'argv', ['example','--expression','input.csv','--metadata','meta.csv','--run-dir',str(tmp_path)])
    namespace['main']()
    assert 'raw_count' not in seen[0]
    (tmp_path / 'foundation.json').write_text('{}')
    with pytest.raises(SystemExit) as stopped:
        namespace['main']()
    assert stopped.value.code == 2 and len(seen) == 1


def test_docker_uses_public_environment_and_private_context_exclusions():
    docker = (ROOT / 'Dockerfile').read_text()
    ignored = (ROOT / '.dockerignore').read_text()
    assert 'COPY environment.yml /tmp/environment.yml' in docker
    assert 'environment-viz-env' not in docker
    for name in ['.step3-validation', 'examples/data', 'omiclass/data/*.csv', 'CellFM']:
        assert name in ignored
    assert '/app/results/runtime_provenance.txt' in (ROOT / 'docker/entrypoint.sh').read_text()
