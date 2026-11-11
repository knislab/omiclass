"""New raw-count routing and historical-scale replay; no new normalization method."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from omiclass.cli.runtime import create_runtime_options_from_path_conf_kwargs
from omiclass.config.loaders import load_config
from omiclass.data.processor import DataProcessor
from omiclass.pipeline.steps.raw_count_norm_step import RawCountNormalizationStep


def config(tmp_path, **kwargs):
    return load_config(dict(meta_path=tmp_path / 'meta.csv', classification_column='class',
        output_dir=tmp_path / 'run', **{'batch_correction.algorithm':'none'}, **kwargs))


@pytest.mark.parametrize('markers', [False, True])
def test_explicit_raw_training_normalizes_even_without_de_or_batch(tmp_path, markers):
    cfg = config(tmp_path, **({'markers_path':tmp_path / 'markers.txt'} if markers else {}))
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'counts.csv',
        {'raw_count':True,'run_de':False})
    assert opts.mode == 'train-only' and opts.input_mode == 'raw_counts'
    assert opts.counts_path == tmp_path / 'counts.csv'
    assert opts.run_de is False
    assert RawCountNormalizationStep().should_run(SimpleNamespace(cfg=cfg, opts=opts, cache={}))


def test_prepared_expression_route_is_unchanged(tmp_path):
    cfg = config(tmp_path, markers_path=tmp_path / 'markers.txt')
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'expression.csv',
        {'raw_count':False,'run_de':False})
    assert opts.mode == 'train-only' and opts.input_mode == 'ready_for_training'
    assert not RawCountNormalizationStep().should_run(SimpleNamespace(cfg=cfg, opts=opts, cache={}))


@pytest.mark.parametrize('stored_mode', ['ready_for_training','raw_counts'])
def test_saved_input_mode_remains_authoritative(tmp_path, stored_mode):
    cfg = config(tmp_path, markers_path=tmp_path / 'markers.txt')
    root = cfg.project.out_dir
    (root / 'preprocessing').mkdir(parents=True)
    (root / 'leaderboard.csv').write_text('model\nfixture\n')
    (root / 'preprocessing/scoped.json').write_text(json.dumps({'input_mode':stored_mode}))
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'prediction.csv', {})
    assert opts.mode == 'clean_and_predict'
    assert (opts.input_mode == 'raw_counts') == (stored_mode == 'raw_counts')
    with pytest.raises(ValueError, match='input scale'):
        create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'prediction.csv',
            {'raw_count':stored_mode != 'raw_counts'})


def test_existing_normalization_is_used_once_and_saved_lengths_are_replayed(tmp_path, monkeypatch):
    lengths = tmp_path / 'lengths.csv'
    pd.Series([1000.,2000.], index=['g1','g2'], name='length').to_csv(lengths)
    counts = pd.DataFrame([[1.,3.],[5.,7.]], index=['g1','g2'], columns=['a','b'])
    cfg = config(tmp_path, gene_lengths_path=lengths, markers_path=tmp_path / 'markers.txt')
    opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'counts.csv',
        {'raw_count':True,'run_de':False})
    ctx = SimpleNamespace(cfg=cfg,opts=opts,out_dir=cfg.project.out_dir,cache={'full_expr':counts})
    step = RawCountNormalizationStep()
    step.run(ctx)
    rpk = counts.div(pd.Series([1.,2.],index=counts.index),axis=0)
    np.testing.assert_allclose(ctx.cache['full_expr'], np.log1p(rpk.div(rpk.sum(0),axis=1)*1e6))
    assert not step.should_run(ctx)  # This call's normalization is already recorded.
    assert (ctx.out_dir / 'preprocessing/raw_count_gene_lengths.joblib').is_file()
    (ctx.out_dir / 'preprocessing/scoped.json').write_text(json.dumps(dict(
        input_mode='raw_counts',frozen_raw_count_normalization=True)))
    (ctx.out_dir / 'leaderboard.csv').write_text('model\nfixture\n')
    monkeypatch.setattr(DataProcessor,'load_precomputed_gene_lengths',
                        lambda *a,**kw: pytest.fail('Prediction refitted normalization lengths'))
    pred_opts = create_runtime_options_from_path_conf_kwargs(cfg, tmp_path / 'external.csv', {})
    external = pd.DataFrame({'external':[2.,4.]},index=counts.index)
    pred = SimpleNamespace(cfg=cfg,opts=pred_opts,out_dir=ctx.out_dir,cache={'full_expr':external})
    step.run(pred)
    expected = np.log1p(pd.DataFrame({'external':[500000.,500000.]},index=counts.index))
    np.testing.assert_allclose(pred.cache['full_expr'], expected)
