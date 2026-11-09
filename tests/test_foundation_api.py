"""Synthetic API coverage; no vendor code, real checkpoints, or manuscript runs."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier

import api
from omiclass.config.loaders import load_config
from omiclass.config.schema import FoundationCfg
from omiclass.foundation import extraction
from omiclass.foundation.io import write_json


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.RandomState(21)
    ids = [f's{i:02}' for i in range(30)]
    counts = pd.DataFrame(rng.randint(1, 100, (6, 30)), index=[f'g{i}' for i in range(6)], columns=ids)
    expression = tmp_path / 'counts.csv'
    counts.to_csv(expression)
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame({'label': ['a' if i % 2 == 0 else 'b' for i in range(30)],
                  'batch': ['A' if i % 3 else 'B' for i in range(30)]}, index=ids).to_csv(metadata)
    split = tmp_path / 'split.json'
    write_json(split, {'train_index': ids[:24], 'test_index': ids[24:]})
    return counts, expression, metadata, split


@pytest.fixture
def mock_encoders(monkeypatch):
    calls = []
    def extract(counts, cfg, out, metadata=None):
        out.mkdir(parents=True, exist_ok=False)
        calls.append((cfg.foundation.model, cfg.foundation.representation, counts.columns.tolist()))
        values = counts.T.astype(np.float32) / 10
        values.columns = [f'emb_{i:04d}' for i in range(values.shape[1])]
        token_cache = None
        if cfg.foundation.representation == 'tokens':
            token_cache = out / 'tokens'
            observer = extraction.TokenObserver(token_cache, extraction.backend_name(cfg),
                cfg.cellfm.num_views if cfg.foundation.model.startswith('cellfm') else 1, True)
            for view in range(observer.num_views):
                tokens = np.repeat(values.to_numpy()[:, None, :], 4, axis=1) + np.arange(4)[None, :, None]*0.01 + view*0.001
                observer(tokens.astype(np.float32), np.ones(tokens.shape[:2], bool),
                    np.tile(np.arange(4), (len(values), 1)), values.index, view, None)
            observer.finish(values.index)
        return extraction.Extraction(values, {'mocked_encoder': True}, token_cache)
    monkeypatch.setattr(extraction, 'extract', extract)
    return calls


@pytest.fixture
def tiny_native(monkeypatch):
    from omiclass.train.trainer import Trainer, TrainSummary
    def train(self, X, y):
        model = self._with_bulkformer_reduction(lambda: RandomForestClassifier(n_estimators=2, random_state=80, n_jobs=1))()
        model.fit(X, y)
        path = self.artifacts_dir / 'rf.joblib'
        joblib.dump(model, path)
        return TrainSummary(pd.DataFrame([dict(model='plain_random_forest', status='ok', f1_macro_mean=0.5)]), {'plain_random_forest': path})
    monkeypatch.setattr(Trainer, 'run', train)


CASES = [(r['model'], r['representation'], r['classifier'], r['pooling']) for r in api.foundation_configurations()]


@pytest.mark.parametrize('model,representation,classifier,pooling', CASES)
def test_api_train_save_load_and_external_prediction(tmp_path, dataset, mock_encoders, tiny_native,
                                                       model, representation, classifier, pooling):
    counts, expression, metadata, split = dataset
    result = api.run(expression, foundation=dict(enabled=True, model=model, representation=representation,
        classifier=classifier, pooling=pooling, split_path=split,
        input_scale='tpm' if model == 'bulkrnabert' else 'counts', lr_c_values=[1.0],
        article_head=dict(epochs=2), bert_head=dict(max_epochs=2, patience=1)),
        meta_path=metadata, classification_column='label', output_dir=tmp_path/'train',
        cv_folds=2, tuning_inner_cv=2,
        **{f'{"cellfm" if model.startswith("cellfm") else model}.reduction.supervised_dimensions': 4,
           f'{"cellfm" if model.startswith("cellfm") else model}.reduction.pca_components': 3})
    manifest = json.loads((result.out_dir/'foundation.json').read_text())
    assert manifest['status'] == 'complete'
    assert set(manifest['split']['train_index']).isdisjoint(manifest['split']['test_index'])
    assert manifest['representation'] == representation
    assert manifest['diagnostic'] == (representation == 'cls')
    scope = json.loads((result.out_dir/'training/fit_scope.json').read_text())
    fit_ids = scope.get('representation_fit_ids', scope.get('constant_filter_fit_ids', scope.get('final_fit_ids', [])))
    assert set(fit_ids) <= set(counts.columns[:24])
    external = counts.iloc[:, 24:].copy()
    external.columns = [s+'_external' for s in external.columns]
    external_path = tmp_path/'external.csv'
    external.to_csv(external_path)
    before = {p: p.read_bytes() for p in result.out_dir.rglob('*') if p.is_file()}
    predicted = api.predict(external_path, result.out_dir, tmp_path/'prediction')
    frame = pd.read_csv(predicted.artifacts['predictions'])
    holdout = pd.read_csv(result.out_dir/'holdout_predictions.csv')
    assert frame.sample_id.tolist() == external.columns.tolist()
    assert frame.y_pred.tolist() == holdout.y_pred.tolist()
    pcols = [c for c in frame if c.startswith('probability_')]
    np.testing.assert_allclose(frame[pcols], holdout[pcols], rtol=1e-5, atol=1e-6)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not json.loads((predicted.out_dir/'prediction.json').read_text())['refitted']
    if classifier in {'mlp','mlp_smote','svm'}:
        assert pd.read_csv(result.leaderboard_csv).f1_macro_mean.isna().all()


@pytest.mark.parametrize('settings', [
    dict(model='bulkrnabert', representation='cls', classifier='svm'),
    dict(model='bulkformer', representation='cls', classifier='mlp'),
    dict(model='cellfm', pooling='max'),
    dict(model='cellfm', representation='tokens', classifier='mlp_smote'),
    dict(model='bulkformer', representation='tokens', classifier='lr'),
    dict(model='bulkrnabert', representation='tokens', classifier='svm'),
    dict(model='cellfm', representation='cls', classifier='omiclass'),
    dict(model='cellfm', input_scale='tpm'),
    dict(correction='limma'),
    dict(mode='predict'),
    dict(article_head={'hidden_dims':[0]}),
])
def test_incompatible_combinations_rejected(settings):
    with pytest.raises(ValidationError):
        FoundationCfg(enabled=True, **settings)


def test_flat_and_cli_overrides_parse():
    cfg = load_config({'foundation_enabled':'true','foundation_model':'cellfm_1view',
        'foundation_classifier':'lr','foundation_representation':'supervised_pca',
        'meta_path':'meta.csv','classification_column':'label'})
    assert cfg.cellfm.enabled and cfg.cellfm.num_views == 1
    assert cfg.cellfm.reduction.strategy == 'supervised_pca'
    assert not cfg.runtime.feature_selection.enabled


def test_cli_uses_same_dispatch(monkeypatch):
    from omiclass.foundation import runner
    seen = []
    monkeypatch.setattr(runner,'run_foundation',lambda path,cfg: seen.append((path,cfg)))
    monkeypatch.setattr(sys,'argv',['api.py','--override','expression_file','input.csv',
        '--override','foundation_enabled','true','--override','foundation_classifier','svm',
        '--override','meta_path','meta.csv','--override','classification_column','label'])
    api.run_from_cli()
    assert seen[0][0] == Path('input.csv')
    assert seen[0][1].foundation.classifier == 'svm'


def test_backend_conflicts_and_outcome_correction_rejected():
    common={'meta_path':'meta.csv','classification_column':'label'}
    with pytest.raises(Exception,match='conflicts'):
        load_config(dict(common, foundation=dict(enabled=True,model='bulkformer'),cellfm_enabled=True))
    with pytest.raises(Exception,match='Outcome'):
        load_config(dict(common,foundation=dict(enabled=True,correction='limma',allow_transductive_correction=True),batch_column='label'))


def test_disabled_foundation_serialization_is_legacy_compatible():
    cfg=load_config({'meta_path':'meta.csv','classification_column':'label'})
    assert not cfg.foundation.enabled
    assert 'foundation' not in cfg.model_dump(mode='json')
    assert cfg.runtime.feature_selection.enabled


def test_marker_pipeline_keeps_original_dispatch(monkeypatch,tmp_path):
    captured=[]
    class Orchestrator:
        def __init__(self,**kwargs):captured.append(kwargs)
        def run(self):return 'legacy-result'
    monkeypatch.setitem(sys.modules,'omiclass.pipeline.orchestrator',SimpleNamespace(PipelineOrchestrator=Orchestrator))
    result=api.run(tmp_path/'expression.csv',meta_path=tmp_path/'meta.csv',classification_column='label',
        markers_path=tmp_path/'markers.txt',output_dir=tmp_path/'marker-output')
    assert result=='legacy-result'
    assert captured[0]['cfg'].project.geneset_path==tmp_path/'markers.txt'
    assert not captured[0]['cfg'].foundation.enabled
    assert not captured[0]['opts'].run_de


def test_predict_cannot_select_training_route():
    with pytest.raises(ValueError,match='cannot switch'):
        api.predict(Path('x.csv'),Path('saved'),Path('new'),**{'foundation.mode':'train'})
    with pytest.raises(ValueError,match='cannot disable'):
        api.predict(Path('x.csv'),Path('saved'),Path('new'),foundation_enabled=False)


def test_saved_artifact_corruption_rejected_before_extraction(tmp_path,dataset,mock_encoders):
    _,expression,metadata,split=dataset
    result=api.run(expression,foundation=dict(enabled=True,classifier='svm',split_path=split),
        meta_path=metadata,classification_column='label',output_dir=tmp_path/'train')
    next(iter(result.artifacts.values())).write_bytes(b'changed artifact')
    call_count=len(mock_encoders)
    with pytest.raises(ValueError,match='artifact changed'):
        api.predict(expression,result.out_dir,tmp_path/'prediction')
    assert len(mock_encoders)==call_count


def test_prediction_cannot_change_representation_or_classifier(tmp_path,dataset,mock_encoders):
    _,expression,metadata,split=dataset
    result=api.run(expression,foundation=dict(enabled=True,classifier='svm',split_path=split),
        meta_path=metadata,classification_column='label',output_dir=tmp_path/'train')
    with pytest.raises(ValueError,match='configuration'):
        api.predict(expression,result.out_dir,tmp_path/'prediction',foundation_classifier='mlp')


@pytest.mark.parametrize('model',['bulkformer','cellfm_20view','bulkrnabert'])
def test_corrected_external_prediction_requires_full_input(tmp_path,dataset,mock_encoders,model):
    counts,expression,metadata,split=dataset
    result=api.run(expression,foundation=dict(enabled=True,model=model,classifier='svm',split_path=split,
        correction='limma',allow_transductive_correction=True),
        meta_path=metadata,classification_column='label',batch_column='batch',output_dir=tmp_path/'train')
    external=counts.iloc[:,:3].copy();external.columns=['external1','external2','external3']
    path=tmp_path/'external.csv';external.to_csv(path)
    with pytest.raises(ValueError,match='full original input cohort'):
        api.predict(path,result.out_dir,tmp_path/'prediction')
    pd.concat([counts,external],axis=1).to_csv(path)
    result=api.predict(path,result.out_dir,tmp_path/'prediction')
    assert pd.read_csv(result.artifacts['predictions']).sample_id.tolist()==external.columns.tolist()


@pytest.mark.parametrize('pooling',['max','median'])
def test_existing_bulkformer_extra_pooling_is_explicit(pooling):
    cfg=load_config(dict(meta_path='m.csv',classification_column='label',
        foundation=dict(enabled=True,classifier='svm',pooling=pooling)))
    assert cfg.bulkformer.aggregate_type==pooling


@pytest.mark.parametrize('model,representation,classifier',[
    ('bulkformer','supervised_pca','lr'),('bulkformer','full','omiclass'),
    ('cellfm_1view','cls','mlp_smote'),('cellfm_20view','tokens','mlp'),
    ('bulkrnabert','supervised_pca','mlp'),('bulkrnabert','full','svm'),('bulkrnabert','tokens','mlp')])
def test_multiclass_canonical_order_and_roundtrip(tmp_path,dataset,mock_encoders,tiny_native,model,representation,classifier):
    counts,expression,metadata,split=dataset
    meta=pd.read_csv(metadata,index_col=0);meta['label']=['a','b','c']*10;meta.to_csv(metadata)
    result=api.run(expression,foundation=dict(enabled=True,model=model,representation=representation,
        classifier=classifier,split_path=split,lr_c_values=[1.0],article_head=dict(epochs=2),bert_head=dict(max_epochs=2,patience=1)),
        meta_path=metadata,classification_column='label',classes_names=['c','b','a'],output_dir=tmp_path/'train',
        cv_folds=2,tuning_inner_cv=2,
        **{f'{"cellfm" if model.startswith("cellfm") else model}.reduction.supervised_dimensions':4,
           f'{"cellfm" if model.startswith("cellfm") else model}.reduction.pca_components':3})
    manifest=json.loads((result.out_dir/'foundation.json').read_text());assert manifest['classes']==['c','b','a']
    external=counts.iloc[:,24:].copy();external.columns=[x+'_new' for x in external.columns]
    path=tmp_path/'external.csv';external.to_csv(path)
    predicted=api.predict(path,result.out_dir,tmp_path/'prediction')
    frame=pd.read_csv(predicted.artifacts['predictions']);holdout=pd.read_csv(result.out_dir/'holdout_predictions.csv')
    assert frame.predicted_label.tolist()==[['c','b','a'][i] for i in frame.y_pred]
    columns=['probability_0','probability_1','probability_2']
    np.testing.assert_allclose(frame[columns],holdout[columns],rtol=1e-5,atol=1e-6)


def test_changed_encoder_asset_rejected_before_prediction(tmp_path,dataset,mock_encoders):
    _,expression,metadata,split=dataset
    asset=tmp_path/'checkpoint.pt';asset.write_bytes(b'synthetic identity only')
    result=api.run(expression,foundation=dict(enabled=True,classifier='svm',split_path=split),
        meta_path=metadata,classification_column='label',bulkformer_checkpoint_path=asset,output_dir=tmp_path/'train')
    asset.write_bytes(b'changed synthetic checkpoint')
    count=len(mock_encoders)
    with pytest.raises(ValueError,match='Encoder assets changed'):
        api.predict(expression,result.out_dir,tmp_path/'prediction')
    assert len(mock_encoders)==count
