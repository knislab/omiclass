"""Small real downstream fits with synthetic frozen embeddings; no encoders/GPU."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import api
from omiclass.config.schema import FoundationCfg
from omiclass.foundation import extraction, heads, runner
from omiclass.features.bulkformer_reduction import EmbeddingRepresentation


@pytest.fixture
def comparison_data(tmp_path, monkeypatch):
    ids = [f's{i:02}' for i in range(64)]
    counts = pd.DataFrame(np.random.RandomState(5).randint(1, 100, (8, 64)),
                          index=[f'g{i}' for i in range(8)], columns=ids)
    expression, metadata, split = [tmp_path/name for name in ['counts.csv', 'meta.csv', 'split.json']]
    counts.to_csv(expression)
    pd.DataFrame({'label': ['a', 'b']*32}, index=ids).to_csv(metadata)
    split.write_text(json.dumps(dict(train_index=ids[:48], test_index=ids[48:])))
    calls, fits = [], []

    def extract(counts, cfg, out, metadata=None):
        out.mkdir(parents=True)
        calls.append((cfg.model_copy(deep=True), counts.columns.tolist()))
        X = counts.T.astype(float)/10
        X.columns = [f'emb_{i}' for i in range(8)]
        return extraction.Extraction(X, {'synthetic': True}, None)
    monkeypatch.setattr(extraction, 'extract', extract)
    original = EmbeddingRepresentation.fit
    def record(self, values, labels=None):
        fits.append(set(values.index))
        return original(self, values, labels)
    monkeypatch.setattr(EmbeddingRepresentation, 'fit', record)

    # Replace only the expensive native search. Retain the actual native adapter,
    # genuine outer OOF predictions, final fitting, serialisation, and selection.
    from omiclass.train.trainer import Trainer, TrainSummary
    def train(self, X, y):
        from omiclass.train.selection import selection_summary
        rows, paths = [], {}
        for name, builder in [('plain_random_forest', lambda: RandomForestClassifier(n_estimators=2,
                max_depth=2, random_state=self.cfg.runtime.random_state)),
                              ('dummy', lambda: DummyClassifier(strategy='most_frequent'))]:
            pred, proba = np.empty(len(y), dtype=int), np.empty((len(y), 2))
            folds = []
            for number, (tr, va) in enumerate(heads.safe_cv(y, 2, self.cfg.runtime.random_state).split(X, y), 1):
                model = Pipeline([('representation', heads.representation(self.cfg)), ('model', builder())])
                model.fit(X.iloc[tr], y.iloc[tr])
                pred[va], proba[va] = model.predict(X.iloc[va]), model.predict_proba(X.iloc[va])
                from sklearn.metrics import f1_score, accuracy_score
                score = float(f1_score(y.iloc[va], pred[va], average='macro'))
                folds.append(dict(fold=number, accuracy=accuracy_score(y.iloc[va], pred[va]),
                                  f1_macro=score, selection_score=score))
            cv = SimpleNamespace(oof_true=y.to_numpy(), oof_pred=pred, oof_proba=proba,
                                 classes=np.arange(2), per_fold_metrics=folds)
            self.cv_results[name] = cv
            model = Pipeline([('representation', heads.representation(self.cfg)), ('model', builder())]).fit(X, y)
            path = self.artifacts_dir/f'{name}.joblib'
            joblib.dump(model, path)
            paths[name] = path
            rows.append(dict(model=name, status='ok', f1_macro_mean=np.mean([r['f1_macro'] for r in folds]),
                             **selection_summary(folds, self.cfg.runtime.scoring)))
        return TrainSummary(pd.DataFrame(rows), paths)
    monkeypatch.setattr(Trainer, 'run', train)
    return ids, expression, metadata, split, calls, fits


@pytest.mark.parametrize('model,representation', [('bulkformer','full'), ('bulkrnabert','supervised_pca'),
    ('cellfm_1view','supervised'), ('cellfm_20view','pca')])
def test_shared_extraction_cv_all_heads_reports_and_predict(tmp_path, comparison_data, model, representation):
    ids, expression, metadata, split, calls, fits = comparison_data
    backend = 'cellfm' if model.startswith('cellfm') else model
    result = api.run(expression, meta_path=metadata, classification_column='label',
        output_dir=tmp_path/'train', random_state=11, cv_folds=2, tuning_inner_cv=2,
        foundation=dict(enabled=True, model=model, classifiers=['mlp','mlp_smote','svm','lr','omiclass'],
            representation=representation, input_scale='tpm' if model=='bulkrnabert' else 'counts',
            split_path=split, article_head=dict(epochs=1), bert_head=dict(max_epochs=1,patience=1), lr_c_values=[1.0]),
        **{f'{backend}.reduction.supervised_dimensions': 4, f'{backend}.reduction.pca_components': 3})
    assert len(calls)==1
    cfg = calls[0][0]
    assert cfg.foundation.extraction_seed==cfg.runtime.random_state==11
    assert getattr(cfg,backend).reduction.random_state==11
    assert extraction.encoder_seed(cfg)==(11,'runtime.random_state')
    if backend=='cellfm': assert cfg.cellfm.random_state==11
    assert fits and all(part<=set(ids[:48]) for part in fits)
    assert any(len(part)<24 for part in fits)  # inner folds or BERT inner validation
    out=result.out_dir
    manifest=json.loads((out/'foundation.json').read_text())
    board=pd.read_csv(out/'leaderboard.csv')
    assert len(board)==6 and set(board.classifier)=={'mlp','mlp_smote','svm','lr','omiclass'}
    selected=json.loads((out/'selection.json').read_text())
    winner=board.sort_values(['selection_score_mean','model'],ascending=[False,True]).iloc[0].model
    assert selected['selected_model']==manifest['selected_model']==winner
    assert selected['source']=='training_outer_cv' and selected['holdout_used'] is False
    cv=pd.read_csv(out/'cv_metrics.csv')
    assert set(cv.model)==set(board.model) and cv.groupby('model').size().eq(2).all()
    assert set(pd.read_csv(out/'oof_predictions.csv').sample_id)==set(ids[:48])
    test=pd.read_csv(out/'test_metrics.csv')
    assert set(test.model)==set(board.model) and test.n_test.eq(16).all()
    assert np.isfinite(test[['accuracy','f1_macro','precision_macro','recall_macro']]).all().all()
    html=(out/'final_report.html').read_text()
    assert 'training_outer_cv' in html and winner in html and 'Holdout used for selection' in html
    assert all(metric in html for metric in ['precision_macro','recall_macro','f1_macro','accuracy'])
    for key in board.model:
        assert f'reports/{key}.html' in html and (out/f'reports/{key}.html').is_file()
        assert '../final_report.html' in (out/f'reports/{key}.html').read_text()
    assert joblib.load(out/'models/svm__svm.joblib')['svm'].random_state==11
    assert joblib.load(out/'models/lr__logistic_regression.joblib')['logistic_regression'].random_state==11
    for classifier in ['mlp','mlp_smote']:
        if backend=='bulkrnabert':
            assert json.loads((out/f'training/{classifier}/final/training.json').read_text())['inner_seed']==11
        else:
            assert json.loads((out/f'training/{classifier}/final/article_training/run.json').read_text())['arguments']['seed']==11
    assert '<svg' in html  # native confusion/ROC/PR plots
    source=pd.read_csv(expression,index_col=0).iloc[:,48:]
    source.columns=[name+'_external' for name in source.columns]
    external=tmp_path/'external.csv';source.to_csv(external)
    prediction=api.predict(external,out,tmp_path/'predicted')
    assert len(calls)==2  # one extraction for the entire saved comparison
    frame=pd.read_csv(prediction.artifacts['predictions'])
    assert set(frame.model)==set(board.model) and frame.groupby('model').size().eq(16).all()
    holdout=pd.read_csv(out/'holdout_predictions.csv')
    np.testing.assert_array_equal(frame.y_pred,holdout.y_pred)
    np.testing.assert_allclose(frame[['probability_0','probability_1']],holdout[['probability_0','probability_1']])
    with pytest.raises(ValueError,match='overrides change'):
        api.predict(external,out,tmp_path/'wrong_seed',random_state=12)


@pytest.mark.parametrize('classifiers', [[], ['svm','svm'], ['omiclass','foundation_classifier']])
def test_list_rejects_empty_or_duplicate_protocols(classifiers):
    with pytest.raises(ValueError,match='nonempty and unique'):
        FoundationCfg(enabled=True,classifiers=classifiers)


def test_list_alias_and_historical_encoder_seed():
    cfg=FoundationCfg(enabled=True,classifiers=['foundation_classifier','lr'])
    assert cfg.classifiers==['omiclass','lr']
    from omiclass.config.schema import Config
    old=Config(project=dict(meta_path='unused.csv',cluster_column='label'),foundation=dict(enabled=True,model='bulkrnabert'),runtime=dict(random_state=11))
    assert extraction.encoder_seed(old)==(0,'fixed JAX PRNGKey(0)')
    old.foundation.seed_policy='runtime_v1'
    assert extraction.encoder_seed(old)==(11,'runtime.random_state')
    with pytest.raises(ValueError,match='pooled'):
        FoundationCfg(enabled=True,classifiers=['mlp'],classifier='mlp',representation='tokens')


@pytest.mark.parametrize('seed', [0, 11, 61])
def test_bert_worker_receives_runtime_seed_and_replays_legacy_zero(tmp_path, monkeypatch, seed):
    from omiclass.config.schema import Config
    from omiclass.features.bulkrnabert_worker import encoder_random_key
    cfg=Config(project=dict(meta_path=tmp_path/'unused.csv',cluster_column='label'), foundation=dict(enabled=True, model='bulkrnabert', seed_policy='runtime_v1'),
               runtime=dict(random_state=seed))
    values=pd.DataFrame([[1.,2.]],index=['g'],columns=['a','b'])
    calls=[]
    def run(cmd, **kwargs):
        calls.append(cmd)
        np.savez(tmp_path/'worker-output.npz', embeddings=np.ones((2,256),np.float32),
                 sample_ids=np.array(['a','b']),metadata='{}')
    monkeypatch.setattr(extraction.subprocess,'run',run)
    extraction.extract_bert(values,cfg,tmp_path,False)
    assert calls[-1][calls[-1].index('--seed')+1]==str(seed)
    seen=[]
    jax=SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda value:seen.append(value) or value))
    assert encoder_random_key(jax,seed)==seed
    assert encoder_random_key(jax)==0
    assert seen==[seed,0]
    cfg.foundation.seed_policy=None
    other=tmp_path/'legacy';other.mkdir()
    def legacy(cmd,**kwargs):
        calls.append(cmd)
        np.savez(other/'worker-output.npz',embeddings=np.ones((2,256),np.float32),
                 sample_ids=np.array(['a','b']),metadata='{}')
    monkeypatch.setattr(extraction.subprocess,'run',legacy)
    extraction.extract_bert(values,cfg,other,False)
    assert '--seed' not in calls[-1]


def test_holdout_outcomes_cannot_change_fits_cv_scores_or_winner(tmp_path,comparison_data):
    ids, expression, metadata, split, calls, fits=comparison_data
    results=[]
    for attempt in range(2):
        if attempt:
            labels=pd.read_csv(metadata,index_col=0)
            labels.loc[ids[48:],'label']=labels.loc[ids[48:],'label'].map({'a':'b','b':'a'})
            labels.to_csv(metadata)
        results.append(api.run(expression,meta_path=metadata,classification_column='label',
            output_dir=tmp_path/f'holdout_{attempt}',random_state=11,cv_folds=2,tuning_inner_cv=2,
            foundation=dict(enabled=True,model='bulkformer',classifiers=['svm','lr'],
                split_path=split,lr_c_values=[1.0])))
    a,b=[result.out_dir for result in results]
    pd.testing.assert_frame_equal(pd.read_csv(a/'leaderboard.csv'),pd.read_csv(b/'leaderboard.csv'))
    assert json.loads((a/'selection.json').read_text())==json.loads((b/'selection.json').read_text())
    first,second=[pd.read_csv(path/'holdout_predictions.csv') for path in [a,b]]
    np.testing.assert_array_equal(first.y_pred,second.y_pred)
    np.testing.assert_allclose(first[['probability_0','probability_1']],second[['probability_0','probability_1']])
    assert np.all(first.y_true!=second.y_true)
    assert len(calls)==2 and all(part<=set(ids[:48]) for part in fits)


def test_real_native_tree_and_stacking_suite_in_shared_comparison(tmp_path,monkeypatch):
    """No native-training stub: three-tree learners and no search keep this tiny."""
    import omiclass.train.trainer as trainer
    original=trainer.get_builder
    def small(key,**kwargs):
        if key=='stacked_ensemble':return original(key,**kwargs)
        params=dict(kwargs.pop('override_params',{}) or {},n_estimators=3,max_depth=2)
        if 'light_gbm' in key:params.update(verbosity=-1,min_child_samples=2,num_leaves=4)
        return original(key,override_params=params,**kwargs)
    monkeypatch.setattr(trainer,'get_builder',small)
    ids=[f's{i}' for i in range(64)]
    values=pd.DataFrame(np.random.RandomState(7).uniform(1,10,(6,64)),index=[f'g{i}' for i in range(6)],columns=ids)
    expression,metadata,split=[tmp_path/name for name in ['counts.csv','meta.csv','split.json']]
    values.to_csv(expression)
    pd.DataFrame({'label':['a','b']*32},index=ids).to_csv(metadata)
    split.write_text(json.dumps(dict(train_index=ids[:48],test_index=ids[48:])))
    seen=[]
    def extract(counts,cfg,out,metadata=None):
        out.mkdir(parents=True);seen.append(1)
        return extraction.Extraction(counts.T.astype(float),{},None)
    monkeypatch.setattr(extraction,'extract',extract)
    result=api.run(expression,meta_path=metadata,classification_column='label',
        output_dir=tmp_path/'native',random_state=11,cv_folds=2,tuning_inner_cv=2,bayesian_iterations=0,
        foundation=dict(enabled=True,model='bulkformer',classifiers=['mlp','mlp_smote','svm','lr','omiclass'],
            split_path=split,article_head=dict(epochs=1),lr_c_values=[1.0]))
    assert seen==[1]
    board=pd.read_csv(result.out_dir/'leaderboard.csv')
    native={'omiclass__'+key for key in ['plain_random_forest','light_gbm','xgboost','stacked_ensemble']}
    assert native<=set(board.model) and board.status.eq('ok').all()
    assert len(board)==8 and np.isfinite(board.selection_score_mean).all()
    cv=joblib.load(result.out_dir/'cv_results.joblib')
    stack=cv['omiclass__stacked_ensemble']
    assert len(stack.per_fold_metrics)==2
    for fold in stack.per_fold_metrics:
        assert set(fold['train_ids']).isdisjoint(fold['validation_ids'])
        assert set(fold['train_ids']+fold['validation_ids'])==set(ids[:48])
        assert fold['stacking_oof_scopes']  # original native nested-CV evidence retained
    tests=pd.read_csv(result.out_dir/'test_metrics.csv')
    assert set(tests.model)==set(board.model) and tests.n_test.eq(16).all()
    assert (result.out_dir/'final_report.html').is_file()


@pytest.mark.parametrize('version',[1,2])
def test_historical_bert_bundle_without_new_schema_fields_replays_key_zero(tmp_path,comparison_data,version):
    import hashlib
    import yaml
    from omiclass.config.schema import Config
    ids,expression,metadata,split,calls,_=comparison_data
    trained=api.run(expression,meta_path=metadata,classification_column='label',
        output_dir=tmp_path/'old',random_state=11,
        foundation=dict(enabled=True,model='bulkrnabert',classifier='svm',input_scale='tpm',split_path=split))
    out=trained.out_dir
    config_path=out/'config.freeze.yaml'
    saved=yaml.safe_load(config_path.read_text())
    saved['foundation'].pop('seed_policy')
    saved['foundation'].pop('classifiers')
    saved['normalization'].pop('annotation_path')
    saved['cellfm'].pop('gene_mapping_path')
    config_path.write_text(yaml.safe_dump(saved))
    manifest=json.loads((out/'foundation.json').read_text())
    manifest['format_version']=version
    manifest['encoder_assets']=runner._resource_digests(Config(**saved),version)
    manifest['extraction_protocol'].update(encoder_seed=0,seed_source='fixed JAX PRNGKey(0)')
    manifest['artifact_sha256']['config.freeze.yaml']=hashlib.sha256(config_path.read_bytes()).hexdigest()
    (out/'foundation.json').write_text(json.dumps(manifest))
    values=pd.read_csv(expression,index_col=0).iloc[:,48:]
    values.columns=[name+'_external' for name in values.columns]
    external=tmp_path/'external.csv';values.to_csv(external)
    api.predict(external,out,tmp_path/'prediction')
    replay=calls[-1][0]
    assert replay.foundation.seed_policy is None and replay.runtime.random_state==11
    assert extraction.encoder_seed(replay)==(0,'fixed JAX PRNGKey(0)')
