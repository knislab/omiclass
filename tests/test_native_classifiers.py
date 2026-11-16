"""Synthetic native comparison and leakage contracts; no research experiments."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import api
from omiclass.config.loaders import load_config
from omiclass.config.schema import ClassificationCfg
from omiclass.train.preprocessing import TrainingPreprocessor


@pytest.fixture
def dataset(tmp_path):
    ids = [f's{i:02}' for i in range(64)]
    counts = pd.DataFrame(np.random.RandomState(27).randint(10,100,(10,64)),
        index=[f'g{i}' for i in range(10)], columns=ids)
    expr,meta,lengths,split = [tmp_path/name for name in ['counts.csv','meta.csv','lengths.csv','split.json']]
    counts.to_csv(expr)
    pd.DataFrame({'label':['a','b']*32,'batch':['A','B']*32,'age':np.arange(64)+20},index=ids).to_csv(meta)
    pd.DataFrame({'length':np.arange(1,11)*1000},index=counts.index).to_csv(lengths)
    split.write_text(json.dumps(dict(train_index=ids[:48],test_index=ids[48:])))
    return ids,expr,meta,lengths,split


def small_native(monkeypatch):
    import omiclass.train.trainer as trainer
    original = trainer.get_builder
    def small(key, **kwargs):
        if key == 'stacked_ensemble':return original(key, **kwargs)
        params=dict(kwargs.pop('override_params',{}) or {},n_estimators=3,max_depth=2)
        if 'light_gbm' in key:params.update(verbosity=-1,min_child_samples=2,num_leaves=4)
        return original(key,override_params=params,**kwargs)
    monkeypatch.setattr(trainer,'get_builder',small)


def options(tmp_path,dataset,classifiers,**kwargs):
    ids,expr,meta,lengths,split=dataset
    return dict(meta_path=meta,classification_column='label',classes_names=['a','b'],
        output_dir=tmp_path/'runs',raw_count=True,run_de=False,run_batch=False,
        gene_lengths_path=lengths,reuse_split=True,split_json=split,
        feature_selection_enabled=False,cv_folds=2,tuning_inner_cv=2,bayesian_iterations=0,
        random_state=61,n_jobs=1,
        classification=dict(classifiers=classifiers,mlp=dict(epochs=1),lr_c_values=[1.0]),**kwargs)


def test_native_api_all_heads_cv_selected_before_holdout_and_reports(tmp_path,dataset,monkeypatch):
    from omiclass.eval import holdout
    small_native(monkeypatch)
    ids,expr,meta,lengths,split=dataset
    fitted=[]
    original=TrainingPreprocessor.fit
    def fit(self,X,y=None):
        fitted.append(set(X.index));return original(self,X,y)
    monkeypatch.setattr(TrainingPreprocessor,'fit',fit)
    result=api.run(expr,**options(tmp_path,dataset,['omiclass','mlp','svm','lr']))
    out=result.out_dir
    board=pd.read_csv(out/'leaderboard.csv')
    assert set(board.classifier)=={'omiclass','mlp','svm','lr'} and len(board)==7
    assert {'plain_random_forest','xgboost','light_gbm','stacked_ensemble','mlp','svm','logistic_regression'}==set(board.model)
    assert board.status.eq('ok').all()
    winner=board.sort_values(['selection_score_mean','model'],ascending=[False,True]).iloc[0].model
    selection=json.loads((out/'selection.json').read_text())
    assert selection['selected_model']==winner and selection['holdout_used'] is False
    manifest=json.loads((out/'classification.json').read_text())
    assert manifest['status']=='complete' and manifest['selected_model']==winner
    assert fitted and all(part<=set(ids[:48]) for part in fitted) and any(len(part)<24 for part in fitted)
    tests=pd.read_csv(out/'test_metrics.csv')
    assert set(tests.model)==set(board.model) and tests.n_test.eq(16).all()
    assert np.isfinite(tests[['accuracy','f1_macro','precision_macro','recall_macro']]).all().all()
    assert set(pd.read_csv(out/'oof_predictions.csv').sample_id)==set(ids[:48])
    html=(out/'final_report.html').read_text()
    assert 'training_outer_cv' in html and winner in html and '<svg' in html
    for key in board.model:assert (out/f'reports/{key}.html').is_file()
    # Models replay without training/DE fitting or the temporary MLP files.
    monkeypatch.setattr(TrainingPreprocessor,'fit',lambda *a,**kw:pytest.fail('Prediction refit'))
    from omiclass.data.processor import DataProcessor
    cfg=load_config(dict(output_dir=out))
    values=DataProcessor(cfg).normalize_raw_counts(pd.read_csv(expr,index_col=0).astype('float32')).T
    for relative in manifest['models'].values():
        model=joblib.load(out/relative)
        assert np.isfinite(model.predict_proba(values.iloc[48:])).all()
    # Full-runs external evaluation retains selected-only files and adds all
    # saved candidates. External outcome changes cannot refit or choose a model.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'full_runs'))
    from _shared.native_comparison import evaluate_external
    external=values.iloc[48:].copy();external.index=[f'external_{i}' for i in range(len(external))]
    truth=pd.Series([0,1]*8,index=external.index,dtype=float);truth.iloc[-1]=np.nan
    evaluated=evaluate_external(out,external,truth,tmp_path/'external_all')
    assert set(evaluated['evaluated_models'])==set(board.model) and evaluated['fit'] is False
    assert evaluated['predicted']==16 and evaluated['scored']==15
    assert len(pd.read_csv(tmp_path/'external_all/all_model_predictions.csv'))==16*7
    assert set(pd.read_csv(tmp_path/'external_all/all_model_metrics.csv').model)==set(board.model)


@pytest.mark.parametrize('classifiers',[[],['svm','svm']])
def test_invalid_classifier_lists(classifiers):
    with pytest.raises(ValueError,match='nonempty and unique'):ClassificationCfg(classifiers=classifiers)


def test_default_configuration_serialization_and_explicit_single_choices(tmp_path):
    cfg=load_config(dict(meta_path=tmp_path/'meta.csv',classification_column='label'))
    assert 'classification' not in cfg.model_dump()
    assert cfg.classification.requested==['omiclass'] and not cfg.classification.active
    assert load_config(dict(meta_path=tmp_path/'meta.csv',classification_column='label',classifier='svm')).classification.requested==['svm']


def test_single_mlp_multiclass_probability_order_and_replay(tmp_path,dataset):
    ids,expr,meta,_,_=dataset
    frame=pd.read_csv(meta,index_col=0);frame['label']=['a','b','c']*21+['a'];frame.to_csv(meta)
    settings=options(tmp_path,dataset,['mlp'])
    settings.update(classes_names=['c','b','a'],classification=dict(classifier='mlp',mlp=dict(epochs=1)))
    result=api.run(expr,**settings)
    manifest=json.loads((result.out_dir/'classification.json').read_text())
    assert manifest['classifiers']==['mlp'] and set(manifest['models'])=={'mlp'}
    tests=pd.read_csv(result.out_dir/'test_metrics.csv')
    assert tests.n_test.tolist()==[16]
    model=joblib.load(result.out_dir/manifest['models']['mlp'])
    from omiclass.data.processor import DataProcessor
    cfg=load_config(dict(output_dir=result.out_dir))
    values=DataProcessor(cfg).normalize_raw_counts(pd.read_csv(expr,index_col=0)).T.iloc[48:]
    proba=model.predict_proba(values)
    assert proba.shape==(16,3) and np.allclose(proba.sum(axis=1),1)
    assert set(model.predict(values))<=set(range(3))


@pytest.mark.parametrize('method',['frozen_quantile','fsva_exact'])
def test_de_with_frozen_references_rejected_explicitly(tmp_path,method):
    with pytest.raises(Exception,match='DE with frozen references is unsupported'):
        load_config(dict(meta_path=tmp_path/'meta.csv',classification_column='label',
            classifiers=['svm','lr'],**{'batch_correction.algorithm':method}))


def test_de_scaling_and_calibration_fit_only_training_scopes_and_holdout_cannot_select(tmp_path,dataset,monkeypatch):
    from sklearn.preprocessing import StandardScaler
    from omiclass.de.limma_voom_runner import LimmaVoomRunner
    ids,expr,meta,lengths,split=dataset
    de_scopes,scale_scopes=[],[]
    original_scale=StandardScaler.fit
    def scale(self,values,*args,**kwargs):
        scale_scopes.append(set(values.index));return original_scale(self,values,*args,**kwargs)
    monkeypatch.setattr(StandardScaler,'fit',scale)
    def markers(self,raw_counts_df,meta_df,design_col,**kwargs):
        scope=set(raw_counts_df.columns)
        assert scope==set(meta_df.index) and scope<=set(ids[:48])
        assert kwargs.get('covariates')==['age']
        de_scopes.append(scope)
        labels=meta_df.loc[raw_counts_df.columns,design_col]
        score=(raw_counts_df.loc[:,labels.eq(0).to_numpy()].mean(axis=1)-
               raw_counts_df.loc[:,labels.eq(1).to_numpy()].mean(axis=1)).abs()
        return {'selected_genes':score.nlargest(3).index.tolist()}
    monkeypatch.setattr(LimmaVoomRunner,'run_limma_voom',markers)
    outputs=[]
    for attempt in range(2):
        settings=options(tmp_path,dataset,['mlp','svm','lr'])
        settings.update(run_de=True,output_dir=tmp_path/f'de_{attempt}',de_covariates=['age'])
        outputs.append(api.run(expr,**settings).out_dir)
        if not attempt:
            metadata=pd.read_csv(meta,index_col=0)
            metadata.loc[ids[48:],'label']=metadata.loc[ids[48:],'label'].map({'a':'b','b':'a'})
            metadata.to_csv(meta)
            counts=pd.read_csv(expr,index_col=0);counts.loc[:,ids[48:]]*=1000;counts.to_csv(expr)
    assert de_scopes and scale_scopes
    assert all(s<=set(ids[:48]) for s in de_scopes+scale_scopes)
    assert any(len(s)<24 for s in de_scopes)  # inner LR/calibration scopes
    assert json.loads((outputs[0]/'selection.json').read_text())==json.loads((outputs[1]/'selection.json').read_text())
    pd.testing.assert_frame_equal(pd.read_csv(outputs[0]/'leaderboard.csv'),pd.read_csv(outputs[1]/'leaderboard.csv'))
    for path in outputs:
        assert json.loads((path/'preprocessing/scoped.json').read_text())['de'] is True
        svm=joblib.load(path/'train_artifacts/svm.joblib')
        part=svm.calibrated_classifiers_[0].estimator
        assert part.preprocessor_.de_fit_ids_==ids[:48]
        assert len(part.preprocessor_.selected_features_)==3
