"""Fit-scope and serialization contracts for the distinct manuscript protocols."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from omiclass.config.loaders import load_config
from omiclass.features.bulkformer_reduction import EmbeddingRepresentation
from omiclass.foundation import heads
from omiclass.foundation.io import write_json


def configuration(tmp_path, classifier='lr', model='bulkformer', representation='supervised_pca', **extra):
    return load_config(dict(meta_path=tmp_path/'meta.csv',classification_column='label',
        foundation=dict(enabled=True,model=model,classifier=classifier,representation=representation,
                        lr_c_values=[1.0],bert_head=dict(max_epochs=2,patience=1),**extra),
        cv_folds=2,tuning_inner_cv=2,
        **{f'{"cellfm" if model.startswith("cellfm") else model}.reduction.supervised_dimensions':3,
           f'{"cellfm" if model.startswith("cellfm") else model}.reduction.pca_components':2}))


def vectors():
    X=pd.DataFrame(np.random.RandomState(3).normal(size=(30,6)),index=[f's{i}' for i in range(30)],columns=[f'f{i}' for i in range(6)])
    return X,pd.Series([i%2 for i in range(30)],index=X.index)


def test_lr_every_transform_is_training_fold_local(tmp_path,monkeypatch):
    X,y=vectors();seen=[];original=EmbeddingRepresentation.fit
    def record(self,values,labels=None):
        seen.append(set(values.index));return original(self,values,labels)
    monkeypatch.setattr(EmbeddingRepresentation,'fit',record)
    cfg=configuration(tmp_path);out=tmp_path/'lr';out.mkdir()
    models,board=heads.fit_lr(cfg,X.iloc[:24],y.iloc[:24],out)
    assert seen and all(s<=set(X.index[:24]) for s in seen)
    assert any(len(s)<12 for s in seen)  # inner fits within outer training
    assert seen[-1]==set(X.index[:24])   # final training-only refit
    assert not any(s&set(X.index[24:]) for s in seen)
    assert len(pd.read_csv(out/'oof_predictions.csv'))==24
    assert np.isfinite(models['logistic_regression'].predict_proba(X.iloc[24:])).all()


def test_bert_reduction_and_smote_exclude_inner_validation(tmp_path,monkeypatch):
    from imblearn.over_sampling import SMOTE
    X,y=vectors();y.iloc[:24]=[0]*16+[1]*8;seen=[]
    original=SMOTE.fit_resample
    def record(self,values,labels):
        seen.append(np.asarray(values).copy());return original(self,values,labels)
    monkeypatch.setattr(SMOTE,'fit_resample',record)
    cfg=configuration(tmp_path,'mlp_smote','bulkrnabert')
    out=tmp_path/'bert';out.mkdir()
    models,_=heads.fit_bert(cfg,X,y,X.index[:24].tolist(),['a','b'],out)
    scope=json.loads((out/'fit_scope.json').read_text())
    fit=set(scope['representation_fit_ids']);val=set(scope['inner_validation_ids'])
    assert fit.isdisjoint(val) and fit|val==set(X.index[:24])
    assert seen and seen[0].shape[0]==len(fit)
    transform=models['bert_mlp'].transform
    assert transform.n_features_in_==6
    info=json.loads((out/'training.json').read_text())
    assert set(info['inner_train_ids'])==fit
    assert not scope['refit_on_all_training']


def test_fixed_svm_keeps_decision_predictions(tmp_path):
    X,y=vectors();cfg=configuration(tmp_path,'svm');out=tmp_path/'fit'
    models,board=heads.fit(cfg,X,y,X.index[:24].tolist(),X.index[24:].tolist(),['a','b'],None,out)
    model=models['svm']
    assert model['svm'].C==1 and model['svm'].kernel=='rbf'
    assert board.f1_macro_mean.isna().all()
    assert set(json.loads((out/'fit_scope.json').read_text())['representation_fit_ids'])==set(X.index[:24])


def test_native_preprocessing_is_persisted_and_training_only(tmp_path,monkeypatch):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from omiclass.train.trainer import Trainer,TrainSummary
    X,y=vectors();X=(X.abs()*100).round();X['constant']=1;seen=[]
    def train(self,values,labels):
        model=self._with_bulkformer_reduction(lambda:RandomForestClassifier(n_estimators=2,random_state=0))().fit(values,labels)
        seen.append(model.prepare(values).copy())
        path=self.artifacts_dir/'rf.joblib';joblib.dump(model,path)
        return TrainSummary(pd.DataFrame([dict(model='rf',status='ok',f1_macro_mean=.5)]),{'rf':path})
    monkeypatch.setattr(Trainer,'run',train)
    cfg=configuration(tmp_path,'omiclass',representation='full');out=tmp_path/'native';out.mkdir()
    models,_=heads.fit_native(cfg,X,y,X.index[:24].tolist(),out)
    wrapper=models['rf'];assert not wrapper.did_log
    prep=wrapper.model.preprocessor_
    assert prep.did_log_ and 'constant' not in prep.selected_features_
    np.testing.assert_allclose(seen[0],np.log1p(X.iloc[:24][prep.selected_features_]))
    np.testing.assert_allclose(wrapper.model.prepare(X.iloc[24:]),np.log1p(X.iloc[24:][prep.selected_features_]))


def test_token_reader_rejects_path_escape(tmp_path):
    from omiclass.foundation.io import token_record
    with pytest.raises(ValueError,match='escapes'):
        with token_record(tmp_path,{'file':'../elsewhere.npz'}):pass


def test_output_scope_and_protocol_override_guards(tmp_path,monkeypatch):
    from omiclass.foundation.runner import _new_output
    cfg=configuration(tmp_path,'svm');cfg.foundation.mode='predict';cfg.foundation.run_dir=tmp_path/'run';cfg.project.out_dir=tmp_path/'run'/'prediction'
    with pytest.raises(ValueError,match='separate'):_new_output(cfg,True)


def test_article_architecture_and_bert_default_remain_distinct():
    import torch
    from omiclass.foundation.article_head import make_head
    from omiclass.foundation.bert_head import make_mlp,MLPConfig
    a=make_head(6,4,[64],8,.2,2);b=make_mlp(2,6)
    assert isinstance(a.norm,torch.nn.LayerNorm)
    assert a.projection.out_features==8
    assert isinstance(b[1],torch.nn.SELU)
    assert MLPConfig().metadata()['refit_on_all_training'] is False
    assert a(torch.ones(2,4,6),torch.ones(2,4,dtype=torch.bool)).shape==(2,2)


@pytest.mark.parametrize('majority,minority,expected',[
    (12,12,{'plain_random_forest','light_gbm','xgboost','stacked_ensemble'}),
    (24,12,{'plain_random_forest','weighted_random_forest','light_gbm','weighted_light_gbm','xgboost','weighted_xgboost','stacked_ensemble'}),
    (48,8,{'plain_random_forest','weighted_random_forest','smote_random_forest','adasyn_random_forest','svm_smote_random_forest','bl_smote_random_forest','light_gbm','weighted_light_gbm','xgboost','weighted_xgboost','stacked_ensemble'})])
def test_native_adapter_preserves_original_imbalance_dispatch(tmp_path,monkeypatch,majority,minority,expected):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from omiclass.train.trainer import Trainer
    calls=[]
    def fit(self,key,X,y,filename_for_artifact,**kwargs):
        calls.append(key);model=self._with_bulkformer_reduction(lambda:RandomForestClassifier(n_estimators=2,random_state=80,n_jobs=1))()
        model.fit(X,y);self.final_models[key]=model
        path=self.artifacts_dir/f'{key}.joblib';joblib.dump(model,path)
        return dict(model=key,f1_macro_mean=.5),path
    for name in ['run_rf_variant','run_boosting','run_stacked']:monkeypatch.setattr(Trainer,name,fit)
    n=majority+minority;X=pd.DataFrame(np.random.RandomState(5).normal(size=(n,6)),index=[f's{i}' for i in range(n)],columns=[f'f{i}' for i in range(6)])
    y=pd.Series([0]*majority+[1]*minority,index=X.index);cfg=configuration(tmp_path,'omiclass',representation='pca')
    out=tmp_path/'native';out.mkdir();models,_=heads.fit_native(cfg,X,y,X.index.tolist(),out)
    assert set(calls)==expected==set(models)


@pytest.mark.parametrize('key',['plain_random_forest','weighted_random_forest','smote_random_forest',
    'adasyn_random_forest','svm_smote_random_forest','bl_smote_random_forest'])
def test_real_rf_and_sampling_pipelines_fit_predict_and_reload(tmp_path,key):
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict,StratifiedKFold
    from omiclass.models.registry import get_builder
    cfg=configuration(tmp_path,'omiclass',representation='supervised_pca')
    rng=np.random.RandomState(9);X=pd.DataFrame(rng.normal(size=(64,6)),columns=[f'f{i}' for i in range(6)])
    y=pd.Series([0]*40+[1]*24)
    prefix='' if key in {'plain_random_forest','weighted_random_forest'} else 'rf__'
    classifier=get_builder(key,random_state=80,n_jobs=1,override_params={prefix+'n_estimators':2,prefix+'max_depth':2})()
    model=Pipeline([('representation',heads.representation(cfg)),('classifier',classifier)])
    p=cross_val_predict(model,X,y,cv=StratifiedKFold(2,shuffle=True,random_state=80),method='predict_proba')
    assert p.shape==(64,2) and np.isfinite(p).all()
    model.fit(X,y);path=tmp_path/'model.joblib';joblib.dump(model,path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(X),model.predict_proba(X))
