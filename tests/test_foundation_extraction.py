"""Mocked forwards and preprocessing guards, without model/vendor downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from omiclass.config.loaders import load_config
from omiclass.foundation import extraction
from omiclass.foundation.correction_compat import restore_bulkrnabert_zero_genes,clip_bulkrnabert_corrected


def cfg(tmp_path,model='bulkformer',**settings):
    return load_config(dict(meta_path=tmp_path/'meta.csv',classification_column='label',
        foundation=dict(enabled=True,model=model,classifier='mlp',**settings)))


def test_observer_retains_tokens_masks_and_genes_over_views(tmp_path):
    observer=extraction.TokenObserver(tmp_path/'tokens','cellfm',2)
    for view in range(2):
        tokens=np.arange(24,dtype=np.float32).reshape(2,3,4)+view
        masks=np.array([[1,1,0],[1,0,0]],dtype=bool)
        observer(tokens,masks,np.array([[1,2,0],[1,0,0]]),['a','b'],view)
    observer.finish(['a','b'])
    manifest=json.loads((tmp_path/'tokens/manifest.json').read_text())
    assert len(manifest['records'])==4
    with np.load(tmp_path/'tokens'/manifest['records'][0]['file']) as z:
        assert set(z.files) == {'tokens', 'mask', 'gene_ids'}
        np.testing.assert_array_equal(z['tokens'],np.arange(12,dtype=np.float32).reshape(3,4))
        np.testing.assert_array_equal(z['mask'],[1,1,0])
        np.testing.assert_array_equal(z['gene_ids'],[1,2,0])
    with pytest.raises(ValueError,match='Duplicated'):
        observer(tokens,masks,np.array([[1,2,0],[1,0,0]]),['a','b'],0)


@pytest.mark.parametrize('model',['bulkformer','cellfm_1view','cellfm_20view','bulkrnabert'])
def test_backend_selection_and_correction_order(tmp_path,monkeypatch,model):
    config=load_config(dict(meta_path=tmp_path/'meta.csv',classification_column='label',batch_column='batch',
        foundation=dict(enabled=True,model=model,classifier='svm',correction='limma',allow_transductive_correction=True)))
    counts=pd.DataFrame([[2.,3.],[4.,5.]],index=['g1','g2'],columns=['a','b']);events=[]
    def continuous(values,c):events.append('normalize');return values+10
    def correct(values,c,out,meta):events.append('correct');return values+100
    monkeypatch.setattr(extraction,'continuous_input',continuous)
    monkeypatch.setattr(extraction,'correct_input',correct)
    if model=='bulkrnabert':
        def bert(values,c,out,tokens):
            events.append('extract');np.testing.assert_array_equal(values,counts+110)
            return extraction.Extraction(values.T,{})
        monkeypatch.setattr(extraction,'extract_bert',bert)
    elif model=='bulkformer':
        from omiclass.features import bulkformer
        class Encoder:
            def __init__(self,c,**kw):assert kw['continuous']
            def extract(self,values):
                events.append('extract');np.testing.assert_array_equal(values,counts+110)
                return SimpleNamespace(features=values,metadata={})
        monkeypatch.setattr(bulkformer,'BulkFormerFeatureExtractor',Encoder)
    else:
        from omiclass.foundation import cellfm_corrected
        def encode(raw,values,c,out,observer=None):
            events.append('extract');np.testing.assert_array_equal(raw,counts);np.testing.assert_array_equal(values,counts+110)
            return values.T,{}
        monkeypatch.setattr(cellfm_corrected,'extract_corrected',encode)
    result=extraction.extract(counts,config,tmp_path/'out',pd.DataFrame(index=counts.columns))
    assert events==['normalize','correct','extract']
    assert result.features.index.tolist()==['a','b']


def test_bert_zero_restore_and_historical_clipping(tmp_path):
    before=pd.DataFrame([[0,0],[1,2]],index=['zero','present'],columns=['a','b'])
    corrected=pd.DataFrame([[-1,2]],index=['present'],columns=['a','b'])
    restored,genes=restore_bulkrnabert_zero_genes(before,corrected)
    assert genes==['zero']
    clipped=clip_bulkrnabert_corrected(restored,tmp_path)
    assert clipped.loc['present','a']==0
    info=json.loads((tmp_path/'clipping.json').read_text());assert info['clipped_entries']==1
    before.loc['zero','a']=1
    with pytest.raises(ValueError,match='Cannot restore'):
        restore_bulkrnabert_zero_genes(before,corrected)


def test_token_worker_isolated_and_explicit_log_scale(tmp_path,monkeypatch):
    config=cfg(tmp_path,'bulkrnabert',representation='tokens')
    frame=pd.DataFrame([[1,2],[3,4]],index=['g1','g2'],columns=['a','b']);seen=[]
    out=tmp_path/'out';out.mkdir()
    def run(cmd,**kw):
        seen.append((cmd,kw))
        np.savez(out/'worker-output.npz',embeddings=np.ones((2,256),np.float32),sample_ids=np.array(['a','b']),metadata='{}')
    monkeypatch.setattr(extraction.subprocess,'run',run)
    result=extraction.extract_bert(frame,config,out,True)
    cmd,options=seen[0]
    assert cmd[cmd.index('--input-scale')+1]=='log10_tpm'
    assert cmd[cmd.index('--token-dir')+1]==str(out/'tokens')
    assert 'PYTHONPATH' not in options['env']
    assert options['env']['JAX_COMPILATION_CACHE_DIR'].startswith(str(out))
    assert result.features.shape==(2,256)


def test_bulkformer_instance_observer_keeps_original_forward_and_pooling(tmp_path,monkeypatch):
    import torch
    from omiclass.features import bulkformer
    counts=pd.DataFrame([[2.,3.],[4.,5.]],index=['g1','g2'],columns=['a','b'])
    c=cfg(tmp_path)
    prepared=bulkformer.align_to_bulkformer_vocabulary(counts,['g1','g2'])
    class Encoder:
        def __call__(self,x,**kw):return torch.stack([x,x+1,x+2],dim=-1)
    monkeypatch.setattr(bulkformer,'_validate_bulkformer_resources',lambda c:None)
    monkeypatch.setattr(bulkformer,'load_bulkformer_gene_list',lambda p:['g1','g2'])
    monkeypatch.setattr(bulkformer,'_load_bulkformer_model',lambda c,d:(Encoder(),{'dim':3,'gene_length':2}))
    monkeypatch.setattr(torch,'load',lambda *a,**k:np.array([0,1]))
    c.bulkformer.device='cpu';calls=[]
    a=bulkformer.BulkFormerFeatureExtractor(c.bulkformer,continuous=True).extract(counts)
    b=bulkformer.BulkFormerFeatureExtractor(c.bulkformer,continuous=True,
        observer=lambda *args:calls.append(args)).extract(counts)
    np.testing.assert_array_equal(a.features,b.features)
    assert len(calls)==1 and list(calls[0][3])==['a','b']
    assert calls[0][0].shape==(2,2,3)


def test_cellfm_instance_observer_is_restored(tmp_path,monkeypatch):
    import torch
    from omiclass.features import cellfm
    c=cfg(tmp_path,'cellfm_20view');c.cellfm.device='cpu'
    counts=pd.DataFrame([[2.,3.],[4.,5.]],index=['g1','g2'],columns=['a','b'])
    prepared=cellfm.CellFMPreparedInput(counts=np.array([[2,4],[3,5]]),gene_ids=np.array([1,2]),
        sample_ids=['a','b'],mapped_genes=['g1','g2'],unmapped_genes=[],raw_library_sizes=np.array([6,8]))
    class Model:
        def __init__(self,**kwargs):self.net=SimpleNamespace(encode=self.encode)
        def encode(self,*args):return torch.ones((2,5,1536)),None
        def to(self,d):return self
        def load_model(self,**kw):pass
        def eval(self):return self
        def requires_grad_(self,v):return self
    models=[]
    def create(**kwargs):m=Model();models.append(m);return m
    utils=SimpleNamespace(Config_80M=lambda:SimpleNamespace(nonz_len=2048,enc_dims=1536),Prepare=lambda **kw:object())
    monkeypatch.setattr(cellfm,'_validate_resources',lambda c:None)
    monkeypatch.setattr(cellfm,'load_cellfm_vocabulary',lambda p:['g1','g2'])
    monkeypatch.setattr(cellfm,'prepare_cellfm_input',lambda *a:prepared)
    monkeypatch.setattr(cellfm,'_import_cellfm',lambda c:(utils,SimpleNamespace(Cell_FM=create)))
    def drive(**kw):
        for i in range(20):kw['model'].net.encode(torch.ones((2,2,2)),torch.tensor([[1,2],[1,2]]),torch.ones(2,2),torch.ones((2,2),dtype=torch.bool))
        return np.ones((2,1536),np.float32),[[2]*20]*2,'stable','mean'
    monkeypatch.setattr(cellfm,'_extract_configured_views',drive)
    observer=extraction.TokenObserver(tmp_path/'tokens','cellfm',20)
    result=cellfm.CellFMFeatureExtractor(c.cellfm,observer=observer).extract(counts)
    observer.finish(['a','b'])
    assert result.features.shape==(1536,2)
    assert len(observer.records)==40
    assert models[0].net.encode==models[0].encode
    with np.load(tmp_path/'tokens'/observer.records[0]['file']) as z:
        assert np.issubdtype(z['gene_ids'].dtype,np.integer)
