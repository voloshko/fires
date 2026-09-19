"""Research acceptance invariants; never access real competition test data."""
import numpy as np
import pytest
from src.comp.hypothesis_lab import (nested_folds,choose_background,binary_counts,binary_scores,paired_interval,confusion,bs_scores,bs_inputs,write_json)


def test_nested_folds_each_chip_evaluated_once_never_fit_or_cal():
    ids=[f'chip-{i:03}' for i in range(100)]; evaluated=[]
    for fit,cal,test in nested_folds(ids):
        assert not(set(fit)&set(cal) or set(fit)&set(test) or set(cal)&set(test))
        assert set(fit+cal+test)==set(ids); evaluated+=test
    assert sorted(evaluated)==ids
    assert list(nested_folds(ids))==list(nested_folds(ids[::-1]))


def test_hard_mining_keeps_budget_unique_and_never_takes_positive():
    negatives=np.arange(100,3100); scores=np.arange(4000)
    selected=choose_background(negatives,np.random.default_rng(1),scores)
    assert len(selected)==len(set(selected))==1500
    assert set(selected)<=set(negatives)
    assert set(range(2350,3100))<=set(selected)
    assert np.array_equal(choose_background(negatives[:4],np.random.default_rng(1),scores),negatives[:4])


def test_micro_f1_uses_counts_not_mean_of_chip_scores():
    counts=np.array([[1,0,0],[9,10,0]])
    assert binary_scores(counts)['f1']==pytest.approx(2/3)
    assert binary_scores([0,0,0])['f1']==0
    assert np.array_equal(binary_counts([1,0,1],[1,1,0]),[1,1,1])
    assert paired_interval(counts,counts,n=10)==[0,0]


def test_bs_confusion_micro_and_missing_classes():
    cm=confusion(np.array([0,1,2,3]),np.array([1,1,2,0]))
    s=bs_scores(cm)
    assert s['iou_burn']==.5
    assert s['miou_sev']==.5
    assert s['weighted']==pytest.approx(.5)
    assert bs_scores(np.zeros((4,4),int))['weighted'] is None


def test_raw_bands_exclude_scl_and_preserve_dates():
    from src.comp.chips import BsChip
    pre=np.full((10,4,4),1000,np.uint16); post=np.full((10,4,4),2000,np.uint16)
    pre[9]=4; post[9]=5
    c=BsChip('x',pre,post,np.zeros((3,4,4)),np.zeros((2,4,4)),None)
    raw=bs_inputs(c,'raw')
    assert raw.shape==(29,4,4)
    assert np.allclose(raw[:9],.1) and np.allclose(raw[9:18],.2)
    assert np.array_equal(raw[18:],bs_inputs(c,'optical'))


def test_results_are_append_only(tmp_path):
    p=tmp_path/'result.json'; write_json(p,{'ok':True})
    with pytest.raises(FileExistsError): write_json(p,{'ok':False})


@pytest.mark.parametrize('variant',['optical','raw','siam'])
def test_model_reload_preserves_spatial_prediction(variant,tmp_path):
    import torch
    from src.comp.hypothesis_models import make_model
    torch.set_num_threads(1)
    model=make_model(variant,width=2,depth=3).eval()
    x=torch.randn(2,11 if variant=='optical' else 29,32,32)
    with torch.no_grad(): y=model(x)
    p=tmp_path/'weights.pt'; torch.save(model.state_dict(),p)
    other=make_model(variant,width=2,depth=3).eval(); other.load_state_dict(torch.load(p,weights_only=True))
    with torch.no_grad(): actual=other(x)
    assert y.shape==(2,4,32,32)
    assert torch.equal(y,actual)


def test_manifest_excludes_argparse_callback(tmp_path):
    import argparse,json
    from scripts.hypothesis_lab import manifest
    m=manifest(argparse.Namespace(task='af-hard',func=lambda x:x))
    assert m['config']=={'task':'af-hard'}
    json.dumps(m)


def test_external_padding_never_contributes_to_loss_or_gradient():
    import torch
    from src.comp.hypothesis_models import masked_binary_loss
    y=torch.tensor([[[0,1],[255,255]]]); a=torch.randn(1,2,2,2,requires_grad=True)
    loss=masked_binary_loss(a,y); loss.backward()
    assert torch.equal(a.grad[:,:,1,:],torch.zeros_like(a.grad[:,:,1,:]))
    b=a.detach().clone(); b[:,:,1,:]=100*torch.randn_like(b[:,:,1,:])
    assert torch.allclose(loss,masked_binary_loss(b,y))
    with pytest.raises(ValueError): masked_binary_loss(a,torch.full_like(y,255))


def test_bs_confirmation_excludes_selection_and_keeps_events_whole():
    import pandas as pd
    from src.comp.hypothesis_lab import bs_confirmation_split
    dev=[f'd{i}' for i in range(20)]; selection=['s0','s1']
    m=pd.DataFrame({'chip_id':dev+selection,'fire_event_id':[f'e{i//2}' for i in range(20)]+['s0','s1']})
    mapping=m.set_index('chip_id').fire_event_id.to_dict(); seen=[]
    for fold in range(5):
        fit,test=bs_confirmation_split(m,dev,selection,fold)
        assert set(selection)<=set(fit) and not(set(selection)&set(test))
        assert not({mapping[c] for c in fit}&{mapping[c] for c in test})
        seen+=test
    assert sorted(seen)==sorted(dev)
    m.loc[20,'fire_event_id']='e0'
    with pytest.raises(ValueError): bs_confirmation_split(m,dev,selection,0)


def test_manifest_rejects_unfrozen_augmentation(monkeypatch):
    import argparse
    from scripts.hypothesis_lab import manifest
    monkeypatch.setenv('ROT90','1')
    with pytest.raises(ValueError,match='ROT90'): manifest(argparse.Namespace())
    monkeypatch.setenv('ROT90','0');monkeypatch.setenv('FEATURES','s1')
    with pytest.raises(ValueError,match='FEATURES'): manifest(argparse.Namespace())


def test_repository_scripts_package_wins():
    import scripts
    from pathlib import Path
    assert Path(scripts.__file__).resolve()==Path(__file__).resolve().parents[1]/'scripts/__init__.py'


def test_bs_scores_match_existing_metric_when_classes_absent():
    from src.comp.metric import score_bs_micro
    for truth,pred in [([0,1,1],[0,1,1]),([1,2,2],[1,1,2]),([0,0,0],[1,0,0])]:
        truth,pred=np.asarray(truth),np.asarray(pred)
        expected=score_bs_micro([truth],[pred]); actual=bs_scores(confusion(truth,pred))
        assert actual['iou_burn']==pytest.approx(expected['iou_burn'])
        assert actual['miou_sev']==pytest.approx(expected['miou_sev'])


def test_cached_postprocessing_matches_product_rules(monkeypatch):
    from src.comp.hypothesis_lab import bs_prediction
    from src.comp.chips import BsChip
    from src.comp import ensemble,unet
    rng=np.random.default_rng(5);pn=rng.random((4,4,4));pb=rng.random((4,4,4));pn/=pn.sum(2,keepdims=True);pb/=pb.sum(2,keepdims=True)
    pre=np.full((10,4,4),1000,np.uint16);pre[9]=4;post=pre.copy();post[9]=np.tile([4,8,9,10],(4,1))
    chip=BsChip('x',pre,post,np.zeros((3,4,4)),np.zeros((2,4,4)),None)
    class Boost:
        def predict_proba(self,x):return pb.reshape(-1,4)
    monkeypatch.setattr(unet,'probs',lambda *a,**k:pn)
    assert np.array_equal(bs_prediction(pn,pb,chip),ensemble.predict(object(),Boost(),chip))


def test_boost_cache_binding_rejects_wrong_split_and_changed_pixels(tmp_path):
    import json
    from src.comp.hypothesis_lab import verify_bs_probability_cache,digest
    source=tmp_path/'pixels';source.write_bytes(b'original')
    m={'fit':['a'],'evaluation':['b'],'files':{str(source):digest(source)}}
    (tmp_path/'data_manifest.json').write_text(json.dumps(m));np.save(tmp_path/'probabilities.npy',np.zeros((1,2,2,4)))
    hashes=verify_bs_probability_cache(tmp_path,['a'],['b'])
    assert str(tmp_path/'probabilities.npy') in hashes
    with pytest.raises(ValueError,match='split'):verify_bs_probability_cache(tmp_path,['b'],['a'])
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source'):verify_bs_probability_cache(tmp_path,['a'],['b'])


def test_sentinel_baseline_offset_is_metadata_driven_and_preserves_scl():
    from src.comp.hypothesis_lab import harmonize_s2_dn
    a=np.array([[[0,500,1500]],[[4,8,9]]],np.uint16)
    new=harmonize_s2_dn(a,['B12','SCL'],'05.10')
    assert np.array_equal(new,np.array([[[0,0,500]],[[4,8,9]]]))
    assert np.array_equal(harmonize_s2_dn(a,['B12','SCL'],'02.14'),a)
    assert a[0,0,2]==1500
    with pytest.raises(ValueError):harmonize_s2_dn(a,['B12','SCL'],None)


def test_candidate_keeps_bs_bytes_and_requires_all_af_rows(tmp_path):
    from scripts.build_af_hard_candidate import replace_af_rows
    base=tmp_path/'base.csv';base.write_bytes(b'chip_id,class_id,rle\nBS_x,1,"1 2"\r\nAF_x,1,\nBS_x,2,\n')
    out=tmp_path/'out.csv';r=replace_af_rows(base,{'AF_x':'3 1'},out)
    assert b'BS_x,1,"1 2"\r\n' in out.read_bytes()
    assert r['bs_bytes_preserved'] and r['changed_af_rows']==1
    with pytest.raises(ValueError,match='missing'):replace_af_rows(base,{'AF_missing':''},tmp_path/'bad.csv')
    assert not (tmp_path/'bad.csv').exists()


def test_sorted_threshold_counts_equal_direct_counts_at_ties():
    from scripts.verify_af_net_hypothesis import counts_for_thresholds
    t=np.array([1,0,1,0,1],bool);p=np.array([.5,.5,.1,1.,0.]);grid=np.array([0.,.1,.5,1.])
    actual=counts_for_thresholds(t,p,grid)
    expected=np.asarray([binary_counts(t,p>=cut) for cut in grid])
    assert np.array_equal(actual,expected)


def test_siamese_fusion_is_normalized_before_decoder():
    import torch
    from src.comp.hypothesis_models import make_model
    net=make_model('siam',width=2,depth=3)
    assert all(isinstance(layer[1],torch.nn.BatchNorm2d) for layer in net.fuse)
    x=torch.randn(2,29,32,32)*100
    y=net(x); assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
