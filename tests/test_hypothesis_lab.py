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
    assert bs_scores(np.zeros((4,4),int))['weighted']==0


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
