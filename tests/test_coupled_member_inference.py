import numpy as np
import pytest
import torch
from src.comp.coupled_member_inference import anchored_probability, member_delta, save_mask, load_mask


def test_covariance_and_exact_member_expansion():
    p=torch.tensor([0.,.2,.4,.6,1.]);q=1-p
    delta=sum(member_delta(a,b)for a,b in zip(p,q))
    actual=p.mean()+delta
    expected=(p+(1-p)*q).mean()
    old=p.mean()+(1-p.mean())*q.mean()
    covariance=(p*q).mean()-p.mean()*q.mean()
    assert torch.allclose(actual,expected) and torch.allclose(old-actual,covariance)
    assert actual > old


def test_anchor_parity_monotonicity_and_tiny_upper_drift():
    anchor=np.array([0.,.5,1.],np.float32);delta=np.array([.2,.1,0.],np.float32)
    result=anchored_probability(anchor,anchor,delta)
    assert np.all(result>=anchor) and result[2]==1
    with pytest.raises(ValueError,match='parity'):anchored_probability(anchor,anchor*.9,delta)
    with pytest.raises(ValueError):member_delta(torch.tensor([.2]),torch.tensor([.3]),3)


def test_lossless_compressed_masks_and_binding(tmp_path):
    rng=np.random.default_rng(9);mask=rng.random((2,64,64))>.95
    path=tmp_path/'mask.npz';signature='fixed'
    save_mask(path,mask,signature)
    assert np.array_equal(load_mask(path,signature,mask.shape),mask)
    with pytest.raises(ValueError):load_mask(path,'other',mask.shape)
