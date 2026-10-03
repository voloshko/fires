import pytest
from src.comp.russian_transfer import hard_background_weights, transfer_acceptance


def test_mining_is_fit_only_and_never_removes_positive_samples():
    records=[dict(sample_id=str(i),partition='fit') for i in range(8)]
    weights=hard_background_weights(records,{str(i):i for i in range(8)})
    assert sum(v==2 for v in weights.values())==2 and min(weights.values())==1
    assert weights['7']==weights['6']==2
    records[0]['partition']='evaluation'
    with pytest.raises(ValueError):hard_background_weights(records,{str(i):i for i in range(8)})


def test_transfer_does_not_claim_competitor_win_from_point_only():
    gain=dict(delta=.03,ci95=[.01,.05]);inconclusive=dict(delta=.04,ci95=[-.01,.08])
    report=transfer_acceptance(gain,{'ems':dict(delta=-.005),'floga':dict(delta=.01)},inconclusive)
    assert report['accepted'] and not report['competitor_superiority']
    assert not transfer_acceptance(gain,{'floga':dict(delta=-.02)},gain)['accepted']
