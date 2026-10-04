import pytest
import torch

from src.comp.equal_cost_additions import LOSS_ID
from src.comp.five_seed_additions import FIVE_BASE_SHAS, PARENT_RECIPE, contribution, member_paths, validate_member
from src.comp.recall_expansion import expand_probability


def bundle(seed, variant='pair'):
    return dict(spec_id='SPEC-114' if seed <= 3 else 'SPEC-118', loss_id=LOSS_ID,
        seed=seed, variant=variant, rank=8, recipe_id=PARENT_RECIPE if seed <= 3 else 'new',
        base_sha256=FIVE_BASE_SHAS[seed-1], smoke=False)


def test_parent_and_new_members_have_distinct_bound_identities():
    for seed in range(1, 6):
        value=bundle(seed);validate_member(value,seed,'pair','new')
        for key,bad in [('spec_id','wrong'),('loss_id','wrong'),('seed',0),
                        ('base_sha256','wrong'),('recipe_id','wrong'),('variant','post'),('smoke',True)]:
            with pytest.raises(ValueError):validate_member({**value,key:bad},seed,'pair','new')
    validate_member({**bundle(4),'smoke':True},4,'pair','new',allow_smoke=True)


def test_paths_reuse_only_original_first_three_and_new_last_two(tmp_path):
    paths=member_paths(tmp_path,'post')
    assert len(paths)==5 and all(str(p).startswith('research/equal-cost-additions-v1/')for p in paths[:3])
    assert paths[3]==tmp_path/'post-s4/adapter.pt' and paths[4]==tmp_path/'post-s5/adapter.pt'


def test_streaming_mean_is_uniform_and_monotonic():
    values=torch.tensor([[0.,1.],[.1,.2],[.3,.4],[.5,.6],[.7,.8]])
    mean=sum(contribution(v,5)for v in values)
    assert torch.allclose(mean,values.mean(0))
    reference=torch.tensor([0.,1.]);actual=expand_probability(reference,mean)
    assert (actual>=reference).all() and actual[1]==1 and actual[0]>0
    with pytest.raises(ValueError):contribution(values[0],3)
    with pytest.raises(ValueError):contribution(torch.tensor([float('nan')]),5)
