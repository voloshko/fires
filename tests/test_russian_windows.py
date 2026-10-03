import pytest
from scripts.prepare_russian_windows import interval_pair_valid, round_robin


def test_size_stratification_is_deterministic():
    records = [dict(sample_id=str(i), area_ha=area) for i, area in enumerate([2, 3, 40, 80, 500, 700])]
    result = round_robin(records, 3)
    assert result == round_robin(list(reversed(records)), 3)
    assert sorted(0 if r['area_ha'] < 25 else 1 if r['area_ha'] < 250 else 2 for r in result) == [0, 1, 2]


def test_interval_pair_does_not_invent_an_ignition_day():
    r = dict(ignition_interval=['2020-01-01T00:00:00Z','2020-05-16T00:00:00Z'],
             pre_at='2019-06-01T00:00:00Z',post_at='2020-06-01T00:00:00Z',
             mgrs='45ABC',pre_mgrs='45ABC',grid={'shape':[512,512]},pre_grid={'shape':[512,512]})
    interval_pair_valid(r)
    r['pre_at']='2020-03-01T00:00:00Z'
    with pytest.raises(ValueError): interval_pair_valid(r)
