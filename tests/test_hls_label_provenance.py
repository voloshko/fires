import numpy as np
import pytest
from src.comp.hls_label_provenance import label_signature, prior_ignition, reuse_witness


def test_signature_requires_same_labels_grid_and_valid_domain():
    labels = np.array([[0, 1], [-1, 1]])
    valid = labels >= 0
    grid = dict(crs='EPSG:32611', affine=[30, 0, 1, 0, -30, 2])
    signature = label_signature(labels, valid, grid)
    assert signature == label_signature(labels.astype(np.int16), valid.copy(), grid)
    assert signature != label_signature(labels, ~valid, grid)
    assert signature != label_signature(labels, valid, {**grid, 'crs': 'EPSG:32612'})
    assert signature != label_signature(np.zeros_like(labels), valid, grid)
    with pytest.raises(ValueError):
        label_signature(labels, valid[:, :1], grid)


def test_prior_ignition_excludes_future_and_unknown_without_age_cutoff():
    assert prior_ignition('2017-01-01', '2020-08-09')
    assert prior_ignition('2020-08-09', '2020-08-09')
    assert not prior_ignition('2020-08-10', '2020-08-09')
    assert not prior_ignition(None, '2020-08-09')


def test_reuse_witness_does_not_assign_an_event_or_fire_date():
    result = reuse_witness(dict(sample_id='old', acquisition_day='2018-08-10', event_id='event'))
    assert result['existing_registry_event'] == 'event'
    assert 'event_id' not in result and 'fire_date' not in result
    assert 'not a physical burn date' in result['attribution']


def test_empty_and_filtered_candidates_keep_geographic_schema():
    import geopandas as gpd
    from shapely.geometry import Point
    from scripts.audit_hls_label_provenance import filter_prior_candidates
    frame = gpd.GeoDataFrame(dict(audit_day=[None, '2021-01-01', '2017-01-01'],
        event_id=['unknown', 'future', 'old']), geometry=[Point(0, 0)] * 3, crs='EPSG:4326')
    selected = filter_prior_candidates(frame, '2020-08-09')
    assert selected.event_id.tolist() == ['old']
    assert selected.to_crs('EPSG:32631').crs.to_epsg() == 32631
    empty = filter_prior_candidates(frame.iloc[:0], '2020-08-09')
    assert empty.empty and isinstance(empty, gpd.GeoDataFrame)
    assert empty.to_crs('EPSG:32631').crs.to_epsg() == 32631
