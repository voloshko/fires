import numpy as np
from src.comp.event_retention import conservative_bound, pixel_survival


def test_unknown_event_cloud_loss_cannot_hide_in_another_surviving_fire():
    proof = np.array([[2, 2], [2, 2]], np.uint8)
    result = pixel_survival(proof, [[1, 1], [0, 0]])
    original = [dict(sample_id='a',event_key='MTBS:a',protected_pixels=0),
                dict(sample_id='u',event_key=None,protected_pixels=4)]
    audit = conservative_bound(original, {'a':dict(known_event_survives=True,protected_pixels=0,lost_protected_pixels=0),'u':result})
    assert not audit['accepted'] and audit['event_retention_lower_bound'] is None


def test_known_frame_residuals_are_protected_too():
    result = pixel_survival([[1, 1, 2]], [[1, 1, 0]])
    audit = conservative_bound([dict(sample_id='a',event_key='MTBS:a',protected_pixels=1)], {'a':result})
    assert not audit['unresolved_guaranteed']


def test_missing_window_does_not_shrink_original_denominator():
    original = [dict(sample_id=str(i),event_key='MTBS:'+str(i),protected_pixels=0) for i in range(10)]
    observed = {str(i):dict(known_event_survives=True,protected_pixels=0,lost_protected_pixels=0) for i in range(7)}
    audit = conservative_bound(original, observed)
    assert audit['original_known_events'] == 10 and audit['event_retention_lower_bound'] == .7
    assert audit['accepted']
    original.append(dict(sample_id='unknown',event_key=None,protected_pixels=1))
    assert not conservative_bound(original, observed)['accepted']


def test_shared_hls_mtbs_id_is_counted_once():
    original = [dict(sample_id='hls',event_key='MTBS:a',protected_pixels=0),
                dict(sample_id='mtbs',event_key='MTBS:a',protected_pixels=0),
                dict(sample_id='unknown',event_key=None,protected_pixels=2)]
    observed = {'hls':dict(known_event_survives=True,protected_pixels=0,lost_protected_pixels=0),
                'unknown':pixel_survival([[2,2]],[[1,1]])}
    audit=conservative_bound(original,observed)
    assert audit['original_known_events']==1 and audit['accepted']
