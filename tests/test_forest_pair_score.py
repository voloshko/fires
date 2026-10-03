import numpy as np
import pytest
from scripts.score_forest_pair import acceptance_report, binary_counts


def test_counts_ignore_unknown_and_include_false_background():
    truth = np.array([True, True, False, False])
    pred = np.array([True, False, True, True])
    assert binary_counts(truth, pred, [True, True, True, False]) == [1, 1, 1]
    with pytest.raises(ValueError): binary_counts(truth, pred[:2], np.ones(4))


def test_acceptance_requires_both_controls_and_each_dataset():
    records = [{"group_id": f"{d}:{i}", "dataset": d} for d in ("ems", "floga") for i in range(3)]
    good, weak = [[90, 5, 5]] * 6, [[60, 20, 20]] * 6
    assert acceptance_report(records, good, weak, weak)["accepted"]
    assert not acceptance_report(records, good, weak, good)["accepted"]
    assert not acceptance_report(records, weak[:3] + good[3:], good, weak)["accepted"]


def test_lost_events_aggregates_multiple_aois():
    records = [{"group_id": f"{d}:{i}", "dataset": d} for d in ("ems", "floga") for i in range(3)]
    records.append(dict(records[0]))
    candidate = [[90, 5, 5]] * 6 + [[0, 0, 10]]
    baseline = [[60, 20, 20]] * 7
    assert acceptance_report(records, candidate, baseline, baseline)["lost_events"]["pair"] == 0
