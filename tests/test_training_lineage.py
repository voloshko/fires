import pytest
from scripts.audit_training_lineage import unique_event, write_once


def test_spatial_match_needs_unique_dominant_official_event():
    assert unique_event({"MTBS-a": .99, "MTBS-b": .01}) == "MTBS-a"
    assert unique_event({"MTBS-a": .94}) is None
    assert unique_event({"MTBS-a": .99, "MTBS-b": .2}) is None
    assert unique_event({}) is None


def test_lineage_audit_is_append_once(tmp_path):
    path = tmp_path / "audit.json"; write_once(path, {"hash": "old"}); write_once(path, {"hash": "old"})
    with pytest.raises(ValueError): write_once(path, {"hash": "changed"})
