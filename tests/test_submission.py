"""SPEC-16: RLE и валидация submission.csv."""

import numpy as np
import pandas as pd
import pytest

from src.comp.rle import decode, encode
from src.comp.submission import rows_for_chip, validate, write


def test_encode_matches_hand_computed():
    mask = np.zeros((3, 3), bool)
    mask[0, 1] = mask[0, 2] = mask[2, 0] = True
    assert encode(mask) == "2 2 7 1"  # нумерация с единицы, построчно


def test_adjacent_runs_merge():
    mask = np.zeros((1, 6), bool)
    mask[0, 1:3] = True
    mask[0, 3:5] = True  # смежный отрезок
    assert encode(mask) == "2 4"  # одна серия, не две


def test_roundtrip_random_masks():
    rng = np.random.default_rng(7)
    for _ in range(120):
        shape = (rng.integers(1, 12), rng.integers(1, 12))
        mask = rng.random(shape) > rng.random()
        assert (decode(encode(mask), shape) == mask).all()


def test_empty_and_full():
    assert encode(np.zeros((4, 4), bool)) == ""
    assert not decode("", (4, 4)).any()
    assert encode(np.ones((2, 2), bool)) == "1 4"


def test_decode_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="за пределы"):
        decode("15 4", (2, 2))


def test_decode_rejects_zero_start():
    with pytest.raises(ValueError, match="с единицы"):
        decode("0 2", (2, 2))


def _template(tmp_path, chips=("C1", "C2")):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        [{"chip_id": c, "class_id": k, "rle": ""} for c in chips for k in (1, 2, 3)]
    ).to_csv(path, index=False)
    return path


def test_validate_accepts_matching_file(tmp_path):
    template = _template(tmp_path)
    rows = []
    for chip in ("C1", "C2"):
        rows.extend(rows_for_chip(chip, np.zeros((4, 4), np.uint8), (1, 2, 3)))
    out = write(rows, tmp_path / "sub.csv")
    assert validate(out, template) == []


def test_validate_rejects_missing_pair(tmp_path):
    template = _template(tmp_path)
    rows = rows_for_chip("C1", np.zeros((4, 4), np.uint8), (1, 2, 3))
    out = write(rows, tmp_path / "sub.csv")
    faults = validate(out, template)
    assert any("недостающие" in f for f in faults)


def test_validate_rejects_extra_pair(tmp_path):
    template = _template(tmp_path)
    rows = []
    for chip in ("C1", "C2", "C3"):
        rows.extend(rows_for_chip(chip, np.zeros((4, 4), np.uint8), (1, 2, 3)))
    out = write(rows, tmp_path / "sub.csv")
    assert any("лишние" in f for f in validate(out, template))


def test_validate_rejects_class_overlap(tmp_path):
    rows = [
        {"chip_id": "C1", "class_id": 1, "rle": "1 4"},
        {"chip_id": "C1", "class_id": 2, "rle": "3 4"},  # пересекается с классом 1
        {"chip_id": "C1", "class_id": 3, "rle": ""},
    ]
    out = write(rows, tmp_path / "sub.csv")
    faults = validate(out, shapes={"C1": (4, 4)})
    assert any("пересекаются" in f for f in faults)


def test_validate_rejects_run_past_chip(tmp_path):
    rows = [{"chip_id": "C1", "class_id": 1, "rle": "14 9"}]
    out = write(rows, tmp_path / "sub.csv")
    assert any("за пределы" in f for f in validate(out, shapes={"C1": (4, 4)}))


def test_validate_rejects_nan(tmp_path):
    out = tmp_path / "sub.csv"
    out.write_text("chip_id,class_id,rle\nC1,1,nan\n", encoding="utf-8")
    assert any("NaN" in f for f in validate(out))
