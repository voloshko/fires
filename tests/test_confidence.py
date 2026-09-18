"""SPEC-3 acceptance tests — офлайн.

Каждый тест назван по критерию приёмки из specs/SPEC-3-confidence-filter.md.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import firms
from src.confidence import (Level, Thresholds, UnknownConfidenceError,
                            load_thresholds, normalize, partition, scale_of)
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = ROOT / "config" / "krasnoyarsk.toml"
T0 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)

DEFAULT = Thresholds()


def det(**kw) -> firms.Detection:
    base = dict(latitude=61.0, longitude=93.0, brightness=330.0, frp=5.0,
                confidence="n", acquired_at=T0, sensor="VIIRS_SNPP_NRT",
                satellite="N", daynight="D")
    base.update(kw)
    return firms.Detection(**base)


# --- AC-01: категориальная шкала VIIRS ---

@pytest.mark.parametrize("raw,expected", [
    ("l", Level.LOW), ("n", Level.NOMINAL), ("h", Level.HIGH),
    ("L", Level.LOW), ("H", Level.HIGH), (" n ", Level.NOMINAL),
])
def test_ac01_viirs_scale_maps_all_three_values(raw, expected):
    assert normalize("VIIRS_SNPP_NRT", raw, DEFAULT) is expected


@pytest.mark.parametrize("raw", ["x", "medium", "", "  ", "1", "nominal"])
def test_ac01_unknown_viirs_value_raises_instead_of_silently_becoming_low(raw):
    """Тихая деградация к low выглядела бы как работающий фильтр
    и незаметно выбрасывала бы реальные пожары."""
    with pytest.raises(UnknownConfidenceError):
        normalize("VIIRS_SNPP_NRT", raw, DEFAULT)


def test_ac01_unknown_sensor_raises():
    with pytest.raises(UnknownConfidenceError, match="unknown sensor"):
        normalize("SENTINEL_9000", "n", DEFAULT)


def test_ac01_scale_is_chosen_per_sensor_family():
    assert scale_of("VIIRS_NOAA21_NRT") == "categorical"
    assert scale_of("MODIS_SP") == "numeric"


# --- AC-02: числовая шкала MODIS, границы из конфига ---

@pytest.mark.parametrize("value,expected", [
    (0, Level.LOW), (29, Level.LOW),        # граница low/nominal = 30
    (30, Level.NOMINAL), (79, Level.NOMINAL),  # граница nominal/high = 80
    (80, Level.HIGH), (100, Level.HIGH),
])
def test_ac02_modis_boundaries_both_sides(value, expected):
    assert normalize("MODIS_NRT", str(value), DEFAULT) is expected


def test_ac02_changing_the_boundary_changes_the_verdict():
    """Тот же вход, другие пороги — другой класс. Ради этого они в конфиге."""
    strict = Thresholds(modis_low_below=50, modis_high_at_least=90)
    assert normalize("MODIS_NRT", "40", DEFAULT) is Level.NOMINAL
    assert normalize("MODIS_NRT", "40", strict) is Level.LOW
    assert normalize("MODIS_NRT", "85", DEFAULT) is Level.HIGH
    assert normalize("MODIS_NRT", "85", strict) is Level.NOMINAL


def test_ac02_thresholds_come_from_the_shipped_config():
    t = load_thresholds(CONFIG)
    assert (t.modis_low_below, t.modis_high_at_least) == (30, 80)
    assert t.min_level is Level.NOMINAL


def test_ac02_config_with_custom_boundaries_is_honoured(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[confidence]\nmodis_low_below = 10\n'
                 'modis_high_at_least = 95\nmin_level = "high"\n')
    t = load_thresholds(p)
    assert (t.modis_low_below, t.modis_high_at_least) == (10, 95)
    assert t.min_level is Level.HIGH


@pytest.mark.parametrize("raw", ["-1", "101", "abc", "", "7.5"])
def test_ac02_out_of_range_or_non_numeric_modis_value_raises(raw):
    with pytest.raises(UnknownConfidenceError):
        normalize("MODIS_NRT", raw, DEFAULT)


def test_ac02_incoherent_boundaries_are_rejected():
    with pytest.raises(ValueError, match="boundaries must satisfy"):
        Thresholds(modis_low_below=90, modis_high_at_least=30)


# --- AC-03: фильтр по умолчанию отсекает low ---

def test_ac03_default_filter_drops_low_keeps_nominal_and_high():
    dets = [det(confidence="l"), det(confidence="n", latitude=61.1),
            det(confidence="h", latitude=61.2)]
    kept, rejected = partition(dets, DEFAULT)
    assert [d.confidence for d in kept] == ["n", "h"]
    assert [d.confidence for d in rejected] == ["l"]


def test_ac03_disabling_the_filter_returns_everything():
    dets = [det(confidence=c, latitude=61.0 + i / 10)
            for i, c in enumerate("lnh")]
    kept, rejected = partition(dets, Thresholds(min_level=Level.LOW))
    assert len(kept) == 3 and rejected == []


def test_ac03_raising_the_threshold_to_high_drops_nominal_too():
    dets = [det(confidence=c, latitude=61.0 + i / 10)
            for i, c in enumerate("lnh")]
    kept, _ = partition(dets, Thresholds(min_level=Level.HIGH))
    assert [d.confidence for d in kept] == ["h"]


def test_ac03_filter_works_across_both_scales_at_once():
    """Ровно тот случай, который ломала формулировка ТЗ."""
    dets = [
        det(sensor="VIIRS_SNPP_NRT", confidence="l", latitude=61.0),
        det(sensor="MODIS_NRT", confidence="12", latitude=61.1),   # low
        det(sensor="MODIS_NRT", confidence="78", latitude=61.2),   # nominal
    ]
    kept, rejected = partition(dets, DEFAULT)
    assert len(kept) == 1 and kept[0].confidence == "78"
    assert len(rejected) == 2


def test_ac03_real_fixture_partitions_as_expected():
    dets = firms.parse_csv(
        (FIXTURES / "firms_viirs_nonempty.csv").read_text(), "VIIRS_SNPP_NRT")
    kept, rejected = partition(dets, DEFAULT)
    assert len(kept) == 2 and len(rejected) == 1
    assert rejected[0].confidence == "l"


# --- AC-04: отсеянные не теряются ---

def test_ac04_rejected_detections_stay_in_the_store():
    """Фильтр — представление, а не мутация: хранилище append-only (SPEC-2)."""
    dets = [det(confidence="l"), det(confidence="h", latitude=61.5)]
    with Store(":memory:") as s:
        s.add_detections(dets)
        kept, rejected = partition(s_detections(s), DEFAULT)
        assert len(kept) == 1 and len(rejected) == 1
        # ни одна строка не удалена
        assert s.count_detections() == 2


def test_ac04_service_layer_can_retrieve_the_rejected_ones():
    with Store(":memory:") as s:
        s.add_detections([det(confidence="l"), det(confidence="h", latitude=61.5)])
        _, rejected = partition(s_detections(s), DEFAULT)
        assert [d.confidence for d in rejected] == ["l"]


def s_detections(store) -> list[firms.Detection]:
    """Прочитать детекции из хранилища обратно в доменный тип."""
    from datetime import datetime as dt
    rows = store.detections_between(
        dt(2000, 1, 1, tzinfo=timezone.utc), dt(2100, 1, 1, tzinfo=timezone.utc))
    return [firms.Detection(
        latitude=r["latitude"], longitude=r["longitude"],
        brightness=r["brightness"], frp=r["frp"], confidence=r["confidence"],
        acquired_at=dt.fromisoformat(r["acquired_at"]), sensor=r["sensor"],
        satellite=r["satellite"], daynight=r["daynight"]) for r in rows]
