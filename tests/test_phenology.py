"""SPEC-9 acceptance tests — офлайн.

Каждый тест назван по критерию приёмки из specs/SPEC-9-phenology-matching.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from src import firms
from src.burn import (BurnConfig, BurnStatus, Candidate, Thresholds, doy_gap,
                      enrichment, load_burn_config, select_pair)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "krasnoyarsk.toml"
UTM = from_origin(500000.0, 6800000.0, 20.0, 20.0)   # 20 м, EPSG:32649
CRS = "EPSG:32649"


def cfg(**kw) -> BurnConfig:
    base = dict(scene_cloud_max=60.0, min_valid_fraction=0.80,
                max_doy_gap_days=30, max_scenes_probed=8,
                min_enrichment=1.5, min_detections_for_validation=20,
                thresholds=Thresholds())
    base.update(kw)
    return BurnConfig(**base)


def at(y, m, d) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def clear(_c):
    return 1.0


def det_at(row, col, transform=UTM, crs_is_utm=True):
    """Детекция, попадающая ровно в пиксель (row, col) растра."""
    from rasterio.warp import transform as warp
    x, y = transform @ (col + 0.5, row + 0.5)
    lon, lat = warp(CRS, "EPSG:4326", [x], [y])
    return firms.Detection(latitude=lat[0], longitude=lon[0], brightness=330.0,
                           frp=5.0, confidence="n", acquired_at=at(2026, 8, 1),
                           sensor="VIIRS_SNPP_NRT", satellite="N", daynight="D")


# --- AC-01: фенологическое ограничение при отборе пары ---

def test_ac01_june_september_pair_is_refused():
    """Ровно та пара, которую выбрал SPEC-7: разрыв 70 суток по дню года."""
    pre = [Candidate("pre-june", "T", at(2026, 6, 28), 0.0)]
    post = [Candidate("post-sept", "T", at(2026, 9, 6), 10.0)]
    assert select_pair(pre, post, cfg(), clear) is None


def test_ac01_august_september_pair_is_accepted():
    pre = [Candidate("pre-aug", "T", at(2026, 8, 20), 10.0)]
    post = [Candidate("post-sept", "T", at(2026, 9, 6), 10.0)]
    pair = select_pair(pre, post, cfg(), clear)
    assert pair is not None and pair.before.item_id == "pre-aug"


def test_ac01_compatible_pair_wins_over_a_cleaner_incompatible_one():
    """Идеально чистая июньская сцена не должна побеждать пыльную августовскую."""
    pre = [Candidate("pre-june", "T", at(2026, 6, 28), 0.0),
           Candidate("pre-aug", "T", at(2026, 8, 20), 40.0)]
    post = [Candidate("post-sept", "T", at(2026, 9, 6), 10.0)]
    pair = select_pair(pre, post, cfg(), clear)
    assert pair.before.item_id == "pre-aug"


def test_ac01_gap_threshold_comes_from_config():
    assert load_burn_config(CONFIG).max_doy_gap_days == 30


def test_ac01_widening_the_gap_admits_the_june_pair():
    pre = [Candidate("pre-june", "T", at(2026, 6, 28), 0.0)]
    post = [Candidate("post-sept", "T", at(2026, 9, 6), 10.0)]
    assert select_pair(pre, post, cfg(max_doy_gap_days=30), clear) is None
    assert select_pair(pre, post, cfg(max_doy_gap_days=90), clear) is not None


@pytest.mark.parametrize("a,b,expected", [
    ((2026, 6, 28), (2026, 9, 6), 70),
    ((2026, 8, 20), (2026, 9, 6), 17),
    ((2026, 9, 6), (2026, 9, 6), 0),
    ((2026, 12, 25), (2027, 1, 5), 11),        # через Новый год
    ((2025, 7, 15), (2026, 7, 20), 5),         # годовщина: разные годы, близкий DOY
])
def test_ac01_doy_gap_is_seasonal_not_calendar(a, b, expected):
    assert doy_gap(at(*a), at(*b)) == expected


def test_ac01_anniversary_pair_is_phenologically_compatible():
    """Сцена прошлого года в ту же декаду фенологически сопоставима."""
    pre = [Candidate("pre-last-year", "T", at(2025, 8, 25), 5.0)]
    post = [Candidate("post", "T", at(2026, 9, 1), 5.0)]
    assert select_pair(pre, post, cfg(), clear) is not None


def test_ac01_still_refuses_scenes_from_different_tiles():
    """Фенология не отменяет ограничения SPEC-7 на единый MGRS-тайл."""
    pre = [Candidate("pre", "46VEH", at(2026, 8, 20), 5.0)]
    post = [Candidate("post", "47VEH", at(2026, 9, 6), 5.0)]
    assert select_pair(pre, post, cfg(), clear) is None


def test_ac01_still_refuses_scenes_clouded_over_the_aoi():
    pre = [Candidate("pre", "T", at(2026, 8, 20), 5.0)]
    post = [Candidate("post", "T", at(2026, 9, 6), 5.0)]
    assert select_pair(pre, post, cfg(), lambda c: 0.30) is None


def test_ac01_probing_is_bounded_by_config():
    """Каждое чтение SCL — сетевая операция; их число обязано быть ограничено."""
    reads = []
    post = [Candidate(f"post-{i}", "T", at(2026, 9, 6), float(i))
            for i in range(40)]
    pre = [Candidate("pre", "T", at(2026, 8, 20), 5.0)]
    select_pair(pre, post, cfg(max_scenes_probed=3), lambda c: (reads.append(c.item_id), 0.1)[1])
    assert len(reads) <= 4


# --- AC-02: обогащение считается верно на синтетике ---

def build(shape=(100, 100), burned_rows=slice(0, 20)):
    classes = np.zeros(shape, dtype="uint8")
    classes[burned_rows, :] = 3
    return classes, np.ones(shape, dtype=bool)


def test_ac02_all_detections_inside_the_scar_give_high_enrichment():
    """Гарь занимает 20% площади, все точки внутри -> обогащение 5.0."""
    classes, valid = build()
    dets = [det_at(r, c) for r in range(0, 20, 2) for c in range(0, 40, 2)]
    score, checked = enrichment(classes, valid, UTM, CRS, dets)
    assert checked == len(dets)
    assert score == pytest.approx(1.0 / 0.20, rel=1e-6)


def test_ac02_detections_spread_uniformly_give_enrichment_of_one():
    classes, valid = build()
    dets = [det_at(r, c) for r in range(0, 100, 2) for c in range(0, 100, 2)]
    score, _ = enrichment(classes, valid, UTM, CRS, dets)
    assert score == pytest.approx(1.0, abs=0.05)


def test_ac02_detections_entirely_outside_the_scar_give_zero():
    classes, valid = build()
    dets = [det_at(r, c) for r in range(40, 90, 2) for c in range(0, 40, 2)]
    score, checked = enrichment(classes, valid, UTM, CRS, dets)
    assert checked == len(dets) and score == pytest.approx(0.0)


def test_ac02_the_spec7_failure_reproduces_as_low_enrichment():
    """Случай SPEC-7-LIVE-002: 40% площади помечено гарью, но точки не там."""
    classes = np.zeros((100, 100), dtype="uint8")
    classes[:40, :] = 1
    valid = np.ones((100, 100), dtype=bool)
    dets = [det_at(r, c) for r in range(60, 100, 2) for c in range(0, 40, 2)]
    score, _ = enrichment(classes, valid, UTM, CRS, dets)
    assert score < 1.0


def test_ac02_masked_pixels_are_excluded_from_both_sides():
    classes, valid = build()
    valid[50:, :] = False
    dets = [det_at(r, c) for r in range(0, 20, 2) for c in range(0, 20, 2)]
    score, checked = enrichment(classes, valid, UTM, CRS, dets)
    assert checked == len(dets)
    assert score == pytest.approx(1.0 / 0.40, rel=1e-6)   # гарь теперь 40% валидных


def test_ac02_no_burned_pixels_yields_no_score():
    classes = np.zeros((50, 50), dtype="uint8")
    score, checked = enrichment(classes, np.ones((50, 50), dtype=bool), UTM, CRS,
                                [det_at(10, 10)])
    assert score is None and checked == 0


def test_ac02_detections_outside_the_raster_are_not_counted():
    classes, valid = build()
    far = firms.Detection(latitude=10.0, longitude=10.0, brightness=330.0, frp=5.0,
                          confidence="n", acquired_at=at(2026, 8, 1),
                          sensor="VIIRS_SNPP_NRT", satellite="N", daynight="D")
    _, checked = enrichment(classes, valid, UTM, CRS, [far])
    assert checked == 0


# --- AC-03: непрошедший результат нельзя опубликовать как проверенный ---

def test_ac03_only_ok_status_counts_as_validated():
    from src.burn import BurnResult, SEVERITY_ORDER
    common = dict(event_id="E", area_ha={k: 1.0 for k in SEVERITY_ORDER},
                  masked_fraction=0.0, thresholds_version="v1")
    assert BurnResult(status=BurnStatus.OK, **common).validated is True
    for bad in (BurnStatus.UNVALIDATED, BurnStatus.DEFERRED, BurnStatus.FAILED):
        assert BurnResult(status=bad, **common).validated is False


def test_ac03_unvalidated_is_a_distinct_status():
    assert len({BurnStatus.OK, BurnStatus.UNVALIDATED,
                BurnStatus.DEFERRED, BurnStatus.FAILED}) == 4


def test_ac03_thresholds_for_validation_come_from_config():
    c = load_burn_config(CONFIG)
    assert c.min_enrichment == 1.5
    assert c.min_detections_for_validation == 20


def test_ac03_api_reports_validated_from_the_result_not_a_constant():
    from src.api import burn_payload
    from src.burn import BurnResult, SEVERITY_ORDER
    common = dict(event_id="E", area_ha={k: 1.0 for k in SEVERITY_ORDER},
                  masked_fraction=0.0, thresholds_version="v1",
                  enrichment=3.2, detections_checked=140, doy_gap_days=17)
    good = burn_payload(BurnResult(status=BurnStatus.OK, **common))
    bad = burn_payload(BurnResult(status=BurnStatus.UNVALIDATED,
                                  reason="обогащение 0.25", **common))
    assert good.validated is True and good.validation_note is None
    assert bad.validated is False and "обогащение" in bad.validation_note
    assert good.enrichment == 3.2 and good.doy_gap_days == 17


# --- AC-02 (регрессия): проекция берётся из растра, а не из свойств STAC ---

def test_window_read_reports_the_raster_crs(tmp_path):
    """Регрессия DEF-01.

    Расширение projection в STAC переименовало proj:epsg в proj:code. Код читал
    proj:epsg, не находил его и молча пропускал перекрёстную проверку — все
    результаты помечались непроверенными по ложной причине. Проекция обязана
    браться из самого растра.
    """
    import rasterio
    from src.burn import _read_window

    path = tmp_path / "t.tif"
    with rasterio.open(path, "w", driver="GTiff", height=50, width=50, count=1,
                       dtype="uint8", crs=CRS, transform=UTM) as dst:
        dst.write(np.ones((50, 50), dtype="uint8"), 1)

    from rasterio.warp import transform_bounds
    west, south, east, north = transform_bounds(
        CRS, "EPSG:4326", *rasterio.open(path).bounds)
    data, transform, requested, crs = _read_window(str(path), (west, south, east, north))
    assert crs.to_string() == CRS
    assert data.size > 0 and requested > 0


def test_enrichment_accepts_a_crs_object_not_only_a_string(tmp_path):
    """_read_window отдаёт rasterio CRS; enrichment обязан его принимать."""
    import rasterio
    classes, valid = build()
    dets = [det_at(r, c) for r in range(0, 20, 2) for c in range(0, 40, 2)]
    score, checked = enrichment(classes, valid, UTM, rasterio.crs.CRS.from_string(CRS), dets)
    assert checked == len(dets) and score == pytest.approx(5.0, rel=1e-6)
