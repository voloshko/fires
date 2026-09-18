"""SPEC-7 acceptance tests — офлайн, синтетические растры и подставной каталог.

Каждый тест назван по критерию приёмки из specs/SPEC-7-burn-mapping.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from src import firms
from src.burn import (SCL_INVALID, SEVERITY_ORDER, BurnConfig, BurnStatus,
                      Candidate, SeasonBand, Thresholds, area_by_class, classify,
                      deferred, load_burn_config, nbr, pixel_area_ha, rbr,
                      season_check, select_pair, valid_mask)
from src.events import build_events, EventConfig

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "krasnoyarsk.toml"
TH = Thresholds()
# 20-метровая сетка UTM: пиксель = 0.04 га
T20 = from_origin(500000.0, 6800000.0, 20.0, 20.0)


def cfg(**kw) -> BurnConfig:
    base = dict(scene_cloud_max=60.0, min_valid_fraction=0.80, thresholds=TH)
    base.update(kw)
    return BurnConfig(**base)


# --- AC-01: пара строго из одного MGRS-тайла ---

def always_clear(_c):
    return 1.0


def test_ac01_pair_is_never_assembled_from_two_different_tiles():
    """Разные тайлы — разные зоны UTM и сетки; вычитание дало бы мусор."""
    pre = [Candidate("pre-46VEH", "46VEH", datetime(2026, 6, 1, tzinfo=timezone.utc), 5.0)]
    post = [Candidate("post-47VEH", "47VEH", datetime(2026, 8, 1, tzinfo=timezone.utc), 5.0)]
    assert select_pair(pre, post, cfg(), always_clear) is None


def test_ac01_pair_is_assembled_when_a_common_tile_exists():
    pre = [Candidate("pre-46VEH", "46VEH", datetime(2026, 6, 1, tzinfo=timezone.utc), 5.0),
           Candidate("pre-47VEH", "47VEH", datetime(2026, 6, 2, tzinfo=timezone.utc), 5.0)]
    post = [Candidate("post-47VEH", "47VEH", datetime(2026, 8, 1, tzinfo=timezone.utc), 5.0)]
    pair = select_pair(pre, post, cfg(), always_clear)
    assert pair is not None and pair.tile == "47VEH"
    assert pair.before.item_id == "pre-47VEH" and pair.after.item_id == "post-47VEH"


def test_ac01_best_common_tile_wins_when_several_are_available():
    pre = [Candidate("pre-A", "A", datetime(2026, 6, 1, tzinfo=timezone.utc), 5.0),
           Candidate("pre-B", "B", datetime(2026, 6, 1, tzinfo=timezone.utc), 5.0)]
    post = [Candidate("post-A", "A", datetime(2026, 8, 1, tzinfo=timezone.utc), 5.0),
            Candidate("post-B", "B", datetime(2026, 8, 1, tzinfo=timezone.utc), 5.0)]
    fracs = {"pre-A": 0.85, "post-A": 0.85, "pre-B": 0.99, "post-B": 0.99}
    pair = select_pair(pre, post, cfg(), lambda c: fracs[c.item_id])
    assert pair.tile == "B"


# --- AC-02: двухступенчатый фильтр облачности ---

def test_ac02_cloudy_scene_that_is_clear_over_the_aoi_is_accepted():
    """Ровно тот случай, который ломала формулировка ТЗ (порог 20% по сцене)."""
    pre = [Candidate("pre", "T", datetime(2026, 6, 1, tzinfo=timezone.utc), 45.0)]
    post = [Candidate("post", "T", datetime(2026, 8, 1, tzinfo=timezone.utc), 45.0)]
    pair = select_pair(pre, post, cfg(), lambda c: 0.95)
    assert pair is not None, "сцена 45% облачности, но чистая над AOI, должна приниматься"


def test_ac02_clear_scene_that_is_clouded_over_the_aoi_is_rejected():
    pre = [Candidate("pre", "T", datetime(2026, 6, 1, tzinfo=timezone.utc), 15.0)]
    post = [Candidate("post", "T", datetime(2026, 8, 1, tzinfo=timezone.utc), 15.0)]
    pair = select_pair(pre, post, cfg(), lambda c: 0.30)
    assert pair is None, "сцена 15% облачности, но закрытая над AOI, должна отклоняться"


def test_ac02_scene_above_the_coarse_cloud_threshold_never_reaches_stage_two():
    read = []
    pre = [Candidate("pre", "T", datetime(2026, 6, 1, tzinfo=timezone.utc), 95.0)]
    post = [Candidate("post", "T", datetime(2026, 8, 1, tzinfo=timezone.utc), 5.0)]

    def spy(c):
        read.append(c.item_id)
        return 1.0

    assert select_pair(pre, post, cfg(), spy) is None
    assert "pre" not in read, "сцена 95% не должна вызывать чтение SCL"


def test_ac02_valid_fraction_threshold_comes_from_config():
    pre = [Candidate("pre", "T", datetime(2026, 6, 1, tzinfo=timezone.utc), 10.0)]
    post = [Candidate("post", "T", datetime(2026, 8, 1, tzinfo=timezone.utc), 10.0)]
    assert select_pair(pre, post, cfg(min_valid_fraction=0.80), lambda c: 0.70) is None
    assert select_pair(pre, post, cfg(min_valid_fraction=0.60), lambda c: 0.70) is not None


def test_ac02_scl_is_read_at_most_once_per_scene():
    calls = []
    pre = [Candidate("pre", "T", datetime(2026, 6, 1, tzinfo=timezone.utc), 10.0)]
    post = [Candidate("post", "T", datetime(2026, 8, 1, tzinfo=timezone.utc), 10.0)]
    select_pair(pre, post, cfg(), lambda c: (calls.append(c.item_id), 0.9)[1])
    assert len(calls) == len(set(calls))


# --- AC-03: классы SCL исключаются из статистики ---

@pytest.mark.parametrize("bad", sorted(SCL_INVALID))
def test_ac03_every_invalid_scl_class_is_excluded(bad):
    scl = np.array([[bad, 4]], dtype="uint8")
    nir = np.array([[3000, 3000]], dtype="uint16")
    swir = np.array([[1000, 1000]], dtype="uint16")
    assert valid_mask(scl, nir, swir).tolist() == [[False, True]]


@pytest.mark.parametrize("good", [2, 4, 5, 6, 7])
def test_ac03_valid_scl_classes_are_kept(good):
    scl = np.array([[good]], dtype="uint8")
    nir = np.array([[3000]], dtype="uint16")
    swir = np.array([[1000]], dtype="uint16")
    assert valid_mask(scl, nir, swir).tolist() == [[True]]


def test_ac03_snow_is_excluded_because_it_distorts_nbr():
    assert 11 in SCL_INVALID


# --- AC-04: площадь в нативной UTM-проекции, пиксель 20 м = 0.04 га ---

def test_ac04_pixel_area_is_exactly_four_hundredths_of_a_hectare():
    assert pixel_area_ha(T20) == pytest.approx(0.04)


def test_ac04_area_of_a_known_rectangle_matches_the_analytic_value():
    """Прямоугольник 10x25 пикселей по 20 м = 200 x 500 м = 10 га."""
    classes = np.zeros((40, 40), dtype="uint8")
    classes[5:15, 10:35] = 4                 # 10 x 25 = 250 пикселей
    valid = np.ones((40, 40), dtype=bool)
    areas = area_by_class(classes, valid, T20)
    assert areas["high"] == pytest.approx(250 * 0.04)
    assert areas["high"] == pytest.approx(10.0)          # 200 м x 500 м
    assert areas["unburnt"] == pytest.approx((40 * 40 - 250) * 0.04)
    assert sum(areas.values()) == pytest.approx(40 * 40 * 0.04)


def test_ac04_masked_pixels_do_not_contribute_to_area():
    classes = np.full((10, 10), 4, dtype="uint8")
    valid = np.zeros((10, 10), dtype=bool)
    valid[:5, :] = True                      # половина закрыта
    areas = area_by_class(classes, valid, T20)
    assert areas["high"] == pytest.approx(50 * 0.04)


def test_ac04_pixel_area_follows_the_transform_not_a_constant():
    t10 = from_origin(500000.0, 6800000.0, 10.0, 10.0)
    assert pixel_area_ha(t10) == pytest.approx(0.01)


# --- AC-05: masked_fraction обязателен ---

def test_ac05_burn_result_cannot_be_built_without_masked_fraction():
    from src.burn import BurnResult
    with pytest.raises(TypeError):
        BurnResult(status=BurnStatus.OK, event_id="E",
                   area_ha={k: 0.0 for k in SEVERITY_ORDER},
                   thresholds_version="v1")      # masked_fraction пропущен


def test_ac05_deferred_result_also_carries_masked_fraction():
    r = deferred("E", "полярная ночь", "2027-06-01", "v1")
    assert r.masked_fraction == 1.0
    assert r.status is BurnStatus.DEFERRED


# --- AC-06: пороги из конфига и версия в ответе ---

def test_ac06_thresholds_come_from_the_shipped_config():
    c = load_burn_config(CONFIG)
    assert c.thresholds.version == "usgs-baseline-1"
    assert c.thresholds.breaks == (0.10, 0.27, 0.44, 0.66)


@pytest.mark.parametrize("value,expected", [
    (-0.5, 0), (0.0, 0), (0.10, 0),          # граница unburnt/low
    (0.11, 1), (0.27, 1),                    # граница low/moderate_low
    (0.28, 2), (0.44, 2),
    (0.45, 3), (0.66, 3),
    (0.67, 4), (1.3, 4),
])
def test_ac06_classification_boundaries_both_sides(value, expected):
    assert classify(np.array([[value]], dtype="float32"), TH)[0, 0] == expected


def test_ac06_changing_the_thresholds_changes_the_classification():
    strict = Thresholds(version="strict", unburnt=0.3, low=0.5,
                        moderate_low=0.7, moderate_high=0.9)
    d = np.array([[0.4]], dtype="float32")
    assert classify(d, TH)[0, 0] == 2
    assert classify(d, strict)[0, 0] == 1


def test_ac06_incoherent_thresholds_are_rejected():
    with pytest.raises(ValueError, match="must increase"):
        Thresholds(unburnt=0.6, low=0.2, moderate_low=0.4, moderate_high=0.8)


# --- AC-07: сезонное окно и статус deferred ---

SEASON = (SeasonBand(60.0, (5, 1), (10, 31)),
          SeasonBand(66.0, (5, 15), (10, 15)),
          SeasonBand(90.0, (6, 1), (9, 30)))


def test_ac07_fire_ending_in_november_above_66n_is_deferred():
    """Полярная ночь: оптики нет вообще, ноль гектаров был бы ложью."""
    ok, reason, retry = season_check(
        68.0, datetime(2026, 11, 10, tzinfo=timezone.utc), cfg(season=SEASON))
    assert ok is False
    assert "вне безснежного" in reason
    assert retry == "2027-06-01"


def test_ac07_fire_ending_in_july_above_66n_is_not_deferred():
    ok, reason, _ = season_check(
        68.0, datetime(2026, 7, 10, tzinfo=timezone.utc), cfg(season=SEASON))
    assert ok is True and reason is None


def test_ac07_season_band_depends_on_latitude():
    """На 55 градусах октябрь ещё рабочий, на 68 — уже нет."""
    late_oct = datetime(2026, 10, 5, tzinfo=timezone.utc)
    assert season_check(55.0, late_oct, cfg(season=SEASON))[0] is True
    assert season_check(68.0, late_oct, cfg(season=SEASON))[0] is False


def test_ac07_retry_date_is_the_next_season_opening():
    _, _, retry = season_check(
        68.0, datetime(2026, 2, 1, tzinfo=timezone.utc), cfg(season=SEASON))
    assert retry == "2026-06-01"       # сезон этого же года ещё впереди


def test_ac07_deferred_never_reports_zero_hectares_as_a_result():
    r = deferred("E", "полярная ночь", "2027-06-01", "v1")
    assert r.status is not BurnStatus.OK
    assert r.reason and r.retry_after
    assert sum(r.area_ha.values()) == 0.0    # нули есть, но статус их дисквалифицирует


def test_ac07_config_season_bands_are_loaded():
    c = load_burn_config(CONFIG)
    assert len(c.season) == 3
    assert c.season_for(55.0).max_lat == 60.0
    assert c.season_for(68.0).max_lat == 90.0


# --- AC-08: ноль в каналах — это nodata ---

def test_ac08_zero_in_nir_or_swir_is_nodata_not_a_valid_zero():
    scl = np.array([[4, 4, 4]], dtype="uint8")
    nir = np.array([[3000, 0, 3000]], dtype="uint16")
    swir = np.array([[1000, 1000, 0]], dtype="uint16")
    assert valid_mask(scl, nir, swir).tolist() == [[True, False, False]]


def test_ac08_nbr_of_a_zero_sum_pixel_is_nan_not_a_number():
    out = nbr(np.array([[0]], dtype="uint16"), np.array([[0]], dtype="uint16"))
    assert np.isnan(out[0, 0])


# --- индексы ---

def test_nbr_matches_the_analytic_formula():
    out = nbr(np.array([[3000]], dtype="uint16"), np.array([[1000]], dtype="uint16"))
    assert out[0, 0] == pytest.approx((0.3 - 0.1) / (0.3 + 0.1))


def test_nbr_is_scale_invariant():
    a = nbr(np.array([[3000]], dtype="uint16"), np.array([[1000]], dtype="uint16"))
    b = nbr(np.array([[6000]], dtype="uint16"), np.array([[2000]], dtype="uint16"))
    assert a[0, 0] == pytest.approx(b[0, 0])


def test_rbr_matches_the_analytic_formula():
    d = np.array([[0.5]], dtype="float32")
    pre = np.array([[0.7]], dtype="float32")
    assert rbr(d, pre)[0, 0] == pytest.approx(0.5 / (0.7 + 1.001), rel=1e-5)


def test_burned_ha_excludes_the_unburnt_class():
    from src.burn import BurnResult
    r = BurnResult(status=BurnStatus.OK, event_id="E",
                   area_ha={"unburnt": 100.0, "low": 10.0, "moderate_low": 5.0,
                            "moderate_high": 2.0, "high": 1.0},
                   masked_fraction=0.1, thresholds_version="v1")
    assert r.burned_ha == pytest.approx(18.0)


# --- защита размера области ---

def test_aoi_extent_is_measured_in_kilometres_not_degrees():
    """Регрессия: порог задавался в градусах и на 64 с.ш. резал область
    вдвое строже, чем на экваторе. Одна и та же угловая ширина — разный размер."""
    from src.burn import aoi_extent_km
    at_equator = aoi_extent_km((100.0, 0.0, 100.5, 0.2))
    at_64n = aoi_extent_km((100.0, 63.6, 100.5, 63.8))
    assert at_equator[0] == pytest.approx(55.7, rel=0.02)
    assert at_64n[0] == pytest.approx(24.7, rel=0.02)
    assert at_64n[0] < at_equator[0] / 2


def test_real_siberian_fire_aoi_is_within_the_guard():
    """AOI реального события FIRE-3ce352bc7175: 0.544 градуса долготы на 63.7 с.ш.
    Это 27 км, и порог в градусах ошибочно её отклонял."""
    from src.burn import aoi_extent_km
    w, h = aoi_extent_km((108.275, 63.596, 108.819, 63.782))
    assert w < 30 and h < 25
    assert max(w, h) <= BurnConfig().max_aoi_km


def test_genuinely_huge_aoi_is_still_rejected():
    from src.burn import aoi_extent_km
    w, h = aoi_extent_km((100.0, 60.0, 104.0, 62.0))
    assert max(w, h) > BurnConfig().max_aoi_km


# --- AC-05 (регрессия): masked_fraction считается от ВСЕЙ области ---

def test_masked_fraction_denominator_is_the_requested_aoi_not_what_was_read():
    """Регрессия DEF-01, найдена живым прогоном.

    Сцена, задевающая область краем полосы съёмки, читается обрезанной.
    Если делить на прочитанное, она рапортует «0% под маской», и площадь,
    посчитанная по десятой части пожара, выглядит полной.
    """
    from src.burn import masked_fraction
    valid = np.ones((100, 100), dtype=bool)      # прочитано 10 000, все валидны
    assert masked_fraction(valid, 10_000) == pytest.approx(0.0)
    # запрошено было 100 000 — значит 90% области вообще не покрыто сценой
    assert masked_fraction(valid, 100_000) == pytest.approx(0.9)


def test_masked_fraction_counts_clouded_pixels_too():
    from src.burn import masked_fraction
    valid = np.zeros((100, 100), dtype=bool)
    valid[:50] = True
    assert masked_fraction(valid, 10_000) == pytest.approx(0.5)


def test_masked_fraction_of_an_uncovered_aoi_is_total():
    from src.burn import masked_fraction
    assert masked_fraction(np.zeros((0, 0), dtype=bool), 0) == 1.0
    assert masked_fraction(np.zeros((1, 1), dtype=bool), 5_000) == pytest.approx(1.0)
