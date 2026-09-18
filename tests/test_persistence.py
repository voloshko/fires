"""SPEC-4 acceptance tests — синтетические ряды.

ВАЖНО: эти тесты доказывают ЛОГИКУ фильтра. Они не доказывают, что фильтр
находит настоящие газовые факелы — для этого нужен прогон на живых данных
(см. receipt SPEC-4-LIVE-001 и раздел Resolution спеки).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import firms
from src.persistence import (PersistenceConfig, PersistentSource, analyse,
                             cluster_detections, load_persistence_config,
                             partition)
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "krasnoyarsk.toml"
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
CFG = PersistenceConfig(window_days=14, radius_m=500, min_distinct_days=10,
                        max_frp_cv=None)   # интенсивность отдельной группой тестов

M_PER_DEG = 111_320.0


def det(lat, lon, when, frp=5.0) -> firms.Detection:
    return firms.Detection(latitude=lat, longitude=lon, brightness=330.0, frp=frp,
                           confidence="n", acquired_at=when,
                           sensor="VIIRS_SNPP_NRT", satellite="N", daynight="D")


def stationary(days: int, lat=61.0, lon=93.0, frp=5.0, per_day=2):
    """Неподвижный источник: та же точка каждый день."""
    return [det(lat, lon, T0 + timedelta(days=d, hours=h), frp)
            for d in range(days) for h in range(per_day)]


def moving(days: int, km_total: float, lat=61.0, lon=93.0):
    """Источник, уходящий на km_total за весь период — как фронт пожара."""
    step = (km_total * 1000 / M_PER_DEG) / max(days - 1, 1)
    return [det(lat + step * d, lon, T0 + timedelta(days=d, hours=h))
            for d in range(days) for h in range(2)]


# --- AC-01: неподвижный помечается, движущийся нет ---

def test_ac01_stationary_source_over_20_days_is_flagged():
    r = analyse(stationary(20), CFG)
    assert r.enough_history is True
    assert len(r.sources) == 1
    assert r.sources[0].distinct_days >= CFG.min_distinct_days


def test_ac01_source_moving_2km_over_5_days_is_not_flagged():
    """Ряд из спеки: смещение 2 км за 5 суток."""
    dets = moving(5, 2.0) + stationary(20, lat=50.0, lon=80.0)  # второй ряд даёт историю
    r = analyse(dets, CFG)
    # помечен только неподвижный, движущийся — нет
    assert all(abs(s.latitude - 61.0) > 0.01 for s in r.sources)


def test_ac01_moving_front_alone_yields_no_persistent_source():
    dets = moving(20, 8.0)     # 8 км за 20 суток — фронт пожара
    r = analyse(dets, CFG)
    assert r.enough_history is True
    assert r.sources == ()


def test_ac01_stationary_but_too_few_days_is_not_flagged():
    """Присутствует только 5 суток из 14 — не факел."""
    dets = stationary(5) + stationary(20, lat=50.0, lon=80.0)
    r = analyse(dets, CFG)
    assert all(abs(s.latitude - 61.0) > 0.01 for s in r.sources)


# --- AC-02: окно и радиус из конфига ---

def test_ac02_config_is_read_from_the_shipped_file():
    c = load_persistence_config(CONFIG)
    assert (c.window_days, c.radius_m, c.min_distinct_days) == (14, 500.0, 10)
    assert c.max_frp_cv == 0.6


def test_ac02_changing_the_window_changes_the_verdict():
    """Один и тот же ряд: при окне 14 истории хватает, при 30 — нет."""
    dets = stationary(20)
    assert analyse(dets, PersistenceConfig(window_days=14, min_distinct_days=10,
                                           max_frp_cv=None)).enough_history is True
    assert analyse(dets, PersistenceConfig(window_days=30, min_distinct_days=10,
                                           max_frp_cv=None)).enough_history is False


def test_ac02_changing_the_radius_changes_the_clustering():
    """Две точки в 800 м: при радиусе 500 — два кластера, при 1000 — один."""
    offset = 800 / M_PER_DEG
    dets = stationary(20) + stationary(20, lat=61.0 + offset)
    tight = analyse(dets, PersistenceConfig(radius_m=500, min_distinct_days=10, max_frp_cv=None))
    loose = analyse(dets, PersistenceConfig(radius_m=1000, min_distinct_days=10, max_frp_cv=None))
    assert len(tight.sources) == 2
    assert len(loose.sources) == 1


def test_ac02_incoherent_config_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        PersistenceConfig(window_days=7, min_distinct_days=14)


# --- AC-03: «недостаточно истории» — отдельный исход ---

def test_ac03_short_history_reports_insufficient_not_empty():
    r = analyse(stationary(5), CFG)
    assert r.enough_history is False
    assert r.status == "insufficient_history"
    assert r.sources == ()


def test_ac03_insufficient_history_is_distinguishable_from_no_flares():
    """Ровно то различие, ради которого статус существует.

    Оба случая дают пустой список источников, но означают
    прямо противоположное."""
    short = analyse(stationary(5), CFG)              # нечего сказать
    long_clean = analyse(moving(20, 8.0), CFG)       # сказать есть что: факелов нет
    assert short.sources == long_clean.sources == ()
    assert short.enough_history is False
    assert long_clean.enough_history is True
    assert short.status != long_clean.status


def test_ac03_empty_input_is_insufficient_history():
    r = analyse([], CFG)
    assert r.enough_history is False and r.history_days == 0.0


def test_ac03_filter_with_insufficient_history_discards_nothing():
    """Фильтр, который не может ответить, не имеет права выбрасывать данные."""
    dets = stationary(5)
    r = analyse(dets, CFG)
    normal, persistent = partition(dets, r, CFG)
    assert len(normal) == len(dets) and persistent == []


# --- AC-04: помеченные не теряются ---

def test_ac04_flagged_points_go_to_the_service_layer_not_to_the_bin():
    dets = stationary(20) + moving(20, 8.0, lat=50.0, lon=80.0)
    r = analyse(dets, CFG)
    normal, persistent = partition(dets, r, CFG)
    assert len(persistent) > 0 and len(normal) > 0
    assert len(normal) + len(persistent) == len(dets)   # ничего не потеряно


def test_ac04_store_is_untouched_by_filtering():
    dets = stationary(20)
    with Store(":memory:") as s:
        s.add_detections(dets)
        before = s.count_detections()
        partition(dets, analyse(dets, CFG), CFG)
        assert s.count_detections() == before


# --- AC-05: глубина истории измерима ---

def test_ac05_history_days_is_published_and_accurate():
    r = analyse(stationary(20), CFG)
    assert 18.5 < r.history_days < 20.5
    assert r.window_days == 14


def test_ac05_history_days_reported_even_when_insufficient():
    """Потребитель должен видеть, СКОЛЬКО осталось накопить."""
    r = analyse(stationary(6), CFG)
    assert r.enough_history is False
    assert 4.5 < r.history_days < 6.5


# --- интенсивность FRP: факел стабилен, пожар скачет ---

def test_frp_variation_separates_a_steady_source_from_an_erratic_one():
    steady = stationary(20, frp=5.0)
    cfg = PersistenceConfig(min_distinct_days=10, max_frp_cv=0.6)
    assert len(analyse(steady, cfg).sources) == 1

    erratic = [det(61.0, 93.0, T0 + timedelta(days=d, hours=h),
                   frp=1.0 if (d + h) % 2 else 200.0)
               for d in range(20) for h in range(2)]
    assert analyse(erratic, cfg).sources == ()


def test_frp_check_can_be_disabled():
    erratic = [det(61.0, 93.0, T0 + timedelta(days=d, hours=h),
                   frp=1.0 if (d + h) % 2 else 200.0)
               for d in range(20) for h in range(2)]
    assert len(analyse(erratic, PersistenceConfig(min_distinct_days=10,
                                                  max_frp_cv=None)).sources) == 1


def test_missing_frp_does_not_crash_the_analysis():
    dets = [det(61.0, 93.0, T0 + timedelta(days=d), frp=None) for d in range(20)]
    r = analyse(dets, PersistenceConfig(min_distinct_days=10, max_frp_cv=0.6))
    assert len(r.sources) == 1 and r.sources[0].frp_cv is None


# --- кластеризация ---

def test_clustering_groups_by_distance_not_by_grid_cell():
    """Две точки в 100 м не должны разойтись по кластерам
    только потому, что легли в соседние ячейки сетки."""
    dets = [det(61.0, 93.0, T0), det(61.0 + 100 / M_PER_DEG, 93.0, T0)]
    assert len(cluster_detections(dets, 500)) == 1


def test_clustering_separates_points_beyond_the_radius():
    dets = [det(61.0, 93.0, T0), det(61.0 + 3000 / M_PER_DEG, 93.0, T0)]
    assert len(cluster_detections(dets, 500)) == 2


def test_clustering_accounts_for_longitude_convergence_at_high_latitude():
    """На 75 градусах северной широты 0.01 градуса долготы — это ~290 м, а не 1100 м."""
    dets = [det(75.0, 93.0, T0), det(75.0, 93.01, T0)]
    assert len(cluster_detections(dets, 500)) == 1
    assert len(cluster_detections(dets, 100)) == 2


def test_clustering_grid_is_consistent_across_latitude_bands():
    """Регрессия: шаг по долготе считался от широты каждой детекции.

    Две точки на одном меридиане в 800 м друг от друга получали чуть разный
    lon_step, попадали в колонки, отличающиеся на 2, и не находили друг друга
    при проверке соседей ±1. При радиусе 1000 м они обязаны слиться.
    """
    offset = 800 / M_PER_DEG
    dets = [det(61.0, 93.0, T0), det(61.0 + offset, 93.0, T0)]
    assert len(cluster_detections(dets, 1000)) == 1
    assert len(cluster_detections(dets, 500)) == 2


@pytest.mark.parametrize("lat", [51.0, 61.0, 70.0, 77.9])
def test_clustering_merges_a_pair_just_inside_the_radius_at_any_latitude(lat):
    """Тот же дефект мог прятаться на других широтах."""
    dets = [det(lat, 93.0, T0), det(lat + 400 / M_PER_DEG, 93.0, T0)]
    assert len(cluster_detections(dets, 500)) == 1
