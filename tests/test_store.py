"""SPEC-2 acceptance tests — офлайн, sqlite in-memory.

Каждый тест назван по критерию приёмки из specs/SPEC-2-detection-store.md.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import firms
from src.store import Coverage, Store

FIXTURES = Path(__file__).resolve().parent / "fixtures"
T0 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


def viirs_fixture() -> list[firms.Detection]:
    text = (FIXTURES / "firms_viirs_nonempty.csv").read_text(encoding="utf-8")
    return firms.parse_csv(text, "VIIRS_SNPP_NRT")


def det(**kw) -> firms.Detection:
    base = dict(latitude=61.0, longitude=93.0, brightness=330.0, frp=5.0,
                confidence="n", acquired_at=T0, sensor="VIIRS_SNPP_NRT",
                satellite="N", daynight="D")
    base.update(kw)
    return firms.Detection(**base)


# --- AC-01: повторная загрузка того же окна не создаёт дублей ---

def test_ac01_loading_the_same_fixture_twice_adds_nothing(store):
    dets = viirs_fixture()
    assert store.add_detections(dets) == 3
    assert store.count_detections() == 3

    assert store.add_detections(dets) == 0        # второй проход — ноль новых
    assert store.count_detections() == 3


def test_ac01_overlapping_windows_only_add_the_genuinely_new(store):
    """FIRMS опрашивается с перекрытием окон — это штатный режим SPEC-1."""
    first = viirs_fixture()
    store.add_detections(first)

    overlapping = first + [det(latitude=62.5, longitude=94.5,
                               acquired_at=T0 + timedelta(hours=3))]
    assert store.add_detections(overlapping) == 1
    assert store.count_detections() == 4


# --- AC-02: ключ уникальности (сенсор, широта, долгота, время) ---

def test_ac02_same_point_from_different_sensors_is_kept_twice(store):
    """Два аппарата видят один пожар — это две разные детекции, не дубль."""
    store.add_detections([
        det(sensor="VIIRS_SNPP_NRT"),
        det(sensor="VIIRS_NOAA20_NRT"),
    ])
    assert store.count_detections() == 2


@pytest.mark.parametrize("field,value", [
    ("latitude", 61.00001),
    ("longitude", 93.00001),
    ("acquired_at", T0 + timedelta(minutes=1)),
    ("sensor", "MODIS_NRT"),
])
def test_ac02_any_key_field_differing_makes_a_new_row(store, field, value):
    store.add_detections([det()])
    assert store.add_detections([det(**{field: value})]) == 1
    assert store.count_detections() == 2


def test_ac02_non_key_fields_do_not_create_duplicates(store):
    """FRP уточняется между NRT и SP; это тот же пожар, а не второй."""
    store.add_detections([det(frp=5.0)])
    assert store.add_detections([det(frp=99.0)]) == 0
    assert store.count_detections() == 1


# --- AC-03: успешный пустой опрос ≠ несостоявшийся опрос ---

def test_ac03_successful_empty_poll_is_recorded(store):
    store.record_observation("VIIRS_SNPP_NRT", T0 - timedelta(days=2), T0,
                             status="ok", detection_count=0)
    cov = store.coverage("VIIRS_SNPP_NRT", T0 - timedelta(days=1), T0)
    assert cov.observed is True
    assert cov.detections == 0


def test_ac03_a_poll_that_never_happened_leaves_no_trace(store):
    cov = store.coverage("VIIRS_SNPP_NRT", T0 - timedelta(days=1), T0)
    assert cov.observed is False
    assert cov.observations == 0


def test_ac03_failed_poll_is_not_counted_as_an_observation(store):
    """Источник упал — это НЕ наблюдение «огня нет»."""
    store.record_observation("VIIRS_SNPP_NRT", T0 - timedelta(days=2), T0,
                             status="failed", detection_count=0,
                             error="HTTPStatusError: 503")
    cov = store.coverage("VIIRS_SNPP_NRT", T0 - timedelta(days=1), T0)
    assert cov.observed is False
    assert cov.failures == 1


def test_ac03_empty_poll_and_failed_poll_are_distinguishable(store):
    """Ровно то различие, ради которого журнал существует."""
    store.record_observation("A", T0 - timedelta(days=2), T0,
                             status="ok", detection_count=0)
    store.record_observation("B", T0 - timedelta(days=2), T0,
                             status="failed", detection_count=0,
                             error="boom")
    empty = store.coverage("A", T0 - timedelta(days=1), T0)
    broken = store.coverage("B", T0 - timedelta(days=1), T0)
    assert empty != broken
    assert (empty.observed, broken.observed) == (True, False)


def test_ac03_invalid_status_is_rejected(store):
    with pytest.raises(ValueError, match="status must be"):
        store.record_observation("A", T0, T0, status="maybe", detection_count=0)


# --- AC-04: три случая ответа на «были ли наблюдения в окне» ---

def test_ac04_three_cases(store):
    window = (T0 - timedelta(days=2), T0)
    # detections считается по таблице детекций, поэтому для случая «есть огонь»
    # нужны настоящие детекции, а не только счётчик в журнале
    store.add_detections([
        det(sensor="WITH_FIRE", latitude=61.0 + i / 100, acquired_at=T0 - timedelta(hours=1))
        for i in range(7)
    ])
    store.record_observation("WITH_FIRE", *window, status="ok", detection_count=7)
    store.record_observation("NO_FIRE", *window, status="ok", detection_count=0)
    # NEVER_POLLED намеренно не записан

    assert store.coverage("WITH_FIRE", *window) == Coverage(
        observed=True, observations=1, detections=7, failures=0)
    assert store.coverage("NO_FIRE", *window) == Coverage(
        observed=True, observations=1, detections=0, failures=0)
    assert store.coverage("NEVER_POLLED", *window) == Coverage(
        observed=False, observations=0, detections=0, failures=0)


def test_ac04_observation_outside_the_asked_window_does_not_count(store):
    store.record_observation("S", T0 - timedelta(days=30),
                             T0 - timedelta(days=28),
                             status="ok", detection_count=3)
    assert store.coverage("S", T0 - timedelta(days=1), T0).observed is False


def test_ac04_partially_overlapping_observation_counts(store):
    store.record_observation("S", T0 - timedelta(days=3), T0 - timedelta(days=1),
                             status="ok", detection_count=2)
    assert store.coverage("S", T0 - timedelta(days=2), T0).observed is True


# --- AC-05: append-only ---

def test_ac05_updating_a_stored_detection_is_refused(store):
    store.add_detections([det()])
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("UPDATE detections SET frp = 999")


def test_ac05_deleting_a_stored_detection_is_refused(store):
    store.add_detections([det()])
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM detections")
    assert store.count_detections() == 1


def test_ac05_observations_are_append_only_too(store):
    store.record_observation("S", T0, T0, status="ok", detection_count=0)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("UPDATE observations SET status = 'failed'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM observations")


# --- интеграция с SPEC-1 ---

def test_record_fetch_writes_detections_and_one_journal_row_per_source(store):
    result = firms.FetchResult(
        detections=viirs_fixture(),
        failures={"MODIS_NRT": "HTTPStatusError: 503"},
    )
    sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]
    new = store.record_fetch(result, sources, T0 - timedelta(days=2), T0)

    assert new == 3
    window = (T0 - timedelta(days=2), T0)
    # Сохранены все три детекции, но в окно [T0-2d, T0] попадают только две:
    # третья строка фикстуры снята в 19:34, то есть позже T0=12:00. Окно опроса
    # и время наблюдения — разные величины, и coverage считает вторую.
    assert store.count_detections() == 3
    assert store.coverage("VIIRS_SNPP_NRT", *window).detections == 2
    # источник опрошен успешно, но детекций не дал
    assert store.coverage("VIIRS_NOAA20_NRT", *window) == Coverage(
        observed=True, observations=1, detections=0, failures=0)
    # источник упал — наблюдением не считается
    assert store.coverage("MODIS_NRT", *window).observed is False


def test_record_fetch_is_idempotent_for_detections(store):
    result = firms.FetchResult(detections=viirs_fixture())
    sources = ["VIIRS_SNPP_NRT"]
    window = (T0 - timedelta(days=2), T0)
    assert store.record_fetch(result, sources, *window) == 3
    assert store.record_fetch(result, sources, *window) == 0
    assert store.count_detections() == 3
    # но журнал фиксирует ОБА опроса: это разные наблюдения
    assert store.coverage("VIIRS_SNPP_NRT", *window).observations == 2


# --- вход для SPEC-4 / SPEC-6 ---

def test_detections_between_returns_history_in_order(store):
    store.add_detections([
        det(latitude=61.0, acquired_at=T0 + timedelta(hours=2)),
        det(latitude=61.1, acquired_at=T0),
        det(latitude=61.2, acquired_at=T0 + timedelta(hours=1)),
    ])
    rows = store.detections_between(T0 - timedelta(hours=1), T0 + timedelta(hours=3))
    assert [r["latitude"] for r in rows] == [61.1, 61.2, 61.0]


def test_naive_datetime_is_refused(store):
    with pytest.raises(ValueError, match="naive datetime"):
        store.add_detections([det(acquired_at=datetime(2026, 9, 16, 12, 0))])


def test_persists_across_reopen(tmp_path):
    db = tmp_path / "fires.db"
    with Store(db) as s:
        s.add_detections(viirs_fixture())
    with Store(db) as s:
        assert s.count_detections() == 3


def test_overlapping_polls_do_not_double_count_detections(store):
    """Регрессия: coverage() суммировал счётчики журнала и удваивал результат.

    Дефект найден живым прогоном (1858 реальных детекций показывались как 3716),
    на фикстурах не проявлялся. Перекрытие окон — штатный режим SPEC-1.
    """
    result = firms.FetchResult(detections=viirs_fixture())
    sources = ["VIIRS_SNPP_NRT"]
    window = (T0 - timedelta(days=2), T0 + timedelta(days=1))

    store.record_fetch(result, sources, *window)
    store.record_fetch(result, sources, *window)   # тот же ответ, второй опрос

    cov = store.coverage("VIIRS_SNPP_NRT", *window)
    assert cov.observations == 2      # опросов действительно два
    assert cov.detections == 3        # но детекций по-прежнему три, не шесть
    assert store.count_detections() == 3


def test_coverage_counts_only_its_own_sensor(store):
    window = (T0 - timedelta(days=1), T0 + timedelta(days=1))
    store.add_detections([det(sensor="VIIRS_SNPP_NRT"),
                          det(sensor="MODIS_NRT")])
    store.record_observation("VIIRS_SNPP_NRT", *window, status="ok",
                             detection_count=1)
    assert store.coverage("VIIRS_SNPP_NRT", *window).detections == 1


def test_store_is_readable_from_another_thread():
    """Регрессия SPEC-8 DEF-01: FastAPI выполняет синхронные обработчики в пуле
    потоков, и /health читал журнал из рабочего потока. sqlite3 по умолчанию
    привязывает соединение к создавшему его потоку — сервер падал бы."""
    import threading
    with Store(":memory:") as s:
        s.add_detections(viirs_fixture())
        result = {}

        def read():
            try:
                result["count"] = s.count_detections()
            except Exception as exc:
                result["error"] = exc

        t = threading.Thread(target=read)
        t.start(); t.join()
        assert "error" not in result, result.get("error")
        assert result["count"] == 3
