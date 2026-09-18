"""SPEC-6 acceptance tests.

Каждый тест назван по критерию приёмки из specs/SPEC-6-event-clustering.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import firms
from src.events import (Event, EventConfig, EventStatus, build_events,
                        evaluate_completion, load_event_config)
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "krasnoyarsk.toml"
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
M_PER_DEG = 111_320.0
CFG = EventConfig(radius_m=3000, time_gap_hours=48, completion_days=7,
                  min_observations=1)
SOURCES = ["VIIRS_SNPP_NRT"]


def det(lat=61.0, lon=93.0, when=None, frp=5.0, sensor="VIIRS_SNPP_NRT"):
    return firms.Detection(latitude=lat, longitude=lon, brightness=330.0, frp=frp,
                           confidence="n", acquired_at=when or T0, sensor=sensor,
                           satellite="N", daynight="D")


def km(m: float) -> float:
    return m * 1000 / M_PER_DEG


# --- AC-01: пороги расстояния и времени ---

def test_ac01_nearby_and_close_in_time_become_one_event():
    dets = [det(when=T0), det(lat=61.0 + km(1), when=T0 + timedelta(hours=6))]
    assert len(build_events(dets, CFG)) == 1


def test_ac01_far_apart_in_space_become_separate_events():
    dets = [det(when=T0), det(lat=61.0 + km(50), when=T0 + timedelta(hours=6))]
    assert len(build_events(dets, CFG)) == 2


def test_ac01_far_apart_in_time_become_separate_events():
    """Та же точка, но через 10 суток — это новый пожар, а не продолжение."""
    dets = [det(when=T0), det(when=T0 + timedelta(days=10))]
    assert len(build_events(dets, CFG)) == 2


def test_ac01_thresholds_come_from_config():
    c = load_event_config(CONFIG)
    assert (c.radius_m, c.time_gap_hours, c.completion_days) == (3000.0, 48.0, 7)


def test_ac01_changing_the_radius_changes_the_grouping():
    dets = [det(when=T0), det(lat=61.0 + km(5), when=T0 + timedelta(hours=6))]
    assert len(build_events(dets, EventConfig(radius_m=3000))) == 2
    assert len(build_events(dets, EventConfig(radius_m=10000))) == 1


def test_ac01_changing_the_time_gap_changes_the_grouping():
    dets = [det(when=T0), det(when=T0 + timedelta(hours=60))]
    assert len(build_events(dets, EventConfig(time_gap_hours=48))) == 2
    assert len(build_events(dets, EventConfig(time_gap_hours=72))) == 1


def test_ac01_a_detection_can_merge_two_existing_events():
    """Фронт разошёлся и снова сомкнулся — это один пожар, а не три."""
    a = det(lat=61.0, when=T0)
    b = det(lat=61.0 + km(5), when=T0 + timedelta(hours=1))
    bridge = det(lat=61.0 + km(2.5), when=T0 + timedelta(hours=2))
    assert len(build_events([a, b], CFG)) == 2
    assert len(build_events([a, b, bridge], CFG)) == 1


def test_ac01_detections_from_different_sensors_join_the_same_event():
    """Два аппарата видят один пожар — одно событие."""
    dets = [det(sensor="VIIRS_SNPP_NRT", when=T0),
            det(lat=61.0 + km(1), sensor="MODIS_NRT", when=T0 + timedelta(hours=2))]
    events = build_events(dets, CFG)
    assert len(events) == 1
    assert events[0].sensors == ["MODIS_NRT", "VIIRS_SNPP_NRT"]


# --- AC-02: устойчивый идентификатор ---

def test_ac02_id_survives_three_rounds_of_new_detections():
    dets = [det(when=T0)]
    first_id = build_events(dets, CFG)[0].id
    for i in range(1, 4):
        dets.append(det(lat=61.0 + km(i), when=T0 + timedelta(hours=6 * i)))
        events = build_events(dets, CFG)
        assert len(events) == 1
        assert events[0].id == first_id, f"id изменился на {i}-м приёме"


def test_ac02_different_events_get_different_ids():
    dets = [det(when=T0), det(lat=61.0 + km(50), when=T0)]
    ids = {e.id for e in build_events(dets, CFG)}
    assert len(ids) == 2


def test_ac02_id_is_deterministic_across_runs():
    dets = [det(when=T0), det(lat=61.0 + km(1), when=T0 + timedelta(hours=3))]
    assert build_events(dets, CFG)[0].id == build_events(dets, CFG)[0].id


def test_ac02_id_looks_like_an_identifier():
    e = build_events([det(when=T0)], CFG)[0]
    assert e.id.startswith("FIRE-") and len(e.id) == 17


# --- AC-03: завершение только при подтверждённых наблюдениях ---

@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


def test_ac03_seven_days_without_any_polls_does_not_complete_the_event(store):
    """Опросов не было — сказать, что пожар потух, нельзя."""
    events = build_events([det(when=T0)], CFG)
    now = T0 + timedelta(days=8)
    evaluate_completion(events, store, SOURCES, CFG, now=now)
    assert events[0].status is EventStatus.UNCONFIRMED
    assert events[0].status is not EventStatus.COMPLETE


def test_ac03_seven_days_of_successful_empty_polls_completes_the_event(store):
    events = build_events([det(when=T0)], CFG)
    now = T0 + timedelta(days=8)
    for day in range(8):
        t = T0 + timedelta(days=day)
        store.record_observation("VIIRS_SNPP_NRT", t, t + timedelta(days=1),
                                 status="ok", detection_count=0)
    evaluate_completion(events, store, SOURCES, CFG, now=now)
    assert events[0].status is EventStatus.COMPLETE


def test_ac03_failed_polls_do_not_complete_the_event(store):
    """Источник падал все семь суток — это не наблюдение «огня нет»."""
    events = build_events([det(when=T0)], CFG)
    now = T0 + timedelta(days=8)
    for day in range(8):
        t = T0 + timedelta(days=day)
        store.record_observation("VIIRS_SNPP_NRT", t, t + timedelta(days=1),
                                 status="failed", detection_count=0, error="503")
    evaluate_completion(events, store, SOURCES, CFG, now=now)
    assert events[0].status is EventStatus.UNCONFIRMED


def test_ac03_event_with_recent_detections_stays_active(store):
    events = build_events([det(when=T0)], CFG)
    evaluate_completion(events, store, SOURCES, CFG, now=T0 + timedelta(days=2))
    assert events[0].status is EventStatus.ACTIVE


def test_ac03_three_statuses_are_distinct():
    assert len({EventStatus.ACTIVE, EventStatus.COMPLETE,
                EventStatus.UNCONFIRMED}) == 3


def test_ac03_completion_days_comes_from_config(store):
    events = build_events([det(when=T0)], CFG)
    for day in range(30):
        t = T0 + timedelta(days=day)
        store.record_observation("VIIRS_SNPP_NRT", t, t + timedelta(days=1),
                                 status="ok", detection_count=0)
    now = T0 + timedelta(days=10)
    evaluate_completion(events, store, SOURCES, EventConfig(completion_days=7), now=now)
    assert events[0].status is EventStatus.COMPLETE
    evaluate_completion(events, store, SOURCES, EventConfig(completion_days=20), now=now)
    assert events[0].status is EventStatus.ACTIVE


# --- AC-04: bbox и суммарный FRP как вход для SPEC-7 ---

def test_ac04_event_exposes_bbox_covering_all_its_detections():
    dets = [det(lat=61.0, lon=93.0, when=T0),
            det(lat=61.0 + km(1), lon=93.02, when=T0 + timedelta(hours=3))]
    events = build_events(dets, CFG)
    assert len(events) == 1, "точки должны попасть в одно событие"
    west, south, east, north = events[0].bbox
    assert (west, south) == (93.0, 61.0)
    assert east == 93.02 and north > 61.0


def test_ac04_bbox_covers_every_detection_of_a_merged_event():
    """bbox не должен потерять ветку после слияния событий."""
    dets = [det(lat=61.0, lon=93.0, when=T0),
            det(lat=61.0 + km(5), lon=93.0, when=T0 + timedelta(hours=1)),
            det(lat=61.0 + km(2.5), lon=93.0, when=T0 + timedelta(hours=2))]
    events = build_events(dets, CFG)
    assert len(events) == 1
    _, south, _, north = events[0].bbox
    assert south == 61.0
    assert north == pytest.approx(61.0 + km(5))


def test_ac04_event_exposes_total_and_max_frp():
    dets = [det(when=T0, frp=10.0),
            det(lat=61.0 + km(1), when=T0 + timedelta(hours=2), frp=250.0)]
    e = build_events(dets, CFG)[0]
    assert e.total_frp == 260.0
    assert e.max_frp == 250.0


def test_ac04_missing_frp_does_not_break_the_sum():
    dets = [det(when=T0, frp=None), det(lat=61.0 + km(1),
                                        when=T0 + timedelta(hours=2), frp=7.0)]
    e = build_events(dets, CFG)[0]
    assert e.total_frp == 7.0


def test_ac04_padded_bbox_is_wider_than_the_raw_one():
    """Периметр гари шире облака термоточек — SPEC-7 ищет сцены с запасом."""
    e = build_events([det(when=T0)], CFG)[0]
    w, s, x, n = e.bbox
    pw, ps, px, pn = e.padded_bbox(5000)
    assert pw < w and ps < s and px > x and pn > n


def test_ac04_single_detection_event_has_degenerate_but_valid_bbox():
    e = build_events([det(when=T0)], CFG)[0]
    w, s, x, n = e.bbox
    assert (w, s) == (x, n) == (93.0, 61.0)
    assert e.padded_bbox(1000)[0] < w      # но с запасом уже не вырожден


# --- AC-05: история целиком и по порядку ---

def test_ac05_history_is_complete_and_chronological():
    times = [T0 + timedelta(hours=h) for h in (10, 2, 30, 6)]
    dets = [det(lat=61.0 + km(i * 0.5), when=t) for i, t in enumerate(times)]
    e = build_events(dets, CFG)[0]
    assert len(e.history) == 4
    assert [d.acquired_at for d in e.history] == sorted(times)


def test_ac05_first_and_last_seen_match_the_history():
    times = [T0 + timedelta(hours=h) for h in (10, 2, 30)]
    dets = [det(lat=61.0 + km(i * 0.5), when=t) for i, t in enumerate(times)]
    e = build_events(dets, CFG)[0]
    assert e.first_seen == e.history[0].acquired_at
    assert e.last_seen == e.history[-1].acquired_at
    assert e.duration_days == pytest.approx(28 / 24)


def test_ac05_nothing_is_lost_during_a_merge():
    a = det(lat=61.0, when=T0)
    b = det(lat=61.0 + km(5), when=T0 + timedelta(hours=1))
    bridge = det(lat=61.0 + km(2.5), when=T0 + timedelta(hours=2))
    events = build_events([a, b, bridge], CFG)
    assert len(events) == 1 and len(events[0].history) == 3


def test_empty_input_yields_no_events():
    assert build_events([], CFG) == []
