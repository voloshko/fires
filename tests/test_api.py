"""SPEC-8 acceptance tests — офлайн, на синтетическом снимке данных.

Каждый тест назван по критерию приёмки из specs/SPEC-8-api-and-map.md.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import firms
from src.api import BurnPayload, Snapshot, create_app
from src.burn import SEVERITY_ORDER, BurnResult, BurnStatus, deferred
from src.events import build_events, EventConfig
from src.persistence import PersistentSource
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
M_PER_DEG = 111_320.0


def det(lat=61.0, lon=93.0, when=None, frp=5.0, sensor="VIIRS_SNPP_NRT"):
    return firms.Detection(latitude=lat, longitude=lon, brightness=330.0, frp=frp,
                           confidence="n", acquired_at=when or T0, sensor=sensor,
                           satellite="N", daynight="D")


def unvalidated_burn(event_id: str) -> BurnResult:
    """Площадь посчитана, но не совпала с местом горения (SPEC-9)."""
    return BurnResult(
        status=BurnStatus.UNVALIDATED, event_id=event_id,
        area_ha={"unburnt": 50.0, "low": 400.0, "moderate_low": 300.0,
                 "moderate_high": 20.0, "high": 0.0},
        masked_fraction=0.03, thresholds_version="usgs-baseline-1",
        mgrs_tile="48WXC", scene_before="S2_june", scene_after="S2_sept",
        scene_before_date="2026-06-28", scene_after_date="2026-09-06",
        doy_gap_days=70, enrichment=0.25, detections_checked=850,
        reason="площадь не совпала с местом горения: обогащение 0.25 при минимуме 1.50")


def ok_burn(event_id: str) -> BurnResult:
    return BurnResult(
        status=BurnStatus.OK, event_id=event_id,
        area_ha={"unburnt": 100.0, "low": 20.0, "moderate_low": 10.0,
                 "moderate_high": 4.0, "high": 1.0},
        masked_fraction=0.07, thresholds_version="usgs-baseline-1",
        mgrs_tile="46VEH", scene_before="S2_before", scene_after="S2_after",
        scene_before_date="2026-08-14", scene_after_date="2026-09-01",
        doy_gap_days=18, enrichment=3.4, detections_checked=140)


@pytest.fixture
def store():
    with Store(":memory:") as s:
        yield s


@pytest.fixture
def snap(store):
    dets = [det(when=T0), det(lat=61.01, when=T0 + timedelta(hours=6)),
            det(lat=55.0, lon=88.0, when=T0, frp=900.0)]
    store.add_detections(dets)
    events = build_events(dets, EventConfig())
    burned, other = events[0], events[1]
    return Snapshot(
        region_name="Красноярский край", bbox=(82.0, 51.0, 109.0, 78.0),
        events=events,
        flares=[PersistentSource(67.3, 83.2, 12, 145, 0.47, T0, T0 + timedelta(days=12))],
        burns={burned.id: ok_burn(burned.id),
               other.id: deferred(other.id, "полярная ночь", "2027-06-01",
                                  "usgs-baseline-1"),
               "FIRE-unvalidated0": unvalidated_burn("FIRE-unvalidated0")},
        store=store, sources=["VIIRS_SNPP_NRT", "MODIS_NRT"])


@pytest.fixture
def client(snap):
    return TestClient(create_app(snap))


def burned_event(snap):
    return next(k for k, v in snap.burns.items() if v.status is BurnStatus.OK)


def deferred_event(snap):
    return next(k for k, v in snap.burns.items() if v.status is BurnStatus.DEFERRED)


def unvalidated_event(snap):
    return next(k for k, v in snap.burns.items()
                if v.status is BurnStatus.UNVALIDATED)


# --- AC-01: площадь не отдаётся без контекста ---

REQUIRED_CONTEXT = ("masked_fraction", "thresholds_version", "status")


def test_ac01_burn_detail_always_carries_the_required_context(client, snap):
    p = client.get(f"/api/v1/burns/{burned_event(snap)}").json()
    assert "area_ha" in p
    for field in REQUIRED_CONTEXT:
        assert field in p and p[field] is not None, f"нет поля {field}"


def test_ac01_burn_collection_also_carries_the_context(client):
    fc = client.get("/api/v1/burns").json()
    assert fc["features"]
    for f in fc["features"]:
        for field in REQUIRED_CONTEXT:
            assert field in f["properties"]


@pytest.mark.parametrize("missing", REQUIRED_CONTEXT)
def test_ac01_schema_refuses_a_payload_without_any_context_field(missing):
    """Обязательность — часть схемы, а не соглашение вызывающего."""
    payload = dict(
        event_id="E", status="ok",
        area_ha={k: 1.0 for k in SEVERITY_ORDER}, burned_ha=4.0,
        masked_fraction=0.1, thresholds_version="v1", validated=False)
    payload.pop(missing)
    with pytest.raises(Exception):
        BurnPayload(**payload)


def test_ac01_masked_fraction_outside_zero_one_is_refused():
    base = dict(event_id="E", status="ok", area_ha={k: 1.0 for k in SEVERITY_ORDER},
                burned_ha=4.0, thresholds_version="v1", validated=False)
    with pytest.raises(Exception):
        BurnPayload(masked_fraction=1.5, **base)


def test_ac01_stats_endpoint_always_reports_its_validation_state(client):
    """Агрегат тоже не отдаёт гектары без указания, проверены ли они."""
    s = client.get("/api/v1/stats/area").json()
    assert "burned_ha" in s
    assert "validated" in s and isinstance(s["validated"], bool)
    assert "events_unvalidated" in s


def test_ac01_validation_flag_comes_from_the_result_not_from_a_constant(client, snap):
    """SPEC-9: проверенный результат и непроверенный обязаны различаться в ответе."""
    good = client.get(f"/api/v1/burns/{burned_event(snap)}").json()
    bad = client.get(f"/api/v1/burns/{unvalidated_event(snap)}").json()
    assert good["validated"] is True and good["validation_note"] is None
    assert bad["validated"] is False and bad["validation_note"]
    assert bad["enrichment"] == pytest.approx(0.25)


# --- AC-02: health различает две величины ---

def test_ac02_health_separates_successful_poll_from_data_received(store, snap):
    """Опросы идут, детекций нет — это не одно и то же."""
    store.record_observation("MODIS_NRT", T0, T0 + timedelta(days=1),
                             status="ok", detection_count=0,
                             requested_at=T0 + timedelta(days=1))
    client = TestClient(create_app(snap))
    rows = {r["source"]: r for r in client.get("/api/v1/health").json()["sources"]}
    modis = rows["MODIS_NRT"]
    assert modis["last_successful_poll"] is not None
    assert modis["last_detection_received"] is None
    assert modis["last_successful_poll"] != modis["last_detection_received"]


def test_ac02_health_lists_every_configured_source(client):
    rows = client.get("/api/v1/health").json()["sources"]
    assert {r["source"] for r in rows} == {"VIIRS_SNPP_NRT", "MODIS_NRT"}


def test_ac02_health_counts_failures_separately_from_observations(store, snap):
    store.record_observation("MODIS_NRT", T0, T0 + timedelta(days=1),
                             status="failed", detection_count=0, error="503")
    client = TestClient(create_app(snap))
    rows = {r["source"]: r for r in client.get("/api/v1/health").json()["sources"]}
    assert rows["MODIS_NRT"]["failures"] == 1
    assert rows["MODIS_NRT"]["observations"] == 0


def test_ac02_health_reports_validation_state_of_burn_results(client):
    assert client.get("/api/v1/health").json()["burn_results_validated"] is False


# --- AC-03: служебный слой теплоисточников ---

def test_ac03_flares_endpoint_exposes_the_service_layer(client):
    fc = client.get("/api/v1/flares").json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 1
    p = fc["features"][0]["properties"]
    assert p["distinct_days"] == 12 and p["detections"] == 145


def test_ac03_flares_are_not_mixed_into_the_hotspots_layer(client):
    hot = client.get("/api/v1/hotspots").json()
    coords = {tuple(f["geometry"]["coordinates"]) for f in hot["features"]}
    assert (83.2, 67.3) not in coords


# --- AC-04: deferred не выглядит как нулевая площадь ---

def test_ac04_deferred_event_reports_a_reason_not_a_zero(client, snap):
    p = client.get(f"/api/v1/burns/{deferred_event(snap)}").json()
    assert p["status"] == "deferred"
    assert p["reason"] and p["retry_after"] == "2027-06-01"


def test_ac04_deferred_and_computed_results_are_distinguishable(client, snap):
    a = client.get(f"/api/v1/burns/{burned_event(snap)}").json()
    b = client.get(f"/api/v1/burns/{deferred_event(snap)}").json()
    assert a["status"] != b["status"]
    assert a["masked_fraction"] != b["masked_fraction"]


def test_ac04_deferred_events_are_not_counted_in_aggregate_area(client):
    s = client.get("/api/v1/stats/area").json()
    assert s["events_counted"] == 1 and s["events_without_result"] == 1
    assert s["burned_ha"] == pytest.approx(35.0)      # только из завершённого


def test_ac04_health_counts_deferred_events(client):
    assert client.get("/api/v1/health").json()["events_deferred"] == 1


# --- AC-06: секреты не попадают в ответы ---

ENDPOINTS = ["/api/v1/hotspots", "/api/v1/burns", "/api/v1/flares",
             "/api/v1/health", "/api/v1/stats/area"]


def test_ac06_no_endpoint_leaks_the_map_key(client, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "s3cr3t-map-key")
    for url in ENDPOINTS:
        body = json.dumps(client.get(url).json(), ensure_ascii=False)
        assert "s3cr3t-map-key" not in body, f"ключ утёк в {url}"


def test_ac06_no_endpoint_leaks_earthdata_credentials(client, monkeypatch):
    monkeypatch.setenv("EARTHDATA_PASSWORD", "p@ssw0rd-secret")
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok3n-secret")
    for url in ENDPOINTS:
        body = json.dumps(client.get(url).json(), ensure_ascii=False)
        assert "p@ssw0rd-secret" not in body and "tok3n-secret" not in body


def test_ac06_real_environment_secrets_are_absent_from_responses(client):
    """Если ключи реально заданы в окружении — их тоже не должно быть в ответах."""
    secrets = [v for k, v in os.environ.items()
               if k in ("FIRMS_MAP_KEY", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN") and v]
    if not secrets:
        pytest.skip("секреты не заданы в окружении")
    for url in ENDPOINTS:
        body = json.dumps(client.get(url).json(), ensure_ascii=False)
        for s in secrets:
            assert s not in body


# --- прочие эндпоинты ---

def test_hotspots_returns_geojson_feature_collection(client):
    fc = client.get("/api/v1/hotspots").json()
    assert fc["type"] == "FeatureCollection"
    assert all(f["geometry"]["type"] == "Point" for f in fc["features"])
    assert all("event_id" in f["properties"] for f in fc["features"])


def test_hotspots_bbox_filter_narrows_the_result(client):
    everything = client.get("/api/v1/hotspots").json()["features"]
    narrow = client.get("/api/v1/hotspots?bbox=92,60,94,62").json()["features"]
    assert 0 < len(narrow) < len(everything)


def test_hotspots_malformed_bbox_is_rejected(client):
    assert client.get("/api/v1/hotspots?bbox=1,2,3").status_code == 400


def test_event_detail_returns_full_chronological_history(client, snap):
    ev = client.get(f"/api/v1/hotspots/{burned_event(snap)}").json()
    times = [f["properties"]["acq_datetime"] for f in ev["history"]["features"]]
    assert times == sorted(times) and len(times) == 2


def test_unknown_event_returns_404(client):
    assert client.get("/api/v1/hotspots/FIRE-000000000000").status_code == 404
    assert client.get("/api/v1/burns/FIRE-000000000000").status_code == 404


def test_unknown_severity_class_is_rejected(client):
    assert client.get("/api/v1/burns?severity_class=catastrophic").status_code == 400


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "leaflet" in r.text.lower()


def test_map_page_renders_both_layers_and_the_report_table():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for needed in ("/api/v1/hotspots", "/api/v1/burns", "/api/v1/flares",
                   "/api/v1/health", "/api/v1/stats/area"):
        assert needed in html, f"карта не обращается к {needed}"
    assert "exportData" in html and "csv" in html.lower()
    assert "validation_note" in html


def test_map_basemap_needs_no_api_key():
    """Регрессия: тайлы CARTO начали требовать ключ, и подложка отдавала
    плитки с надписью API KEY REQUIRED. Публичный сервис не должен зависеть
    от чужой регистрации (REQ-005)."""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "cartocdn" not in html
    assert "tile.openstreetmap.org" in html
    assert "OpenStreetMap contributors" in html


def test_map_shows_the_validation_numbers_next_to_the_hectares():
    """SPEC-9: обогащение и разрыв по дню года — часть контекста площади,
    и пользователь обязан видеть их рядом с числом гектаров."""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "burn.enrichment" in html
    assert "burn.doy_gap_days" in html
    assert "detections_checked" in html


def test_snapshot_burn_selection_skips_events_that_cannot_be_mapped():
    """Регрессия: отбор по одному FRP тратил бюджет расчётов на события,
    которые шире max_aoi_km или ещё не отпустили окно поиска сцены «после».
    Пять расчётов подряд возвращали deferred/failed."""
    import inspect
    from src import api
    src = inspect.getsource(api.build_snapshot)
    assert "mappable" in src
    assert "aoi_extent_km" in src and "post_window_days" in src
