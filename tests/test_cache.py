"""SPEC-14 acceptance tests — офлайн, кеш собирается из синтетического снимка."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import cache, firms
from src.api import Snapshot
from src.burn import SEVERITY_ORDER, BurnResult, BurnStatus, deferred
from src.cache import CacheError
from src.events import EventConfig, EventStatus, build_events
from src.persistence import PersistentSource
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def det(lat=61.0, lon=93.0, when=None, frp=5.0, sensor="VIIRS_SNPP_NRT"):
    return firms.Detection(latitude=lat, longitude=lon, brightness=330.0, frp=frp,
                           confidence="n", acquired_at=when or T0, sensor=sensor,
                           satellite="N", daynight="D")


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "region.toml"
    p.write_text('[region]\nname = "Тест"\nbbox = [82.0, 51.0, 109.0, 78.0]\n\n'
                 '[firms]\nsources = ["VIIRS_SNPP_NRT"]\nday_range = 2\n',
                 encoding="utf-8")
    return p


@pytest.fixture
def built(tmp_path, config):
    """Снимок, записанный на диск: parquet детекций и наблюдений + метаданные."""
    db = tmp_path / "source.db"
    dets = [det(when=T0), det(lat=61.01, when=T0 + timedelta(hours=6)),
            det(lat=55.0, lon=88.0, when=T0, frp=900.0)]
    store = Store(db)
    store.add_detections(dets)
    store.record_observation("VIIRS_SNPP_NRT", T0, T0 + timedelta(days=1),
                             status="ok", detection_count=3)
    events = build_events(dets, EventConfig())
    events[0].status = EventStatus.COMPLETE
    burns = {
        events[0].id: BurnResult(
            status=BurnStatus.OK, event_id=events[0].id,
            area_ha={k: 10.0 for k in SEVERITY_ORDER}, masked_fraction=0.05,
            thresholds_version="usgs-baseline-1", mgrs_tile="46VEH",
            doy_gap_days=20, enrichment=3.1, detections_checked=99),
        events[1].id: deferred(events[1].id, "полярная ночь", "2027-06-01", "v1"),
    }
    snap = Snapshot(region_name="Тест", bbox=(82.0, 51.0, 109.0, 78.0),
                    events=events, flares=[PersistentSource(
                        67.3, 83.2, 12, 145, 0.47, T0, T0 + timedelta(days=12))],
                    burns=burns, store=store, sources=["VIIRS_SNPP_NRT"])
    root = tmp_path / "cache"
    cache.save(snap, root, config)
    store.close()
    return snap, root, config


# --- AC-01: старт без единого сетевого запроса ---

def test_ac01_loading_needs_no_network(built, monkeypatch):
    """Любая попытка выйти в сеть при загрузке кеша должна упасть."""
    snap, root, config = built

    def forbidden(*_a, **_kw):
        raise AssertionError("загрузка кеша не должна обращаться к сети")

    monkeypatch.setattr("httpx.Client.get", forbidden, raising=False)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    restored = cache.load(root, config)
    assert len(restored.events) == len(snap.events)


# --- AC-02: загруженный снимок эквивалентен собранному ---

def test_ac02_events_survive_the_round_trip(built):
    snap, root, config = built
    restored = cache.load(root, config)
    assert {e.id for e in restored.events} == {e.id for e in snap.events}
    assert {e.status for e in restored.events} == {e.status for e in snap.events}


def test_ac02_detection_history_survives_the_round_trip(built):
    snap, root, config = built
    restored = cache.load(root, config)
    before = sorted(len(e.detections) for e in snap.events)
    after = sorted(len(e.detections) for e in restored.events)
    assert before == after


def test_ac02_burn_results_survive_including_validation_numbers(built):
    snap, root, config = built
    restored = cache.load(root, config)
    for eid, original in snap.burns.items():
        got = restored.burns[eid]
        assert got.status is original.status
        assert got.area_ha == original.area_ha
        assert got.masked_fraction == original.masked_fraction
        assert got.enrichment == original.enrichment
        assert got.validated == original.validated


def test_ac02_flares_survive_the_round_trip(built):
    snap, root, config = built
    restored = cache.load(root, config)
    assert len(restored.flares) == 1
    assert restored.flares[0].distinct_days == snap.flares[0].distinct_days


def test_ac02_api_serves_the_restored_snapshot(built):
    from fastapi.testclient import TestClient
    from src.api import create_app
    _, root, config = built
    client = TestClient(create_app(cache.load(root, config)))
    assert client.get("/api/v1/hotspots").json()["features"]
    assert client.get("/api/v1/burns").json()["features"]
    assert client.get("/api/v1/health").status_code == 200


# --- AC-03: отпечаток конфигурации ---

def test_ac03_changed_config_is_refused_not_silently_accepted(built):
    """Пороги влияют на события и на площади; молчаливая загрузка скрыла бы
    расхождение между тем, как построены события, и тем, как посчитаны гари."""
    _, root, config = built
    config.write_text(config.read_text() + '\n[confidence]\nmin_level = "high"\n',
                      encoding="utf-8")
    with pytest.raises(CacheError, match="другой конфигурации"):
        cache.load(root, config)


def test_ac03_same_config_loads(built):
    _, root, config = built
    assert cache.load(root, config) is not None


def test_ac03_fingerprint_changes_with_the_config():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.toml", Path(d) / "b.toml"
        # менять надо секцию, влияющую на производные данные: отпечаток
        # намеренно не реагирует на всё остальное
        a.write_text('[region]\nname = "А"\nbbox = [1.0, 2.0, 3.0, 4.0]\n')
        b.write_text('[region]\nname = "А"\nbbox = [1.0, 2.0, 9.0, 9.0]\n')
        assert cache.config_fingerprint(a) != cache.config_fingerprint(b)


# --- AC-04: версия схемы ---

def test_ac04_foreign_schema_version_is_refused(built):
    _, root, config = built
    payload = json.loads((root / "snapshot.json").read_text())
    payload["schema_version"] = 999
    (root / "snapshot.json").write_text(json.dumps(payload))
    with pytest.raises(CacheError, match="схемой версии 999"):
        cache.load(root, config)


# --- AC-05: отсутствие и порча кеша различимы ---

def test_ac05_missing_cache_is_reported_as_missing(tmp_path, config):
    with pytest.raises(CacheError, match="не найден"):
        cache.load(tmp_path / "нет", config)


def test_ac05_corrupt_cache_is_an_error_not_a_silent_fallback(built):
    """Тихий откат к сети превратил бы повреждённый кеш в 12-минутную паузу."""
    _, root, config = built
    (root / "snapshot.json").write_text("{не json")
    with pytest.raises(CacheError, match="повреждён"):
        cache.load(root, config)



def test_ac05_missing_parquet_is_reported(built):
    _, root, config = built
    (root / "detections.parquet").unlink()
    with pytest.raises(CacheError, match="отсутствует"):
        cache.load(root, config)


def test_ac05_unreadable_parquet_is_reported(built):
    _, root, config = built
    (root / "detections.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(CacheError, match="нечитаемы"):
        cache.load(root, config)


# --- AC-07: секреты в кеш не попадают ---

def test_ac07_cache_file_contains_no_secrets(built, monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "s3cr3t-map-key")
    _, root, _ = built
    assert "s3cr3t-map-key" not in (root / "snapshot.json").read_text(encoding="utf-8")


def test_ac07_real_environment_secrets_are_absent_from_the_cache(built):
    secrets = [v for k, v in os.environ.items()
               if k in ("FIRMS_MAP_KEY", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN") and v]
    if not secrets:
        pytest.skip("секреты не заданы в окружении")
    _, root, _ = built
    text = (root / "snapshot.json").read_text(encoding="utf-8")
    for s in secrets:
        assert s not in text


# --- размер и скорость: то, ради чего кеш существует ---

def test_parquet_cache_is_far_smaller_than_the_naive_format(built):
    """Замер на реальных данных: 933354 детекции заняли 14.5 МБ в Parquet+zstd
    против 268 МБ в SQLite и ещё 53 МБ в JSON со строковыми ключами событий."""
    _, root, _ = built
    total = sum(p.stat().st_size for p in root.iterdir())
    per_detection = total / 3          # три детекции в фикстуре
    assert per_detection < 4000, "накладные расходы на детекцию слишком велики"


def test_save_reports_measured_sizes(tmp_path, config, built):
    _, root, _ = built
    restored = cache.load(root, config)          # фикстура закрыла своё хранилище
    stats = cache.save(restored, tmp_path / "again", config)
    assert stats["detections"] == 3 and stats["events"] == 2
    assert stats["total_bytes"] > 0
    assert set(stats) >= {"detections_bytes", "observations_bytes",
                          "snapshot_bytes", "total_bytes"}


def test_detections_outside_any_event_are_kept(built):
    """Отсеянные SPEC-3 и SPEC-4 детекции нужны для /health: без них время
    последней детекции было бы неверным."""
    _, root, config = built
    restored = cache.load(root, config)
    in_events = sum(len(e.detections) for e in restored.events)
    assert restored.store.count_detections() >= in_events


@pytest.mark.parametrize("version", [2, 3])
def test_cache_written_before_the_fingerprint_rule_changed_is_refused(built, version):
    """В 4-й версии изменился СМЫСЛ отпечатка: он считается по секциям, влияющим
    на производные данные, а не по всему файлу. Отпечатки прежних версий
    несопоставимы, и принять их молча значило бы принять непроверенный отпечаток."""
    _, root, config = built
    meta_p = root / "snapshot.json"
    meta = json.loads(meta_p.read_text())
    meta["schema_version"] = version
    meta_p.write_text(json.dumps(meta, ensure_ascii=False))
    with pytest.raises(CacheError, match="читаемы"):
        cache.load(root, config)


def test_unreadable_cache_version_is_still_refused(built):
    _, root, config = built
    meta_p = root / "snapshot.json"
    meta = json.loads(meta_p.read_text())
    meta["schema_version"] = 1
    meta_p.write_text(json.dumps(meta, ensure_ascii=False))
    with pytest.raises(CacheError, match="читаемы"):
        cache.load(root, config)


def test_archive_markers_survive_the_round_trip(tmp_path, config):
    """SPEC-10: пометка Standard Processing и класс источника обязаны пережить кеш."""
    from src.archive import load_export
    from src.api import Snapshot
    from src.events import build_events, EventConfig
    dets = load_export(ROOT / "tests" / "fixtures" / "fire_archive_SV-C2_krasnoyarsk.csv")
    store = Store(":memory:"); store.add_detections(dets)
    snap = Snapshot(region_name="Тест", bbox=(82.0, 51.0, 109.0, 78.0),
                    events=build_events(dets, EventConfig()), flares=[], burns={},
                    store=store, sources=["VIIRS_SNPP_NRT"])
    root = tmp_path / "arch"
    cache.save(snap, root, config)
    restored = cache.load(root, config)
    all_dets = [d for e in restored.events for d in e.detections]
    assert all(d.is_archive for d in all_dets)
    assert 2 in {d.fire_type for d in all_dets}


def test_fingerprint_ignores_sections_that_do_not_affect_derived_data(tmp_path):
    """Регрессия: хеш считался по всему файлу, и добавление секции, которая
    настраивает лишь сверку с контрольными продуктами, обесценило собранный
    за 13 минут кеш — хотя ни события, ни площади от неё не зависят."""
    base = '[region]\nname = "Т"\nbbox = [82.0, 51.0, 109.0, 78.0]\n\n[firms]\nday_range = 2\n'
    a = tmp_path / "a.toml"; a.write_text(base, encoding="utf-8")
    b = tmp_path / "b.toml"; b.write_text(base + '\n[labels]\nproduct = "MCD64A1.061"\n', encoding="utf-8")
    assert cache.config_fingerprint(a) == cache.config_fingerprint(b)


def test_fingerprint_still_changes_when_a_threshold_changes(tmp_path):
    base = '[region]\nname = "Т"\nbbox = [82.0, 51.0, 109.0, 78.0]\n\n[firms]\nday_range = 2\n'
    a = tmp_path / "a.toml"; a.write_text(base, encoding="utf-8")
    for section in ('[confidence]\nmin_level = "high"\n', '[persistence]\nwindow_days = 30\n',
                    '[burn]\nmax_doy_gap_days = 90\n', '[events]\nradius_m = 9000\n'):
        b = tmp_path / "b.toml"; b.write_text(base + "\n" + section, encoding="utf-8")
        assert cache.config_fingerprint(a) != cache.config_fingerprint(b), section


def test_fingerprint_ignores_comments_and_key_order(tmp_path):
    a = tmp_path / "a.toml"
    a.write_text('[region]\nname = "Т"\nbbox = [1.0, 2.0, 3.0, 4.0]\n', encoding="utf-8")
    b = tmp_path / "b.toml"
    b.write_text('# пояснение\n[region]\nbbox = [1.0, 2.0, 3.0, 4.0]\nname = "Т"\n', encoding="utf-8")
    assert cache.config_fingerprint(a) == cache.config_fingerprint(b)
