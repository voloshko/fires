"""SPEC-10 acceptance tests — офлайн, на фикстурах архивного формата."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from src import archive, firms
from src.archive import (FIRE_TYPE_STATIC_LAND, ArchiveError, NrtDepth,
                         load_export, probe_nrt_depth, sensor_from_name, summarise)
from src.confidence import Level, normalize
from src.events import EventConfig, build_events
from src.persistence import PersistenceConfig, analyse
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
VIIRS = FIX / "fire_archive_SV-C2_krasnoyarsk.csv"
MODIS = FIX / "fire_archive_M-C61_krasnoyarsk.csv"


# --- AC-01: схема выгрузки приводится к Detection ---

def test_ac01_viirs_export_parses_into_detections():
    dets = load_export(VIIRS)
    assert len(dets) == 4
    d = dets[0]
    assert d.latitude == pytest.approx(61.31742)
    assert d.brightness == pytest.approx(330.5)      # из bright_ti4
    assert d.sensor == "VIIRS_SNPP_NRT"
    assert d.acquired_at.tzinfo is timezone.utc


def test_ac01_modis_export_parses_with_its_own_brightness_column():
    dets = load_export(MODIS)
    assert len(dets) == 2
    assert dets[0].brightness == pytest.approx(325.8)   # из brightness
    assert dets[0].sensor == "MODIS_SP"


def test_ac01_confidence_scales_still_differ_between_sensors():
    """Ровно то различие, которое SPEC-3 разводит: архив его не устраняет."""
    viirs, modis = load_export(VIIRS)[0], load_export(MODIS)[0]
    assert viirs.confidence == "n"          # категориальная
    assert modis.confidence == "78"         # числовая
    assert normalize(viirs.sensor, viirs.confidence) is Level.NOMINAL
    assert normalize(modis.sensor, modis.confidence) is Level.NOMINAL


def test_ac01_zip_bundle_is_unpacked_and_sensors_inferred_per_member():
    dets = load_export(FIX / "fire_archive_bundle.zip")
    assert {d.sensor for d in dets} == {"VIIRS_SNPP_NRT", "MODIS_SP"}
    assert len(dets) == 6


@pytest.mark.parametrize("name,expected", [
    ("fire_archive_SV-C2_123456.csv", "VIIRS_SNPP_NRT"),
    ("fire_archive_J1V-C2_123456.csv", "VIIRS_NOAA20_NRT"),
    ("fire_archive_J2V-C2_123456.csv", "VIIRS_NOAA21_NRT"),
    ("fire_archive_M-C61_123456.csv", "MODIS_SP"),
])
def test_ac01_sensor_is_inferred_from_the_export_name(name, expected):
    assert sensor_from_name(name) == expected


def test_ac01_unknown_export_name_is_refused_not_guessed():
    with pytest.raises(ArchiveError, match="не удалось определить сенсор"):
        sensor_from_name("какой-то-файл.csv")


def test_ac01_a_file_that_is_not_a_firms_export_is_refused():
    p = FIX / "firms_error.txt"
    with pytest.raises(ArchiveError, match="не похож на выгрузку"):
        load_export(p, sensor="VIIRS_SNPP_NRT")


def test_ac01_missing_file_is_refused():
    with pytest.raises(ArchiveError, match="не найдена"):
        load_export(FIX / "нет-такого.csv")


# --- AC-03: архивные детекции помечены иначе, чем оперативные ---

def test_ac03_archive_detections_are_marked_as_standard_processing():
    """Standard Processing пересчитан с уточнённой геопривязкой; смешивать
    его с NRT в статистике без пометки нельзя."""
    for d in load_export(VIIRS):
        assert d.version == "2.0"
        assert d.is_archive is True


def test_ac03_operational_detections_are_not_marked_as_archive():
    nrt = firms.parse_csv((FIX / "firms_viirs_nonempty.csv").read_text(),
                          "VIIRS_SNPP_NRT")
    assert all(d.version.endswith("NRT") for d in nrt)
    assert all(d.is_archive is False for d in nrt)


def test_ac03_unknown_version_is_treated_as_operational_not_archive():
    d = firms.Detection(latitude=61.0, longitude=93.0, brightness=330.0, frp=1.0,
                        confidence="n", acquired_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
                        sensor="VIIRS_SNPP_NRT", satellite="N", daynight="D")
    assert d.version == "" and d.is_archive is False


def test_ac03_store_keeps_the_version_and_type_columns():
    dets = load_export(VIIRS)
    with Store(":memory:") as s:
        s.add_detections(dets)
        rows = s.conn.execute(
            "SELECT version, fire_type FROM detections ORDER BY acquired_at").fetchall()
        assert {r["version"] for r in rows} == {"2.0"}
        assert FIRE_TYPE_STATIC_LAND in {r["fire_type"] for r in rows}


def test_ac03_archive_carries_the_static_land_source_class():
    """Класс 2 — то, что SPEC-4 вычисляет персистентностью за 14 суток;
    в архиве он проставлен NASA."""
    flagged = [d for d in load_export(VIIRS) if d.fire_type == FIRE_TYPE_STATIC_LAND]
    assert len(flagged) == 1
    assert flagged[0].latitude == pytest.approx(67.78890)


def test_ac03_summary_separates_archive_from_operational():
    stats = summarise(load_export(VIIRS) + firms.parse_csv(
        (FIX / "firms_viirs_nonempty.csv").read_text(), "VIIRS_SNPP_NRT"))
    assert stats["archive"] == 4 and stats["nrt"] == 3


# --- AC-02: архив не дублирует оперативные данные ---

def test_ac02_reloading_the_same_export_adds_nothing():
    dets = load_export(VIIRS)
    with Store(":memory:") as s:
        assert s.add_detections(dets) == 4
        assert s.add_detections(dets) == 0
        assert s.count_detections() == 4


def test_ac02_archive_and_operational_do_not_duplicate_the_same_detection():
    """Одна и та же точка, пришедшая обоими путями, — одна строка."""
    nrt = firms.parse_csv((FIX / "firms_viirs_nonempty.csv").read_text(),
                          "VIIRS_SNPP_NRT")
    same = [firms.Detection(**{**d.__dict__, "version": "2.0", "fire_type": 0})
            for d in nrt]
    with Store(":memory:") as s:
        assert s.add_detections(nrt) == 3
        assert s.add_detections(same) == 0        # версия не входит в ключ
        assert s.count_detections() == 3


# --- AC-04: конвейеры этапа 1 работают на архиве ---

def test_ac04_archive_detections_flow_through_the_stage_one_pipeline():
    dets = load_export(VIIRS) + load_export(MODIS)
    events = build_events(dets, EventConfig())
    assert events, "события из архива не построились"
    assert all(e.id.startswith("FIRE-") for e in events)
    assert sum(len(e.detections) for e in events) == len(dets)


def test_ac04_persistence_runs_on_archive_and_finds_nothing_on_sparse_data():
    """Выгрузка охватывает 2021-2023, то есть истории с избытком — фильтр
    отвечает `ok`, а не «недостаточно истории». Источников он при этом не
    находит: четырёх детекций за два года не хватает на 10 разных суток в окне."""
    result = analyse(load_export(VIIRS), PersistenceConfig())
    assert result.enough_history is True
    assert result.status == "ok"
    assert result.history_days > 600
    assert result.sources == ()


def test_ac04_short_export_reports_insufficient_history():
    """А вот выгрузка на одни сутки окна в 14 дней не даёт, и фильтр
    обязан сказать именно это, а не «факелов нет»."""
    one_day = [d for d in load_export(VIIRS) if d.acquired_at.year == 2021]
    result = analyse(one_day, PersistenceConfig())
    assert result.enough_history is False
    assert result.status == "insufficient_history"


# --- граница оперативного хранения: тихий ноль ---

def build_probe_client(oldest: date):
    """Каталог, повторяющий поведение FIRMS: за границей — валидный CSV с нулём строк."""
    header = ("latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
              "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n")
    row = "61.0,93.0,330.0,0.4,0.4,{d},0800,N,VIIRS,n,2.0NRT,289.0,4.0,D\n"

    def handler(request: httpx.Request) -> httpx.Response:
        when = date.fromisoformat(str(request.url).rsplit("/", 1)[-1])
        body = header + (row.format(d=when.isoformat()) if when >= oldest else "")
        return httpx.Response(200, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_out_of_window_request_returns_empty_csv_not_an_error():
    """Замерено на живом API 18.09.2026: за границей приходит валидный CSV
    с нулём строк. Наивный цикл построил бы историю из пустоты."""
    region = firms.Region("t", (82.0, 51.0, 109.0, 78.0))
    client = build_probe_client(date(2026, 7, 1))
    res = firms.fetch(region, ["VIIRS_SNPP_NRT"], 1, "KEY", client=client,
                      date="2026-05-01")
    assert res.detections == [] and res.failures == {}


def test_probe_finds_the_retention_boundary():
    region = firms.Region("t", (82.0, 51.0, 109.0, 78.0))
    depth = probe_nrt_depth(region, ["VIIRS_SNPP_NRT"], "KEY",
                            probed_at=date(2026, 9, 18),
                            client=build_probe_client(date(2026, 7, 1)))
    assert depth.oldest_available == date(2026, 7, 1)
    assert depth.days == 79


def test_probe_distinguishes_inside_from_outside_the_window():
    depth = NrtDepth(oldest_available=date(2026, 7, 1), probed_at=date(2026, 9, 18))
    assert depth.covers(date(2026, 8, 1)) is True
    assert depth.covers(date(2026, 6, 1)) is False


def test_probe_reports_a_failing_source_instead_of_calling_it_empty():
    def handler(_r):
        return httpx.Response(503, text="unavailable")

    region = firms.Region("t", (82.0, 51.0, 109.0, 78.0))
    with pytest.raises(ArchiveError, match="проба"):
        probe_nrt_depth(region, ["VIIRS_SNPP_NRT"], "KEY",
                        probed_at=date(2026, 9, 18),
                        client=httpx.Client(transport=httpx.MockTransport(handler)))


# --- AC-06: состав выгрузки публикуется ---

def test_ac06_summary_reports_counts_by_year_sensor_and_type():
    stats = summarise(load_export(VIIRS) + load_export(MODIS))
    assert stats["total"] == 6
    assert stats["by_year"] == {2021: 3, 2022: 2, 2023: 1}
    assert stats["by_sensor"] == {"MODIS_SP": 2, "VIIRS_SNPP_NRT": 4}
    assert "стационарный наземный источник" in stats["by_fire_type"]


def test_ac06_empty_export_summarises_to_zero_without_crashing():
    dets = load_export(FIX / "fire_archive_empty.csv", sensor="VIIRS_SNPP_NRT")
    assert dets == []
    assert summarise(dets)["total"] == 0


# --- AC-05: ручной шаг описан ---

def test_ac05_manual_download_procedure_is_documented():
    doc = (ROOT / "docs" / "archive.md").read_text(encoding="utf-8")
    assert "firms.modaps.eosdis.nasa.gov/download" in doc
    assert "82,51,109,78" in doc
    for token in ("почт", "вручную"):
        assert token in doc.lower(), f"ручной шаг не описан: нет упоминания {token}"
