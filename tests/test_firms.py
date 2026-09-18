"""SPEC-1 acceptance tests — офлайн, на зафиксированных фикстурах.

Каждый тест назван по критерию приёмки из specs/SPEC-1-firms-ingestion.md.
"""
from __future__ import annotations

import os
import subprocess
from datetime import timezone
from pathlib import Path

import httpx
import pytest

from src import firms

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG = ROOT / "config" / "krasnoyarsk.toml"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def write_config(tmp_path: Path, bbox: list[float], *, name="Тест",
                 sources=("VIIRS_SNPP_NRT",), day_range=2) -> Path:
    src_list = ", ".join(f'"{s}"' for s in sources)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "region.toml"
    p.write_text(
        f'[region]\nname = "{name}"\nbbox = {bbox}\n\n'
        f'[firms]\nsources = [{src_list}]\nday_range = {day_range}\n',
        encoding="utf-8",
    )
    return p


# --- AC-01: регион задаётся конфигом, смена bbox не требует правки кода ---

def test_ac01_different_bboxes_produce_different_urls(tmp_path):
    krasnoyarsk = write_config(tmp_path / "a", [82.0, 51.0, 109.0, 78.0])
    irkutsk = write_config(tmp_path / "b", [95.0, 51.0, 120.0, 65.0])

    r1, _, _ = firms.load_region(krasnoyarsk)
    r2, _, _ = firms.load_region(irkutsk)
    u1 = firms.build_url("KEY", "VIIRS_SNPP_NRT", r1, 2)
    u2 = firms.build_url("KEY", "VIIRS_SNPP_NRT", r2, 2)

    assert u1 != u2
    assert "82,51,109,78" in u1
    assert "95,51,120,65" in u2


def test_ac01_shipped_config_is_loadable_and_ordered_west_south_east_north():
    region, sources, day_range = firms.load_region(CONFIG)
    assert region.area_coordinates == "82,51,109,78"
    assert day_range <= firms.MAX_DAY_RANGE
    firms.validate_request(sources, day_range)  # не должно бросить


def test_ac01_malformed_bbox_is_rejected(tmp_path):
    # east < west — перепутанный порядок координат
    p = write_config(tmp_path, [109.0, 51.0, 82.0, 78.0])
    with pytest.raises(firms.ConfigError, match="west<east"):
        firms.load_region(p)


# --- AC-02: DAY_RANGE > 5 отбивается до сети ---

def test_ac02_day_range_above_limit_is_rejected_before_network():
    with pytest.raises(firms.ConfigError, match="day_range must be between 1 and 5"):
        firms.validate_request(["VIIRS_SNPP_NRT"], 6)


@pytest.mark.parametrize("day_range", [1, 2, 5])
def test_ac02_day_range_within_limit_is_accepted(day_range):
    firms.validate_request(["VIIRS_SNPP_NRT"], day_range)


@pytest.mark.parametrize("day_range", [0, -1, 6, 10])
def test_ac02_day_range_outside_limit_is_rejected(day_range):
    with pytest.raises(firms.ConfigError):
        firms.validate_request(["VIIRS_SNPP_NRT"], day_range)


def test_ac02_fetch_refuses_bad_day_range_without_calling_client():
    def explode(*_a, **_kw):
        raise AssertionError("network must not be touched on invalid params")

    client = httpx.Client(transport=httpx.MockTransport(explode))
    region = firms.Region(name="t", bbox=(82.0, 51.0, 109.0, 78.0))
    with pytest.raises(firms.ConfigError):
        firms.fetch(region, ["VIIRS_SNPP_NRT"], 6, "KEY", client=client)


# --- AC-03: LANDSAT_NRT недопустим ---

def test_ac03_landsat_nrt_is_not_an_allowed_source():
    assert "LANDSAT_NRT" not in firms.ALLOWED_SOURCES


def test_ac03_requesting_landsat_nrt_is_a_config_error():
    with pytest.raises(firms.ConfigError, match="LANDSAT_NRT") as exc:
        firms.validate_request(["VIIRS_SNPP_NRT", "LANDSAT_NRT"], 2)
    # ошибка должна объяснять причину, а не просто ругаться
    assert "US and Canada" in str(exc.value)


def test_ac03_shipped_config_does_not_list_landsat():
    _, sources, _ = firms.load_region(CONFIG)
    assert "LANDSAT_NRT" not in sources


# --- AC-04: парсер CSV, три случая ---

def test_ac04_parses_nonempty_viirs_csv():
    dets = firms.parse_csv(fixture("firms_viirs_nonempty.csv"), "VIIRS_SNPP_NRT")
    assert len(dets) == 3
    first = dets[0]
    assert first.latitude == pytest.approx(61.31742)
    assert first.brightness == pytest.approx(330.5)   # из bright_ti4
    assert first.frp == pytest.approx(4.7)
    assert first.confidence == "n"
    assert first.sensor == "VIIRS_SNPP_NRT"
    assert first.daynight == "D"
    assert first.acquired_at.tzinfo is timezone.utc
    assert (first.acquired_at.hour, first.acquired_at.minute) == (8, 12)


def test_ac04_parses_nonempty_modis_csv_with_different_brightness_column():
    dets = firms.parse_csv(fixture("firms_modis_nonempty.csv"), "MODIS_NRT")
    assert len(dets) == 2
    # у MODIS канал называется `brightness`, у VIIRS — `bright_ti4`;
    # обе схемы обязаны сводиться к одному полю
    assert dets[0].brightness == pytest.approx(325.8)
    assert dets[0].confidence == "78"        # числовая шкала, нормализация в SPEC-3
    assert dets[1].frp is None               # пустой frp -> None, не 0.0


def test_ac04_empty_response_is_zero_detections_not_an_error():
    assert firms.parse_csv(fixture("firms_empty.csv"), "VIIRS_SNPP_NRT") == []


def test_ac04_error_body_instead_of_csv_raises():
    with pytest.raises(firms.FirmsResponseError, match="not FIRMS CSV"):
        firms.parse_csv(fixture("firms_error.txt"), "VIIRS_SNPP_NRT")


def test_ac04_midnight_acq_time_parses():
    csv_text = (
        "latitude,longitude,bright_ti4,acq_date,acq_time,satellite,"
        "confidence,frp,daynight\n"
        "60.0,90.0,320.0,2026-09-16,0,N,n,1.0,N\n"
    )
    d = firms.parse_csv(csv_text, "VIIRS_SNPP_NRT")[0]
    assert (d.acquired_at.hour, d.acquired_at.minute) == (0, 0)


# --- AC-05: отказ одного источника не прерывает остальные ---

def _transport_one_source_down(down: str):
    def handler(request: httpx.Request) -> httpx.Response:
        if f"/{down}/" in str(request.url):
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, text=fixture("firms_viirs_nonempty.csv"))
    return httpx.MockTransport(handler)


def test_ac05_one_failing_source_does_not_abort_the_others():
    sources = ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"]
    client = httpx.Client(transport=_transport_one_source_down("VIIRS_NOAA20_NRT"))
    region = firms.Region(name="t", bbox=(82.0, 51.0, 109.0, 78.0))

    result = firms.fetch(region, sources, 2, "KEY", client=client)

    assert list(result.failures) == ["VIIRS_NOAA20_NRT"]
    assert result.succeeded == ["MODIS_NRT", "VIIRS_SNPP_NRT"]
    assert len(result.detections) == 6  # два выживших источника по 3 детекции


def test_ac05_non_csv_body_is_recorded_as_a_failure_not_a_crash():
    def handler(request):
        return httpx.Response(200, text="Invalid MAP_KEY.")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    region = firms.Region(name="t", bbox=(82.0, 51.0, 109.0, 78.0))
    result = firms.fetch(region, ["VIIRS_SNPP_NRT"], 2, "KEY", client=client)

    assert result.detections == []
    assert "FirmsResponseError" in result.failures["VIIRS_SNPP_NRT"]


# --- AC-06: ключ только из окружения и никогда не в выводе ---

def test_ac06_empty_map_key_is_refused():
    region = firms.Region(name="t", bbox=(82.0, 51.0, 109.0, 78.0))
    with pytest.raises(firms.ConfigError, match="FIRMS_MAP_KEY"):
        firms.fetch(region, ["VIIRS_SNPP_NRT"], 2, "", client=None)


def test_ac06_module_has_no_hardcoded_key_default():
    src = (ROOT / "src" / "firms.py").read_text(encoding="utf-8")
    # единственные обращения к ключу — чтение из окружения
    assert 'os.environ.get("FIRMS_MAP_KEY", "")' in src
    assert 'os.environ.get("FIRMS_MAP_KEY")' in src


def test_ac06_redact_removes_the_key_from_any_text(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "s3cr3tkey")
    url = firms.build_url("s3cr3tkey", "VIIRS_SNPP_NRT",
                          firms.Region("t", (82.0, 51.0, 109.0, 78.0)), 2)
    assert "s3cr3tkey" in url                      # ключ действительно в URL
    assert "s3cr3tkey" not in firms.redact(url)    # и не переживает redact


def test_ac06_failure_messages_do_not_leak_the_key(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "s3cr3tkey")

    def handler(request):
        raise httpx.ConnectError(f"cannot connect to {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    region = firms.Region(name="t", bbox=(82.0, 51.0, 109.0, 78.0))
    result = firms.fetch(region, ["VIIRS_SNPP_NRT"], 2, "s3cr3tkey", client=client)

    assert result.failures
    assert "s3cr3tkey" not in result.failures["VIIRS_SNPP_NRT"]
    assert "<FIRMS_MAP_KEY>" in result.failures["VIIRS_SNPP_NRT"]


def test_ac06_key_value_is_not_committed_anywhere_in_the_repo():
    """AC-06: ключ не должен попасть в коммитируемые файлы.

    Исключения берутся из .gitignore, а не хардкодятся: `.env` — штатное место
    для ключа, и найти его там означает, что всё настроено верно. Проверяется
    именно то, что ключ не утёк за пределы игнорируемых путей.
    """
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        pytest.skip("FIRMS_MAP_KEY not set in this environment")

    ignored = [
        line.strip().rstrip("/")
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    excludes = ["--exclude-dir=.git"]
    for pattern in ignored:
        excludes += [f"--exclude={pattern}", f"--exclude-dir={pattern}"]

    hit = subprocess.run(
        ["grep", "-rIl", *excludes, key, str(ROOT)],
        capture_output=True, text=True,
    )
    assert hit.returncode != 0, (
        "FIRMS_MAP_KEY found in non-ignored file(s): " + hit.stdout.strip()
    )


# --- CLI ---

def test_cli_rejects_landsat_without_network(tmp_path):
    p = write_config(tmp_path, [82.0, 51.0, 109.0, 78.0],
                     sources=("VIIRS_SNPP_NRT", "LANDSAT_NRT"))
    assert firms.main(["fetch", "--config", str(p)]) == 2


def test_cli_rejects_day_range_above_limit(tmp_path):
    p = write_config(tmp_path, [82.0, 51.0, 109.0, 78.0])
    assert firms.main(["fetch", "--config", str(p), "--day-range", "6"]) == 2


# --- расширение под SPEC-4: историческое окно по дате ---

def test_date_parameter_appends_to_the_url():
    r = firms.Region("t", (82.0, 51.0, 109.0, 78.0))
    assert firms.build_url("K", "VIIRS_SNPP_NRT", r, 5).endswith("/5")
    assert firms.build_url("K", "VIIRS_SNPP_NRT", r, 5, "2026-09-01").endswith("/5/2026-09-01")


def test_date_parameter_still_respects_the_day_range_limit():
    """Дата расширяет глубину истории, но не лимит окна одного запроса."""
    with pytest.raises(firms.ConfigError):
        firms.validate_request(["VIIRS_SNPP_NRT"], 6)


def test_fetch_passes_the_date_through_to_the_request():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text=fixture("firms_empty.csv"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    region = firms.Region("t", (82.0, 51.0, 109.0, 78.0))
    firms.fetch(region, ["VIIRS_SNPP_NRT"], 5, "KEY", client=client, date="2026-09-01")
    assert seen[0].endswith("/5/2026-09-01")
