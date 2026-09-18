"""SPEC-1: конфигурация региона и приём NASA FIRMS Area API.

Модуль забирает точки активного горения по bbox региона, разбирает CSV и
отдаёт нормализованные детекции. Конкретика источника, определяющая дизайн
(проверено 18.09.2026, docs/TZ.md §3.1):

* `DAY_RANGE` не может превышать 5 суток за запрос;
* `LANDSAT_NRT` покрывает только США и Канаду — по России данных не даёт;
* схемы CSV у MODIS и VIIRS различаются именем канала яркостной температуры.

Ключ `FIRMS_MAP_KEY` живёт только в окружении и подставляется в URL. Поэтому
всё, что может оказаться в логе, в тексте исключения или в receipt'е, проходит
через `redact()` — URL с ключом утекает незаметно (REQ-005).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

API_ROOT = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Глобальные источники FIRMS. LANDSAT_NRT сознательно отсутствует: продукт
# Landsat Fire and Thermal Anomalies строится из прямого приёма станции USGS
# EROS и покрывает только CONUS, юг Канады и север Мексики.
ALLOWED_SOURCES = frozenset({
    "MODIS_NRT",
    "MODIS_SP",
    "VIIRS_SNPP_NRT",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA20_SP",
    "VIIRS_NOAA21_NRT",
})

# Жёсткий лимит FIRMS Area API.
MAX_DAY_RANGE = 5

# Имя канала яркостной температуры зависит от сенсора; приводим к `brightness`.
_BRIGHTNESS_FIELDS = ("brightness", "bright_ti4")

# Признак того, что пришёл именно CSV, а не текст ошибки.
_REQUIRED_COLUMNS = frozenset({"latitude", "longitude", "acq_date", "acq_time"})


class ConfigError(ValueError):
    """Параметры запроса неверны — до сети дело не дойдёт."""


class FirmsResponseError(RuntimeError):
    """FIRMS ответил чем-то, что не является ожидаемым CSV."""


def redact(text: str) -> str:
    """Убрать значение FIRMS_MAP_KEY из строки, пригодной к публикации.

    URL запроса содержит ключ, а URL попадает в сообщения об ошибках httpx.
    Без этой функции ключ уходит в логи и receipt'ы.
    """
    key = os.environ.get("FIRMS_MAP_KEY")
    if key:
        text = text.replace(key, "<FIRMS_MAP_KEY>")
    return text


@dataclass(frozen=True)
class Region:
    name: str
    bbox: tuple[float, float, float, float]  # west, south, east, north

    @property
    def area_coordinates(self) -> str:
        return ",".join(f"{v:g}" for v in self.bbox)


@dataclass(frozen=True)
class Detection:
    """Одна детекция, приведённая к общей схеме (ФТ-1.2)."""
    latitude: float
    longitude: float
    brightness: float | None
    frp: float | None
    confidence: str          # сырое значение сенсора; нормализация — SPEC-3
    acquired_at: datetime    # UTC
    sensor: str
    satellite: str
    daynight: str
    # Версия продукта из CSV: "2.0NRT" у оперативных, "2.0" у Standard Processing.
    # Продукты различаются геопривязкой, и смешивать их без пометки нельзя (SPEC-10).
    version: str = ""
    # Колонка `type` архивных выгрузок: 0 — растительность, 1 — вулкан,
    # 2 — стационарный наземный источник, 3 — офшор. В NRT-потоке отсутствует.
    fire_type: int | None = None

    @property
    def is_archive(self) -> bool:
        """Standard Processing против NRT. Пустая версия — неизвестно, считаем NRT."""
        return bool(self.version) and "NRT" not in self.version.upper()


@dataclass
class FetchResult:
    """Итог опроса. Сбой источника не отменяет остальные (AC-05)."""
    detections: list[Detection] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> list[str]:
        return sorted({d.sensor for d in self.detections})


def load_region(config_path: str | Path) -> tuple[Region, list[str], int]:
    """Прочитать регион и параметры опроса из TOML (AC-01)."""
    data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    region_cfg = data["region"]
    bbox = tuple(float(v) for v in region_cfg["bbox"])
    if len(bbox) != 4:
        raise ConfigError(f"bbox must have 4 values (west,south,east,north), got {len(bbox)}")
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ConfigError(f"bbox must be west<east and south<north, got {bbox}")
    firms_cfg = data.get("firms", {})
    sources = list(firms_cfg.get("sources", []))
    day_range = int(firms_cfg.get("day_range", 1))
    return Region(name=region_cfg["name"], bbox=bbox), sources, day_range


def validate_request(sources: list[str], day_range: int) -> None:
    """Отбить неверные параметры до обращения к сети (AC-02, AC-03)."""
    if not sources:
        raise ConfigError("no FIRMS sources configured")
    rejected = [s for s in sources if s not in ALLOWED_SOURCES]
    if rejected:
        extra = ""
        if "LANDSAT_NRT" in rejected:
            extra = (" LANDSAT_NRT covers only the US and Canada (USGS EROS direct"
                     " broadcast) and returns no data for Russia.")
        raise ConfigError(
            f"unsupported FIRMS source(s): {', '.join(rejected)}."
            f" Allowed: {', '.join(sorted(ALLOWED_SOURCES))}.{extra}"
        )
    if not 1 <= day_range <= MAX_DAY_RANGE:
        raise ConfigError(
            f"day_range must be between 1 and {MAX_DAY_RANGE} "
            f"(FIRMS Area API hard limit), got {day_range}"
        )


def build_url(map_key: str, source: str, region: Region, day_range: int,
              date: str | None = None) -> str:
    """URL запроса. `date` (YYYY-MM-DD) задаёт КОНЕЦ окна day_range.

    Без даты FIRMS отдаёт последние day_range суток. С датой — окно,
    заканчивающееся этой датой, в пределах примерно двух месяцев NRT-хранения.
    Это позволяет собрать историю глубже 5 суток (нужно SPEC-4) без обращения
    к архиву и без Earthdata Login.
    """
    url = f"{API_ROOT}/{map_key}/{source}/{region.area_coordinates}/{day_range}"
    return f"{url}/{date}" if date else url


def _to_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_csv(text: str, sensor: str) -> list[Detection]:
    """Разобрать ответ FIRMS (AC-04).

    Три случая: непустой CSV, CSV с одним заголовком, и текст ошибки вместо
    CSV — последний отличается отсутствием обязательных колонок.
    """
    reader = csv.DictReader(io.StringIO(text))
    columns = set(reader.fieldnames or ())
    missing = _REQUIRED_COLUMNS - columns
    if missing:
        preview = text.strip()[:200]
        raise FirmsResponseError(
            f"{sensor}: response is not FIRMS CSV (missing columns: "
            f"{', '.join(sorted(missing))}); body starts with: {redact(preview)!r}"
        )

    detections: list[Detection] = []
    for row in reader:
        lat, lon = _to_float(row.get("latitude")), _to_float(row.get("longitude"))
        if lat is None or lon is None:
            continue
        brightness = next(
            (_to_float(row[f]) for f in _BRIGHTNESS_FIELDS if row.get(f)), None
        )
        detections.append(Detection(
            latitude=lat,
            longitude=lon,
            brightness=brightness,
            frp=_to_float(row.get("frp")),
            confidence=(row.get("confidence") or "").strip(),
            acquired_at=_parse_acq(row.get("acq_date"), row.get("acq_time")),
            sensor=sensor,
            satellite=(row.get("satellite") or "").strip(),
            daynight=(row.get("daynight") or "").strip(),
            version=(row.get("version") or "").strip(),
            fire_type=_to_int(row.get("type")),
        ))
    return detections


def _to_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_acq(acq_date: str | None, acq_time: str | None) -> datetime:
    """FIRMS отдаёт дату и время раздельно; время — HHMM без разделителя."""
    raw_time = (acq_time or "0").strip() or "0"
    minutes = int(raw_time) % 100
    hours = int(raw_time) // 100
    day = datetime.strptime((acq_date or "").strip(), "%Y-%m-%d")
    return day.replace(hour=hours, minute=minutes, tzinfo=timezone.utc)


def fetch(
    region: Region,
    sources: list[str],
    day_range: int,
    map_key: str,
    client: httpx.Client | None = None,
    date: str | None = None,
) -> FetchResult:
    """Опросить все источники. Сбой одного не прерывает остальные (AC-05)."""
    validate_request(sources, day_range)
    if not map_key:
        raise ConfigError("FIRMS_MAP_KEY is empty; set it in the environment")

    result = FetchResult()
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        for source in sources:
            url = build_url(map_key, source, region, day_range, date)
            try:
                response = client.get(url)
                response.raise_for_status()
                result.detections.extend(parse_csv(response.text, source))
            except Exception as exc:  # источник упал — остальные продолжаем
                result.failures[source] = redact(f"{type(exc).__name__}: {exc}")
    finally:
        if owns_client:
            client.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m src.firms")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="fetch active fire detections for a region")
    p.add_argument("--config", required=True)
    p.add_argument("--day-range", type=int, default=None,
                   help=f"override config; 1..{MAX_DAY_RANGE}")

    args = parser.parse_args(argv)
    region, sources, day_range = load_region(args.config)
    if args.day_range is not None:
        day_range = args.day_range

    try:
        validate_request(sources, day_range)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    map_key = os.environ.get("FIRMS_MAP_KEY", "")
    if not map_key:
        print("FIRMS_MAP_KEY is not set in the environment", file=sys.stderr)
        return 2

    print(f"region: {region.name}  bbox: {region.area_coordinates}  "
          f"day_range: {day_range}")
    result = fetch(region, sources, day_range, map_key)

    by_source: dict[str, int] = {s: 0 for s in sources}
    for d in result.detections:
        by_source[d.sensor] = by_source.get(d.sensor, 0) + 1
    for source in sources:
        if source in result.failures:
            print(f"  {source:20s} FAILED  {result.failures[source]}")
        else:
            print(f"  {source:20s} {by_source[source]:6d} detections")
    print(f"total: {len(result.detections)} detections, "
          f"{len(result.failures)} source(s) failed")

    # Полный отказ всех источников — это ошибка, пустой результат — нет.
    return 1 if len(result.failures) == len(sources) else 0


if __name__ == "__main__":
    sys.exit(main())
