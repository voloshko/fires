"""SPEC-10: многолетний архив детекций FIRMS.

Оперативный контур собирает историю через Area API с параметром даты (SPEC-4),
но эта глубина ограничена. Измерено 18.09.2026 бинарным поиском по границе:
данные есть с **1 июля 2026**, то есть **79 суток** назад, а на предыдущие даты
API отдаёт валидный CSV с нулём строк.

**Это тихий отказ, и он опаснее ошибки.** Наивный цикл по датам построил бы
«десятилетнюю историю» из пустых ответов, фильтр персистентности отрапортовал бы
«факелов не найдено» вместо «данных нет», а событий не оказалось бы вовсе.
Поэтому загрузчик отличает пустое окно внутри доступной глубины от окна за её
пределами и отказывается молчать.

Многолетний период берётся через **FIRMS Archive Download**
(https://firms.modaps.eosdis.nasa.gov/download/). Это не REST API: запрос
подтверждается кодом на почту, ссылка приходит письмом. Модуль загружает уже
полученную выгрузку; ручной шаг описан в `docs/archive.md` и здесь не
изображается автоматизированным.

Архивные выгрузки несут две колонки, которых нет в NRT-потоке:

* `version` — «2.0» у Standard Processing против «2.0NRT» у оперативных.
  Продукты пересчитаны с уточнённой геопривязкой, и смешивать их в статистике
  без пометки нельзя.
* `type` — 0 растительность, 1 вулкан, **2 стационарный наземный источник**,
  3 офшор. Класс 2 — это ровно то, что SPEC-4 вычисляет персистентностью за
  14 суток; в архиве он проставлен NASA.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.firms import ALLOWED_SOURCES, Detection, FirmsResponseError, parse_csv

# Колонка `type` архивных выгрузок.
FIRE_TYPE_VEGETATION = 0
FIRE_TYPE_VOLCANO = 1
FIRE_TYPE_STATIC_LAND = 2
FIRE_TYPE_OFFSHORE = 3

FIRE_TYPE_NAMES = {
    FIRE_TYPE_VEGETATION: "растительность",
    FIRE_TYPE_VOLCANO: "вулкан",
    FIRE_TYPE_STATIC_LAND: "стационарный наземный источник",
    FIRE_TYPE_OFFSHORE: "офшор",
}


class ArchiveError(RuntimeError):
    """Выгрузка непригодна."""


@dataclass(frozen=True)
class NrtDepth:
    """Измеренная глубина оперативного хранения FIRMS."""
    oldest_available: date
    probed_at: date

    @property
    def days(self) -> int:
        return (self.probed_at - self.oldest_available).days

    def covers(self, when: date) -> bool:
        return when >= self.oldest_available


def sensor_from_name(name: str) -> str:
    """Определить источник по имени файла выгрузки.

    Archive Download называет файлы по продукту: fire_archive_SV-C2_… для
    VIIRS S-NPP, …_J1V-C2_… для NOAA-20, fire_archive_M-C61_… для MODIS.
    """
    n = name.upper()
    if "J2V" in n or "NOAA21" in n or "NOAA-21" in n:
        return "VIIRS_NOAA21_NRT"
    if "J1V" in n or "NOAA20" in n or "NOAA-20" in n:
        return "VIIRS_NOAA20_NRT"
    if "SV-C2" in n or "SUOMI" in n or "SNPP" in n:
        return "VIIRS_SNPP_NRT"
    if "M-C6" in n or "MODIS" in n:
        return "MODIS_SP"
    raise ArchiveError(
        f"не удалось определить сенсор по имени {name!r}; переименуйте файл или "
        f"передайте --sensor явно. Допустимые: {', '.join(sorted(ALLOWED_SOURCES))}")


def load_export(path: str | Path, sensor: str | None = None) -> list[Detection]:
    """Прочитать выгрузку Archive Download: CSV или ZIP с CSV внутри."""
    path = Path(path)
    if not path.exists():
        raise ArchiveError(f"выгрузка {path} не найдена")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise ArchiveError(f"в архиве {path} нет ни одного CSV")
            out = []
            for member in members:
                out.extend(_parse(z.read(member).decode("utf-8"),
                                  sensor or sensor_from_name(member), path, member))
            return out

    return _parse(path.read_text(encoding="utf-8"),
                  sensor or sensor_from_name(path.name), path, path.name)


def _parse(text: str, sensor: str, path: Path, member: str) -> list[Detection]:
    if sensor not in ALLOWED_SOURCES:
        raise ArchiveError(f"недопустимый источник {sensor!r} для {member}")
    try:
        return parse_csv(text, sensor)
    except FirmsResponseError as exc:
        raise ArchiveError(f"{path}:{member} не похож на выгрузку FIRMS: {exc}") from None


def probe_nrt_depth(region, sources, map_key, probed_at: date | None = None,
                    max_back_days: int = 120, client=None) -> NrtDepth:
    """Найти границу оперативного хранения бинарным поиском.

    Граница нужна, чтобы отличать «огня не было» от «данных за этот период уже
    нет»: API в обоих случаях отдаёт валидный CSV, и без замера эти исходы
    неразличимы.
    """
    from src import firms

    probed_at = probed_at or date.today()
    source = sources[0]

    def has_data(when: date) -> bool:
        res = firms.fetch(region, [source], 1, map_key, client=client,
                          date=when.isoformat())
        if res.failures:
            raise ArchiveError(f"проба на {when}: {res.failures[source]}")
        return bool(res.detections)

    lo, hi = probed_at - timedelta(days=max_back_days), probed_at
    if has_data(lo):
        return NrtDepth(oldest_available=lo, probed_at=probed_at)
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) // 2
        if has_data(mid):
            hi = mid
        else:
            lo = mid
    return NrtDepth(oldest_available=hi, probed_at=probed_at)


def summarise(detections) -> dict:
    """Состав выгрузки: по годам, по сенсорам, по типу источника."""
    by_year = Counter(d.acquired_at.year for d in detections)
    by_sensor = Counter(d.sensor for d in detections)
    by_type = Counter(d.fire_type for d in detections if d.fire_type is not None)
    archive = sum(1 for d in detections if d.is_archive)
    return {
        "total": len(detections),
        "archive": archive,
        "nrt": len(detections) - archive,
        "by_year": dict(sorted(by_year.items())),
        "by_sensor": dict(sorted(by_sensor.items())),
        "by_fire_type": {FIRE_TYPE_NAMES.get(k, str(k)): v
                         for k, v in sorted(by_type.items())},
    }


def main(argv=None) -> int:
    import argparse, json, os, sys
    from src import firms
    from src.store import Store

    ap = argparse.ArgumentParser(prog="python3 -m src.archive")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("load", help="загрузить выгрузку Archive Download")
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True, nargs="+", help="CSV или ZIP")
    p.add_argument("--sensor", help="переопределить источник")
    p.add_argument("--db", default="data/archive/detections.db")

    p = sub.add_parser("probe", help="измерить глубину оперативного хранения")
    p.add_argument("--config", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "probe":
        map_key = os.environ.get("FIRMS_MAP_KEY", "")
        if not map_key:
            print("FIRMS_MAP_KEY is not set in the environment", file=sys.stderr)
            return 2
        region, sources, _ = firms.load_region(args.config)
        depth = probe_nrt_depth(region, sources, map_key)
        print(f"оперативное хранение: данные с {depth.oldest_available} "
              f"({depth.days} суток назад на {depth.probed_at})")
        print("глубже — только FIRMS Archive Download, см. docs/archive.md")
        return 0

    detections = []
    for path in args.input:
        got = load_export(path, args.sensor)
        print(f"  {path}: {len(got)} детекций")
        detections.extend(got)
    if not detections:
        print("выгрузки пусты — загружать нечего", file=sys.stderr)
        return 1

    stats = summarise(detections)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    store = Store(args.db)
    before = store.count_detections()
    new = store.add_detections(detections)
    print(f"\nв базе было {before}, добавлено новых {new}, стало {store.count_detections()}")
    store.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
