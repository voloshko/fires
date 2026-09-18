"""SPEC-7: подбор пары сцен Sentinel-2 и расчёт dNBR/RBR.

Четыре решения отличают этот модуль от буквального прочтения ТЗ и проверены
обращением к данным (docs/TZ.md §6):

1. **Каналы B8A + B12, оба 20 м.** ТЗ говорило абстрактно «NIR/SWIR», но
   фиксировало площадь пикселя 20 м. Это согласуется только при B8A; B08 (10 м)
   потребовал бы ресемплинга и дал бы 0.01 га на пиксель.
2. **Пара строго из одного MGRS-тайла.** Разные тайлы — разные зоны UTM и разные
   растровые сетки; поэлементное вычитание без перепроецирования дало бы молча
   неверный результат.
3. **Двухступенчатый фильтр облачности.** `eo:cloud_cover` относится ко всему
   тайлу 110x110 км. Порог 20% из ТЗ в сибирский сезон, с дымом от самих
   пожаров, не оставил бы ни одной пары. Схема: грубый фильтр по метаданным,
   затем доля валидных пикселей SCL внутри самой области пожара.
4. **Маска по SCL, не по s2cloudless.** Ассета с вероятностью облачности в
   Planetary Computer нет (проверено: B01-B12, B8A, SCL, AOT, WVP).

Отдельно: **площадь никогда не возвращается без `masked_fraction`**. Число
гектаров, посчитанное по области, наполовину закрытой облаком, неотличимо от
честного — и это главный способ, которым такой сервис начинает врать.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Каналы NBR. Оба 20 м — сетки совпадают, ресемплинг не нужен.
NIR_BAND, SWIR_BAND = "B8A", "B12"
S2_SCALE = 10000.0

# Классы SCL, исключаемые из статистики: нет данных, насыщение, тень облака,
# облако средней и высокой вероятности, перистые, снег/лёд.
SCL_INVALID = frozenset({0, 1, 3, 8, 9, 10, 11})

SEVERITY_ORDER = ("unburnt", "low", "moderate_low", "moderate_high", "high")


class BurnStatus(Enum):
    OK = "ok"
    DEFERRED = "deferred"     # расчёт невозможен сейчас, но станет возможен
    FAILED = "failed"         # пары нет и не будет


@dataclass(frozen=True)
class Thresholds:
    version: str = "usgs-baseline-1"
    unburnt: float = 0.10
    low: float = 0.27
    moderate_low: float = 0.44
    moderate_high: float = 0.66

    @property
    def breaks(self) -> tuple[float, ...]:
        return (self.unburnt, self.low, self.moderate_low, self.moderate_high)

    def __post_init__(self):
        b = self.breaks
        if list(b) != sorted(b):
            raise ValueError(f"severity breaks must increase, got {b}")


@dataclass(frozen=True)
class SeasonBand:
    max_lat: float
    start: tuple[int, int]
    end: tuple[int, int]

    def contains(self, when: date) -> bool:
        return self.start <= (when.month, when.day) <= self.end


@dataclass(frozen=True)
class BurnConfig:
    pre_window_days: tuple[int, int] = (5, 60)
    post_window_days: tuple[int, int] = (5, 60)
    scene_cloud_max: float = 60.0
    min_valid_fraction: float = 0.80
    aoi_pad_m: float = 3000.0
    max_aoi_km: float = 60.0
    thresholds: Thresholds = field(default_factory=Thresholds)
    season: tuple[SeasonBand, ...] = ()

    def season_for(self, lat: float) -> SeasonBand | None:
        for band in sorted(self.season, key=lambda b: b.max_lat):
            if abs(lat) <= band.max_lat:
                return band
        return None


def load_burn_config(config_path: str | Path) -> BurnConfig:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8")).get("burn", {})
    th = cfg.get("thresholds", {})
    def _md(s: str) -> tuple[int, int]:
        m, d = s.split("-")
        return (int(m), int(d))
    return BurnConfig(
        pre_window_days=tuple(cfg.get("pre_window_days", (5, 60))),
        post_window_days=tuple(cfg.get("post_window_days", (5, 60))),
        scene_cloud_max=float(cfg.get("scene_cloud_max", 60.0)),
        min_valid_fraction=float(cfg.get("min_valid_fraction", 0.80)),
        aoi_pad_m=float(cfg.get("aoi_pad_m", 3000.0)),
        max_aoi_km=float(cfg.get("max_aoi_km", 60.0)),
        thresholds=Thresholds(
            version=th.get("version", "usgs-baseline-1"),
            unburnt=float(th.get("unburnt", 0.10)),
            low=float(th.get("low", 0.27)),
            moderate_low=float(th.get("moderate_low", 0.44)),
            moderate_high=float(th.get("moderate_high", 0.66)),
        ),
        season=tuple(SeasonBand(float(b["max_lat"]), _md(b["start"]), _md(b["end"]))
                     for b in cfg.get("season", [])),
    )


# --- чистые вычисления: тестируются без сети ---

def valid_mask(scl: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Валидные пиксели: не забракованы SCL и не nodata в каналах.

    Ноль в B8A или B12 — это nodata Sentinel-2, а не валидная нулевая
    отражательная способность.
    """
    ok = ~np.isin(scl, list(SCL_INVALID))
    return ok & (nir > 0) & (swir > 0)


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """NBR = (NIR - SWIR) / (NIR + SWIR). Масштаб 10000 сокращается."""
    a = nir.astype("float32") / S2_SCALE
    b = swir.astype("float32") / S2_SCALE
    denom = a + b
    out = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, denom, out=out, where=denom != 0)
    return out


def rbr(dnbr: np.ndarray, nbr_pre: np.ndarray) -> np.ndarray:
    return dnbr / (nbr_pre + 1.001)


def classify(dnbr: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """dNBR -> коды классов 0..4 в порядке SEVERITY_ORDER."""
    out = np.zeros(dnbr.shape, dtype="uint8")
    for i, edge in enumerate(thresholds.breaks):
        out[dnbr > edge] = i + 1
    return out


def masked_fraction(valid: np.ndarray, requested_pixels: int) -> float:
    """Доля области пожара, исключённая из статистики.

    Знаменатель — ВСЯ запрошенная область, а не прочитанная её часть. Сцена,
    задевающая область краем полосы съёмки, покрывает считанные проценты AOI;
    если делить на прочитанное, она отрапортует «0% под маской», и площадь
    гари, посчитанная по одной десятой пожара, будет выглядеть полной.
    """
    if requested_pixels <= 0:
        return 1.0
    return float(1.0 - valid.sum() / requested_pixels)


def aoi_extent_km(bbox) -> tuple[float, float]:
    """Размер области в километрах.

    Ограничивать область в градусах нельзя: градус долготы на 64 градусах
    северной широты вдвое короче экваториального, поэтому порог в градусах
    получается строже на юге и слабее на севере — ровно наоборот тому, что нужно.
    """
    from src.persistence import meters_between
    west, south, east, north = bbox
    mid_lat = (south + north) / 2
    return (meters_between(mid_lat, west, mid_lat, east) / 1000,
            meters_between(south, west, north, west) / 1000)


def pixel_area_ha(transform) -> float:
    """Площадь пикселя в гектарах из аффинного преобразования сцены (UTM)."""
    return abs(transform.a * transform.e) / 10_000.0


def area_by_class(classes: np.ndarray, valid: np.ndarray, transform) -> dict[str, float]:
    """Площадь по классам тяжести, га. Считается в НАТИВНОЙ проекции сцены."""
    per_pixel = pixel_area_ha(transform)
    return {name: float((classes[valid] == i).sum()) * per_pixel
            for i, name in enumerate(SEVERITY_ORDER)}


@dataclass(frozen=True)
class BurnResult:
    """Итог картирования.

    `masked_fraction` — не опциональное поле: площадь без неё
    неинтерпретируема (REQ-002).
    """
    status: BurnStatus
    event_id: str
    area_ha: dict[str, float]
    masked_fraction: float
    thresholds_version: str
    mgrs_tile: str | None = None
    scene_before: str | None = None
    scene_after: str | None = None
    scene_before_date: str | None = None
    scene_after_date: str | None = None
    reason: str | None = None
    retry_after: str | None = None

    @property
    def burned_ha(self) -> float:
        """Площадь гари: всё, кроме класса unburnt."""
        return sum(v for k, v in self.area_ha.items() if k != "unburnt")


def deferred(event_id: str, reason: str, retry_after: str | None,
             thresholds_version: str) -> BurnResult:
    """Расчёт невозможен сейчас. Ноль гектаров здесь был бы ложью."""
    return BurnResult(
        status=BurnStatus.DEFERRED, event_id=event_id,
        area_ha={k: 0.0 for k in SEVERITY_ORDER}, masked_fraction=1.0,
        thresholds_version=thresholds_version, reason=reason, retry_after=retry_after)


# --- сезонное окно ---

def season_check(lat: float, fire_end: datetime, config: BurnConfig
                 ) -> tuple[bool, str | None, str | None]:
    """Успеет ли окно поиска сцены «после» уложиться в сезон.

    Севернее ~66 градусов с ноября по февраль оптики нет вообще: Sentinel-2 не
    снимает в полярную ночь. Плюс снег искажает NBR. Возврат нулевой площади в
    такой ситуации был бы неотличим от «гарь не найдена».
    """
    band = config.season_for(lat)
    if band is None:
        return True, None, None
    earliest = (fire_end + timedelta(days=config.post_window_days[0])).date()
    if band.contains(earliest):
        return True, None, None
    year = earliest.year + (0 if (earliest.month, earliest.day) < band.start else 1)
    retry = date(year, band.start[0], band.start[1])
    return False, (
        f"пожар завершился вне безснежного и светового окна для широты "
        f"{lat:.1f}: сезон {band.start[0]:02d}-{band.start[1]:02d}.."
        f"{band.end[0]:02d}-{band.end[1]:02d}"), retry.isoformat()


# --- подбор пары ---

@dataclass(frozen=True)
class Candidate:
    item_id: str
    tile: str
    when: datetime
    scene_cloud: float


@dataclass(frozen=True)
class Pair:
    tile: str
    before: Candidate
    after: Candidate
    before_valid: float
    after_valid: float


def select_pair(pre: list[Candidate], post: list[Candidate], config: BurnConfig,
                valid_fraction) -> Pair | None:
    """Выбрать пару «до/после» строго из одного MGRS-тайла (AC-01, AC-02).

    `valid_fraction(candidate) -> float` читает окно SCL по области пожара.
    Кандидаты перебираются в порядке возрастания облачности сцены, чтобы не
    читать заведомо худшие.
    """
    def by_tile(items):
        out: dict[str, list[Candidate]] = {}
        for c in items:
            if c.scene_cloud <= config.scene_cloud_max:
                out.setdefault(c.tile, []).append(c)
        for v in out.values():
            v.sort(key=lambda c: c.scene_cloud)
        return out

    pre_t, post_t = by_tile(pre), by_tile(post)
    best: Pair | None = None
    cache: dict[str, float] = {}

    def frac(c: Candidate) -> float:
        if c.item_id not in cache:
            cache[c.item_id] = valid_fraction(c)
        return cache[c.item_id]

    def first_good(items):
        for c in items:
            f = frac(c)
            if f >= config.min_valid_fraction:
                return c, f
        return None, 0.0

    for tile in sorted(set(pre_t) & set(post_t)):
        b, bf = first_good(pre_t[tile])
        if b is None:
            continue
        a, af = first_good(post_t[tile])
        if a is None:
            continue
        if best is None or (bf + af) > (best.before_valid + best.after_valid):
            best = Pair(tile=tile, before=b, after=a, before_valid=bf, after_valid=af)
    return best


# --- слой, работающий с сетью ---

def _stac_client():
    import planetary_computer as pc
    from pystac_client import Client
    return Client.open(STAC_URL, modifier=pc.sign_inplace)


def _search(client, bbox, start: datetime, end: datetime) -> list[Candidate]:
    items = client.search(
        collections=[COLLECTION], bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
    ).item_collection()
    out = []
    for it in items:
        tile = it.properties.get("s2:mgrs_tile")
        if not tile:
            continue
        out.append(Candidate(
            item_id=it.id, tile=tile,
            when=datetime.fromisoformat(it.properties["datetime"].replace("Z", "+00:00")),
            scene_cloud=float(it.properties.get("eo:cloud_cover", 100.0))))
    return out, {it.id: it for it in items}


def _read_window(href: str, bbox):
    """Окно растра по bbox в WGS84, в НАТИВНОЙ проекции сцены.

    Возвращает также ЗАПРОШЕННОЕ число пикселей. rasterio обрезает окно по
    границам растра, поэтому прочитанный массив может покрывать лишь часть
    области пожара. Без этого числа доля под маской считалась бы от того, что
    удалось прочитать, и сцена, задевающая область краем, рапортовала бы
    «0% под маской» — см. defect DEF-01 в receipt SPEC-7-LIVE-001.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = window_from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        requested = int(round(win.width)) * int(round(win.height))
        return src.read(1, window=win), src.window_transform(win), requested


def compute_pair(item_before, item_after, bbox, config: BurnConfig, event_id: str,
                 tile: str) -> BurnResult:
    """Прочитать каналы обеих сцен и посчитать площадь по классам."""
    import planetary_computer as pc
    b_before = {k: _read_window(pc.sign(item_before.assets[k]).href, bbox)
                for k in (NIR_BAND, SWIR_BAND, "SCL")}
    b_after = {k: _read_window(pc.sign(item_after.assets[k]).href, bbox)
               for k in (NIR_BAND, SWIR_BAND, "SCL")}

    shapes = {v[0].shape for v in list(b_before.values()) + list(b_after.values())}
    if len(shapes) != 1:
        raise ValueError(f"grids differ between scenes/bands: {shapes}; "
                         f"pair must come from one MGRS tile")
    transform = b_before[NIR_BAND][1]
    requested = max(b_before[NIR_BAND][2], b_after[NIR_BAND][2])

    valid = (valid_mask(b_before["SCL"][0], b_before[NIR_BAND][0], b_before[SWIR_BAND][0])
             & valid_mask(b_after["SCL"][0], b_after[NIR_BAND][0], b_after[SWIR_BAND][0]))
    nbr_pre = nbr(b_before[NIR_BAND][0], b_before[SWIR_BAND][0])
    nbr_post = nbr(b_after[NIR_BAND][0], b_after[SWIR_BAND][0])
    d = np.nan_to_num(nbr_pre - nbr_post, nan=0.0)
    classes = classify(d, config.thresholds)

    return BurnResult(
        status=BurnStatus.OK, event_id=event_id,
        area_ha=area_by_class(classes, valid, transform),
        masked_fraction=masked_fraction(valid, requested),
        thresholds_version=config.thresholds.version,
        mgrs_tile=tile,
        scene_before=item_before.id, scene_after=item_after.id,
        scene_before_date=item_before.properties["datetime"][:10],
        scene_after_date=item_after.properties["datetime"][:10])


def run_for_event(event, config: BurnConfig, client=None) -> BurnResult:
    """Сквозной путь: событие SPEC-6 -> площадь гари по классам (AC-07)."""
    tv = config.thresholds.version
    _, lat = event.centroid
    ok, reason, retry = season_check(lat, event.last_seen, config)
    if not ok:
        return deferred(event.id, reason, retry, tv)

    bbox = event.padded_bbox(config.aoi_pad_m)
    width_km, height_km = aoi_extent_km(bbox)
    if max(width_km, height_km) > config.max_aoi_km:
        return BurnResult(
            status=BurnStatus.FAILED, event_id=event.id,
            area_ha={k: 0.0 for k in SEVERITY_ORDER}, masked_fraction=1.0,
            thresholds_version=tv,
            reason=f"область события {width_km:.0f}x{height_km:.0f} км превышает "
                   f"max_aoi_km={config.max_aoi_km:g}")

    client = client or _stac_client()
    pre, pre_items = _search(
        client, bbox,
        event.first_seen - timedelta(days=config.pre_window_days[1]),
        event.first_seen - timedelta(days=config.pre_window_days[0]))
    post, post_items = _search(
        client, bbox,
        event.last_seen + timedelta(days=config.post_window_days[0]),
        event.last_seen + timedelta(days=config.post_window_days[1]))

    items = {**pre_items, **post_items}

    def valid_fraction(c: Candidate) -> float:
        import planetary_computer as pc
        scl, _, requested = _read_window(
            pc.sign(items[c.item_id].assets["SCL"]).href, bbox)
        if not requested:
            return 0.0
        # знаменатель — ВСЯ запрошенная область, а не прочитанная её часть
        return float((~np.isin(scl, list(SCL_INVALID))).sum()) / requested

    pair = select_pair(pre, post, config, valid_fraction)
    if pair is None:
        return deferred(
            event.id,
            f"нет пары сцен одного MGRS-тайла с долей валидных пикселей "
            f">= {config.min_valid_fraction:.0%} (кандидатов до: {len(pre)}, после: {len(post)})",
            None, tv)
    return compute_pair(items[pair.before.item_id], items[pair.after.item_id],
                        bbox, config, event.id, pair.tile)


def main(argv=None) -> int:
    import argparse, os, sys
    from src import firms
    from src.confidence import load_thresholds, partition as cpart
    from src.events import build_events, load_event_config
    from src.persistence import (analyse, backfill, load_persistence_config,
                                 partition as ppart)

    ap = argparse.ArgumentParser(prog="python3 -m src.burn")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="посчитать площадь гари для события")
    p.add_argument("--config", required=True)
    p.add_argument("--event-id", help="по умолчанию — крупнейшее завершённое событие")
    p.add_argument("--days", type=int, default=25)
    args = ap.parse_args(argv)

    map_key = os.environ.get("FIRMS_MAP_KEY", "")
    if not map_key:
        print("FIRMS_MAP_KEY is not set in the environment", file=sys.stderr)
        return 2

    region, sources, _ = firms.load_region(args.config)
    dets, _ = backfill(region, sources, map_key, args.days, verbose=False)
    kept, _ = cpart(dets, load_thresholds(args.config))
    pcfg = load_persistence_config(args.config)
    kept, _ = ppart(kept, analyse(kept, pcfg), pcfg)
    events = build_events(kept, load_event_config(args.config))

    if args.event_id:
        chosen = [e for e in events if e.id == args.event_id]
        if not chosen:
            print(f"событие {args.event_id} не найдено", file=sys.stderr)
            return 1
        event = chosen[0]
    else:
        event = max(events, key=lambda e: e.total_frp)

    cfg = load_burn_config(args.config)
    lon, lat = event.centroid
    print(f"событие {event.id}  центр {lat:.3f},{lon:.3f}  "
          f"детекций {len(event.detections)}  FRP {event.total_frp:.0f} МВт")
    print(f"горело {event.first_seen:%Y-%m-%d}..{event.last_seen:%Y-%m-%d}  "
          f"AOI {[round(v,3) for v in event.padded_bbox(cfg.aoi_pad_m)]}")

    res = run_for_event(event, cfg)
    print(f"\nстатус: {res.status.value}  пороги: {res.thresholds_version}")
    if res.status is not BurnStatus.OK:
        print(f"причина: {res.reason}")
        if res.retry_after:
            print(f"повторить после: {res.retry_after}")
        return 0

    print(f"тайл MGRS: {res.mgrs_tile}")
    print(f"сцена до : {res.scene_before_date}  {res.scene_before}")
    print(f"сцена после: {res.scene_after_date}  {res.scene_after}")
    print(f"под маской: {res.masked_fraction:.1%} площади AOI")
    print("\nплощадь по классам тяжести, га:")
    for k in SEVERITY_ORDER:
        print(f"  {k:15s} {res.area_ha[k]:12.2f}")
    print(f"  {'ГАРЬ ВСЕГО':15s} {res.burned_ha:12.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
