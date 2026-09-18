"""SPEC-5: маска сельхозземель из ESA WorldCover.

ТЗ требовало «сезонную маску сельхозземель (открытые слои землепользования)»,
не называя слой. Готовый нашёлся в Planetary Computer: коллекция
`esa-worldcover`, 10 м, CC-BY-4.0, анонимный доступ, класс 40 — Cropland.

Три решения, определяющих модуль:

* **Сезонность — календарное окно, а не смена слоя.** WorldCover статичен
  (эпохи 2020 и 2021, обновлений не будет), поэтому «сезонная маска» иначе
  нереализуема в принципе.
* **Кэш строится по тайлам и лениво.** Регион покрывают ~81 тайл 3x3 градуса
  по 36000x36000 пикселей; качать их целиком незачем. Тайл скачивается, когда
  в него впервые попадает детекция, и кладётся локально уменьшенным.
* **Три исхода, а не два.** Точка вне покрытия слоя даёт `UNKNOWN`, а не
  `NOT_CROPLAND`: отсутствие данных не равно отрицательному ответу. Схлопывание
  этих случаев молча пометило бы всю Арктику как «не сельхоз» и выглядело бы
  как работающая проверка.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "esa-worldcover"
CROPLAND_CLASS = 40
TILE_DEG = 3           # размер тайла WorldCover
_FULL_TILE_PX = 36000  # сторона тайла при полном разрешении 10 м


class LandClass(Enum):
    CROPLAND = "cropland"
    NOT_CROPLAND = "not_cropland"
    UNKNOWN = "unknown"      # слой не покрывает точку — это НЕ «не сельхоз»


@dataclass(frozen=True)
class Window:
    """Календарное окно палов, без привязки к году."""
    start: tuple[int, int]   # (месяц, день)
    end: tuple[int, int]

    @classmethod
    def parse(cls, spec: str) -> "Window":
        try:
            a, b = spec.split("/")
            return cls(tuple(int(x) for x in a.split("-")),
                       tuple(int(x) for x in b.split("-")))
        except Exception:
            raise ValueError(
                f"burn window must look like 'MM-DD/MM-DD', got {spec!r}") from None

    def contains(self, when: date) -> bool:
        md = (when.month, when.day)
        if self.start <= self.end:
            return self.start <= md <= self.end
        return md >= self.start or md <= self.end   # окно через Новый год


@dataclass(frozen=True)
class CroplandConfig:
    cache_dir: Path = Path("data/worldcover")
    epoch: int = 2021
    overview: int = 16
    burn_windows: tuple[Window, ...] = field(default_factory=tuple)


def load_cropland_config(config_path: str | Path) -> CroplandConfig:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8")).get("cropland", {})
    return CroplandConfig(
        cache_dir=Path(cfg.get("cache_dir", "data/worldcover")),
        epoch=int(cfg.get("epoch", 2021)),
        overview=int(cfg.get("overview", 16)),
        burn_windows=tuple(Window.parse(s) for s in cfg.get("burn_windows", [])),
    )


def tile_origin(lon: float, lat: float) -> tuple[int, int]:
    """Юго-западный угол тайла WorldCover, содержащего точку."""
    import math
    return (int(math.floor(lon / TILE_DEG) * TILE_DEG),
            int(math.floor(lat / TILE_DEG) * TILE_DEG))


def tile_name(lon: float, lat: float) -> str:
    x, y = tile_origin(lon, lat)
    return f"{'N' if y >= 0 else 'S'}{abs(y):02d}{'E' if x >= 0 else 'W'}{abs(x):03d}"


class CroplandMask:
    """Маска пашни с ленивым потайловым кэшем.

    `offline=True` запрещает любые обращения к сети: используется, чтобы
    доказать, что повторный прогон действительно работает с кэша (AC-03).
    """

    def __init__(self, config: CroplandConfig | None = None, *, offline: bool = False):
        self.config = config or CroplandConfig()
        self.offline = offline
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, rasterio.DatasetReader | None] = {}
        self.network_calls = 0      # наблюдаемый счётчик для проверки AC-03

    def cache_path(self, lon: float, lat: float) -> Path:
        return self.config.cache_dir / f"cropland_{self.config.epoch}_{tile_name(lon, lat)}.tif"

    # --- построение кэша ---

    def _download_tile(self, lon: float, lat: float) -> Path | None:
        """Скачать тайл уменьшенным и сохранить булеву маску пашни."""
        if self.offline:
            raise RuntimeError(
                f"offline=True, но тайл {tile_name(lon, lat)} отсутствует в кэше "
                f"{self.config.cache_dir}")
        import planetary_computer as pc
        from pystac_client import Client

        self.network_calls += 1
        client = Client.open(STAC_URL, modifier=pc.sign_inplace)
        x, y = tile_origin(lon, lat)
        items = [
            it for it in client.search(
                collections=[COLLECTION],
                bbox=[x + 0.01, y + 0.01, x + TILE_DEG - 0.01, y + TILE_DEG - 0.01],
            ).items()
            if str(self.config.epoch) in it.id
        ]
        if not items:
            return None                      # суши здесь нет — останется UNKNOWN

        item = items[0]
        side = _FULL_TILE_PX // self.config.overview
        with rasterio.open(item.assets["map"].href) as src:
            data = src.read(1, out_shape=(side, side))
            bounds = src.bounds
        mask = (data == CROPLAND_CLASS).astype("uint8")

        out = self.cache_path(lon, lat)
        with rasterio.open(
            out, "w", driver="GTiff", height=side, width=side, count=1,
            dtype="uint8", crs="EPSG:4326", compress="deflate",
            transform=from_bounds(*bounds, side, side),
        ) as dst:
            dst.write(mask, 1)
            dst.update_tags(source_item=item.id, worldcover_class=CROPLAND_CLASS,
                            overview=self.config.overview)
        return out

    def _dataset(self, lon: float, lat: float):
        key = tile_name(lon, lat)
        if key not in self._open:
            path = self.cache_path(lon, lat)
            if not path.exists():
                path = self._download_tile(lon, lat)
            self._open[key] = rasterio.open(path) if path and path.exists() else None
        return self._open[key]

    # --- использование ---

    def classify(self, lon: float, lat: float) -> LandClass:
        """CROPLAND / NOT_CROPLAND / UNKNOWN (AC-01, AC-05)."""
        ds = self._dataset(lon, lat)
        if ds is None:
            return LandClass.UNKNOWN
        row, col = ds.index(lon, lat)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            return LandClass.UNKNOWN
        value = ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        return LandClass.CROPLAND if value else LandClass.NOT_CROPLAND

    def in_burn_window(self, when: date | datetime) -> bool:
        d = when.date() if isinstance(when, datetime) else when
        return any(w.contains(d) for w in self.config.burn_windows)

    def flag(self, detections):
        """Разделить на обычные и вероятные сельхозпалы (AC-01, AC-04).

        Хранилище не трогается: это представление, как и в SPEC-3.
        """
        normal, agricultural = [], []
        for d in detections:
            suspect = (self.in_burn_window(d.acquired_at)
                       and self.classify(d.longitude, d.latitude) is LandClass.CROPLAND)
            (agricultural if suspect else normal).append(d)
        return normal, agricultural

    def close(self) -> None:
        for ds in self._open.values():
            if ds is not None:
                ds.close()
        self._open.clear()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def main(argv=None) -> int:
    import argparse, sys
    ap = argparse.ArgumentParser(prog="python3 -m src.masks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch-worldcover", help="pre-warm the cropland tile cache")
    p.add_argument("--config", required=True)
    p.add_argument("--bbox", help="west,south,east,north; по умолчанию — регион из конфига")
    args = ap.parse_args(argv)

    from src.firms import load_region
    region, _, _ = load_region(args.config)
    bbox = ([float(v) for v in args.bbox.split(",")] if args.bbox else list(region.bbox))
    cfg = load_cropland_config(args.config)

    west, south, east, north = bbox
    mask = CroplandMask(cfg)
    built, empty = 0, 0
    import math
    for y in range(int(math.floor(south / TILE_DEG) * TILE_DEG), int(north), TILE_DEG):
        for x in range(int(math.floor(west / TILE_DEG) * TILE_DEG), int(east), TILE_DEG):
            lon, lat = x + TILE_DEG / 2, y + TILE_DEG / 2
            path = mask.cache_path(lon, lat)
            if path.exists():
                continue
            if mask._download_tile(lon, lat):
                built += 1
                with rasterio.open(path) as ds:
                    frac = float(ds.read(1).mean())
                print(f"  {tile_name(lon, lat)}  cropland {frac:6.2%}  {path}")
            else:
                empty += 1
                print(f"  {tile_name(lon, lat)}  нет данных (суши нет)")
    print(f"построено тайлов: {built}, пустых: {empty}, кэш: {cfg.cache_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
