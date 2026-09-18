"""SPEC-5 acceptance tests.

Тесты AC-01..AC-05 офлайн: кэш-тайлы синтезируются локально, сеть не трогается.
Отдельная группа помечена `live` и работает по реальному кэшу, если он построен.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from src import firms
from src.masks import (CROPLAND_CLASS, CroplandConfig, CroplandMask, LandClass,
                       TILE_DEG, Window, load_cropland_config, tile_name, tile_origin)
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "krasnoyarsk.toml"

SPRING = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)    # внутри окна палов
SUMMER = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)   # вне окна
WINDOWS = (Window.parse("04-01/05-31"), Window.parse("09-01/10-31"))


def write_tile(cache_dir: Path, lon: float, lat: float, *, epoch=2021,
               side=300, cropland_half=True) -> Path:
    """Синтетический кэш-тайл: западная половина — пашня, восточная — нет."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    x, y = tile_origin(lon, lat)
    data = np.zeros((side, side), dtype="uint8")
    if cropland_half:
        data[:, : side // 2] = 1
    path = cache_dir / f"cropland_{epoch}_{tile_name(lon, lat)}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=side, width=side,
                       count=1, dtype="uint8", crs="EPSG:4326", compress="deflate",
                       transform=from_bounds(x, y, x + TILE_DEG, y + TILE_DEG,
                                             side, side)) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def mask(tmp_path):
    """Маска с готовым кэшем и запретом на сеть — любой сетевой вызов упадёт."""
    write_tile(tmp_path, 91.0, 55.0)
    cfg = CroplandConfig(cache_dir=tmp_path, burn_windows=WINDOWS)
    with CroplandMask(cfg, offline=True) as m:
        yield m


# точки внутри тайла N54E090: западная половина = пашня, восточная = нет
CROP_POINT = (90.5, 55.0)
FOREST_POINT = (92.5, 55.0)


def det(lon, lat, when, **kw) -> firms.Detection:
    base = dict(latitude=lat, longitude=lon, brightness=330.0, frp=5.0,
                confidence="n", acquired_at=when, sensor="VIIRS_SNPP_NRT",
                satellite="N", daynight="D")
    base.update(kw)
    return firms.Detection(**base)


# --- AC-01: класс 40 внутри окна vs вне окна ---

def test_ac01_same_point_flagged_inside_the_window(mask):
    normal, agri = mask.flag([det(*CROP_POINT, SPRING)])
    assert len(agri) == 1 and normal == []


def test_ac01_same_point_not_flagged_outside_the_window(mask):
    """Та же точка, другая дата — не сельхозпал."""
    normal, agri = mask.flag([det(*CROP_POINT, SUMMER)])
    assert len(normal) == 1 and agri == []


def test_ac01_forest_point_inside_the_window_is_not_flagged(mask):
    normal, agri = mask.flag([det(*FOREST_POINT, SPRING)])
    assert len(normal) == 1 and agri == []


def test_ac01_classify_distinguishes_cropland_from_not(mask):
    assert mask.classify(*CROP_POINT) is LandClass.CROPLAND
    assert mask.classify(*FOREST_POINT) is LandClass.NOT_CROPLAND


# --- AC-02: окно — параметр конфигурации ---

def test_ac02_window_comes_from_config():
    cfg = load_cropland_config(CONFIG)
    assert cfg.burn_windows == WINDOWS
    assert cfg.epoch == 2021 and cfg.overview == 16


def test_ac02_changing_the_window_changes_the_verdict(tmp_path):
    write_tile(tmp_path, 91.0, 55.0)
    july_only = CroplandConfig(cache_dir=tmp_path,
                               burn_windows=(Window.parse("07-01/07-31"),))
    with CroplandMask(july_only, offline=True) as m:
        assert m.flag([det(*CROP_POINT, SPRING)])[1] == []    # май больше не окно
        assert len(m.flag([det(*CROP_POINT, SUMMER)])[1]) == 1  # июль стал окном


@pytest.mark.parametrize("when,expected", [
    (date(2026, 3, 31), False), (date(2026, 4, 1), True),    # границы весеннего
    (date(2026, 5, 31), True), (date(2026, 6, 1), False),
    (date(2026, 8, 31), False), (date(2026, 9, 1), True),    # границы осеннего
    (date(2026, 10, 31), True), (date(2026, 11, 1), False),
])
def test_ac02_window_boundaries_both_sides(mask, when, expected):
    assert mask.in_burn_window(when) is expected


def test_ac02_window_spanning_new_year_works():
    w = Window.parse("12-15/01-15")
    assert w.contains(date(2026, 12, 20)) and w.contains(date(2026, 1, 5))
    assert not w.contains(date(2026, 6, 1))


def test_ac02_malformed_window_is_rejected():
    with pytest.raises(ValueError, match="MM-DD/MM-DD"):
        Window.parse("april to may")


# --- AC-03: кэш, второй прогон без сети ---

def test_ac03_second_run_works_with_network_forbidden(tmp_path):
    """offline=True делает любой сетевой вызов исключением."""
    write_tile(tmp_path, 91.0, 55.0)
    cfg = CroplandConfig(cache_dir=tmp_path, burn_windows=WINDOWS)
    with CroplandMask(cfg, offline=True) as m:
        assert m.classify(*CROP_POINT) is LandClass.CROPLAND
        assert m.network_calls == 0


def test_ac03_missing_tile_offline_raises_instead_of_silently_downloading(tmp_path):
    cfg = CroplandConfig(cache_dir=tmp_path, burn_windows=WINDOWS)
    with CroplandMask(cfg, offline=True) as m:
        with pytest.raises(RuntimeError, match="offline=True"):
            m.classify(100.5, 60.0)


def test_ac03_repeated_lookups_do_not_reopen_the_tile(mask):
    for _ in range(50):
        mask.classify(*CROP_POINT)
    assert mask.network_calls == 0


# --- AC-04: помеченные не теряются ---

def test_ac04_flagged_detections_stay_in_the_store(mask, tmp_path):
    dets = [det(*CROP_POINT, SPRING), det(*FOREST_POINT, SPRING)]
    with Store(":memory:") as s:
        s.add_detections(dets)
        normal, agri = mask.flag(dets)
        assert len(normal) == 1 and len(agri) == 1
        assert s.count_detections() == 2      # ни одна строка не удалена


def test_ac04_service_layer_can_retrieve_the_flagged_ones(mask):
    normal, agri = mask.flag([det(*CROP_POINT, SPRING), det(*FOREST_POINT, SPRING)])
    assert (agri[0].longitude, agri[0].latitude) == CROP_POINT


# --- AC-05: вне покрытия — UNKNOWN, а не "не сельхоз" ---

def test_ac05_point_outside_coverage_is_unknown_not_not_cropland(tmp_path):
    """Схлопывание UNKNOWN в NOT_CROPLAND пометило бы всю Арктику
    как «проверено, не сельхоз»."""
    write_tile(tmp_path, 91.0, 55.0)
    cfg = CroplandConfig(cache_dir=tmp_path, burn_windows=WINDOWS)
    with CroplandMask(cfg, offline=False) as m:
        m._open[tile_name(105.0, 76.0)] = None      # тайла нет — суши нет
        result = m.classify(105.0, 76.0)
    assert result is LandClass.UNKNOWN
    assert result is not LandClass.NOT_CROPLAND


def test_ac05_unknown_is_never_flagged_as_agricultural(tmp_path):
    write_tile(tmp_path, 91.0, 55.0)
    cfg = CroplandConfig(cache_dir=tmp_path, burn_windows=WINDOWS)
    with CroplandMask(cfg, offline=False) as m:
        m._open[tile_name(105.0, 76.0)] = None
        normal, agri = m.flag([det(105.0, 76.0, SPRING)])
    assert len(normal) == 1 and agri == []


def test_ac05_three_outcomes_are_distinct():
    assert len({LandClass.CROPLAND, LandClass.NOT_CROPLAND, LandClass.UNKNOWN}) == 3


# --- служебное ---

@pytest.mark.parametrize("lon,lat,expected", [
    (90.5, 55.0, "N54E090"), (93.1, 56.9, "N54E093"),
    (82.0, 51.0, "N51E081"), (108.9, 77.9, "N75E108"),
])
def test_tile_naming_matches_worldcover_convention(lon, lat, expected):
    assert tile_name(lon, lat) == expected


# --- LIVE: по реально построенному кэшу ---

REAL_CACHE = ROOT / "data" / "worldcover" / "cropland_2021_N54E090.tif"


@pytest.mark.skipif(not REAL_CACHE.exists(),
                    reason="реальный кэш не построен: python3 -m src.masks fetch-worldcover")
def test_live_real_tile_has_plausible_cropland_fraction():
    with rasterio.open(REAL_CACHE) as ds:
        frac = float(ds.read(1).mean())
    # юг края — земледельческая зона; ноль означал бы, что класс 40 не читается,
    # а близкое к единице — что маска захватила не то
    assert 0.01 < frac < 0.5, f"доля пашни {frac:.2%} неправдоподобна"


@pytest.mark.skipif(not REAL_CACHE.exists(), reason="реальный кэш не построен")
def test_live_real_tile_classifies_both_ways():
    cfg = load_cropland_config(CONFIG)
    with CroplandMask(cfg, offline=True) as m:
        with rasterio.open(REAL_CACHE) as ds:
            data = ds.read(1)
            crop = np.argwhere(data == 1)[0]
            bare = np.argwhere(data == 0)[0]
            lon_c, lat_c = ds.xy(int(crop[0]), int(crop[1]))
            lon_b, lat_b = ds.xy(int(bare[0]), int(bare[1]))
        assert m.classify(lon_c, lat_c) is LandClass.CROPLAND
        assert m.classify(lon_b, lat_b) is LandClass.NOT_CROPLAND
        assert m.network_calls == 0        # всё с кэша


@pytest.mark.skipif(not REAL_CACHE.exists(), reason="реальный кэш не построен")
def test_live_detection_on_real_cropland_is_flagged_end_to_end():
    """Сквозной путь на настоящем WorldCover, а не на синтетике.

    Живой прогон по FIRMS дал ноль помеченных точек — это истинный отрицательный
    (в южном поясе было всего 12 детекций, пашня занимает единицы процентов).
    Но «ноль» не доказывает, что фильтр вообще способен сработать. Этот тест
    берёт реальный пиксель пашни из кэша и проверяет, что детекция на нём
    внутри окна палов действительно помечается, а вне окна — нет.
    """
    cfg = load_cropland_config(CONFIG)
    with rasterio.open(REAL_CACHE) as ds:
        row, col = np.argwhere(ds.read(1) == 1)[0]
        lon, lat = ds.xy(int(row), int(col))

    with CroplandMask(cfg, offline=True) as m:
        assert m.classify(lon, lat) is LandClass.CROPLAND
        in_window = datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)   # окно палов
        out_window = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)  # вне окна
        assert len(m.flag([det(lon, lat, in_window)])[1]) == 1
        assert m.flag([det(lon, lat, out_window)])[1] == []
        assert m.network_calls == 0
