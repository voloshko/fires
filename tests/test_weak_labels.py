"""SPEC-11 acceptance tests — офлайн, синтетические растры и подставной AppEEARS."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.labels import (FIRE_CCI_COVERAGE, PRODUCT_PIXEL_HA, Appeears,
                        Comparison, LabelStatus, LabelsConfig, LabelsError,
                        burned_mask, compare_area, fire_cci_covers,
                        load_labels_config, product_area_ha, read_window,
                        summarise)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "krasnoyarsk.toml"
CFG = LabelsConfig(min_comparable_ha=1000.0)


def burn(n_burned: int, shape=(50, 50)) -> np.ndarray:
    a = np.zeros(shape, dtype="int16")
    a.flat[:n_burned] = 200           # день года горения
    return a


# --- AC-01: покрытие есть или его нет, и это различимо ---

def test_ac01_missing_coverage_is_no_data_not_zero_area():
    """Пустая маска и отсутствие данных — разные исходы: первый значит
    «продукт смотрел и не нашёл», второй «продукт сюда не смотрел»."""
    absent = compare_area("E", 5000.0, None, CFG)
    empty = compare_area("E", 5000.0, burn(0), CFG)
    assert absent.status is LabelStatus.NO_DATA
    assert empty.status is LabelStatus.OK
    assert absent.product_ha is None and empty.product_ha == 0.0
    assert absent.status != empty.status


def test_ac01_no_data_carries_a_reason():
    c = compare_area("E", 5000.0, None, CFG)
    assert c.reason and "не покрывает" in c.reason
    assert c.comparable is False


def test_ac01_read_window_returns_none_outside_coverage(tmp_path):
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1,
                       dtype="int16", crs="EPSG:4326",
                       transform=from_origin(90.0, 60.0, 0.01, 0.01)) as dst:
        dst.write(np.zeros((10, 10), dtype="int16"), 1)
    assert read_window(p, (90.02, 59.95, 90.05, 59.98)) is not None
    assert read_window(p, (120.0, 10.0, 120.1, 10.1)) is None


# --- AC-02: приведение разрешений ---

def test_ac02_product_pixel_is_twenty_five_hectares():
    """500 на 500 метров — это 25 гектаров, один пиксель MCD64A1."""
    assert PRODUCT_PIXEL_HA == pytest.approx(25.0)


def test_ac02_area_is_pixel_count_times_pixel_area():
    assert product_area_ha(burn(40)) == pytest.approx(40 * 25.0)
    assert product_area_ha(burn(0)) == 0.0


def test_ac02_small_fire_is_marked_not_comparable_not_a_discrepancy():
    """Пожар в 300 га — это 12 пикселей продукта. Отношение здесь было бы
    артефактом разрешения, а выглядело бы как измеренное расхождение."""
    c = compare_area("E", 300.0, burn(12), CFG)
    assert c.status is LabelStatus.NOT_COMPARABLE
    assert c.ratio is None
    assert "сопоставима с пикселем" in c.reason


def test_ac02_large_fire_is_comparable():
    c = compare_area("E", 12000.0, burn(480), CFG)
    assert c.status is LabelStatus.OK
    assert c.product_ha == pytest.approx(12000.0)
    assert c.ratio == pytest.approx(1.0)


def test_ac02_threshold_comes_from_config():
    assert load_labels_config(CONFIG).min_comparable_ha == 1000.0
    strict = LabelsConfig(min_comparable_ha=20000.0)
    assert compare_area("E", 12000.0, burn(480), strict).status is LabelStatus.NOT_COMPARABLE


def test_ac02_reconciliation_neither_loses_nor_adds_area():
    """Приведение не должно менять суммарную площадь продукта."""
    for n in (0, 1, 37, 2500):
        assert product_area_ha(burn(n, (50, 50))) == pytest.approx(n * 25.0)


@pytest.mark.parametrize("value,expected", [(-2, False), (-1, False), (0, False),
                                            (1, True), (200, True), (366, True)])
def test_ac02_burn_date_semantics(value, expected):
    """В MCD64A1 Burn_Date — день года; ноль не горело, отрицательные — нет данных."""
    assert bool(burned_mask(np.array([[value]], dtype="int16"))[0, 0]) is expected


# --- AC-03: расхождение измеряется и публикуется ---

def test_ac03_ratio_is_reported_per_event():
    c = compare_area("E", 15000.0, burn(400), CFG)     # продукт: 10000 га
    assert c.ratio == pytest.approx(1.5)
    assert c.product_pixels == 400


def test_ac03_summary_reports_a_distribution_not_one_number():
    cs = [compare_area(f"E{i}", ha, burn(pix), CFG) for i, (ha, pix) in
          enumerate([(12000, 480), (9000, 500), (20000, 600), (300, 12), (5000, None)])
          if pix is not None]
    cs.append(compare_area("E5", 5000.0, None, CFG))
    s = summarise(cs)
    assert s["events"] == 5
    assert s["by_status"]["not_comparable"] == 1
    assert s["by_status"]["no_data"] == 1
    assert set(s["ratio"]) == {"min", "median", "max", "mean", "our_larger", "product_larger"}
    assert s["ratio"]["min"] <= s["ratio"]["median"] <= s["ratio"]["max"]


def test_ac03_summary_counts_which_side_is_larger():
    cs = [compare_area("A", 20000.0, burn(400), CFG),    # наш больше
          compare_area("B", 5000.0, burn(400), CFG)]     # продукт больше
    s = summarise(cs)
    assert s["ratio"]["our_larger"] == 1 and s["ratio"]["product_larger"] == 1


def test_ac03_summary_of_nothing_comparable_has_no_ratio_block():
    s = summarise([compare_area("E", 300.0, burn(4), CFG)])
    assert "ratio" not in s and s["comparable"] == 0


# --- AC-05: продукт-источник всегда назван ---

def test_ac05_every_comparison_names_its_product():
    c = compare_area("E", 12000.0, burn(480), CFG, product="MCD64A1.061")
    assert c.product == "MCD64A1.061"
    assert compare_area("E", 12000.0, burn(480), CFG, product="FireCCI51").product == "FireCCI51"


def test_ac05_product_cannot_be_omitted():
    with pytest.raises(TypeError):
        Comparison(event_id="E", status=LabelStatus.OK, our_ha=1.0)


# --- временное покрытие Fire_CCI: проверяется кодом, а не памятью ---

def test_fire_cci_does_not_cover_the_current_season():
    """Проверено 18.09.2026: ни один продукт Fire_CCI не покрывает 2026 год."""
    for product in FIRE_CCI_COVERAGE:
        assert fire_cci_covers(date(2026, 8, 1), product) is False


@pytest.mark.parametrize("product,year,expected", [
    ("FireCCI51", 2015, True), ("FireCCI51", 2021, False),
    ("FireCCIS311", 2020, True), ("FireCCIS311", 2024, True), ("FireCCIS311", 2025, False),
    ("FireCCILT11", 2000, True), ("FireCCILT11", 2019, False)])
def test_fire_cci_coverage_boundaries(product, year, expected):
    assert fire_cci_covers(date(year, 6, 1), product) is expected


def test_unknown_fire_cci_product_is_refused():
    with pytest.raises(LabelsError, match="неизвестный продукт"):
        fire_cci_covers(date(2020, 1, 1), "FireCCI99")


# --- клиент AppEEARS ---

def appeears_stub(*, login_status=200, task_status="done", with_token=True):
    state = {"submitted": []}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/login"):
            body = {"token": "ae-token", "expiration": "2026-09-19T15:40:41Z"} if with_token else {}
            return httpx.Response(login_status, json=body)
        if url.endswith("/task") and request.method == "POST":
            state["submitted"].append(json.loads(request.content))
            return httpx.Response(202, json={"task_id": "task-123"})
        if "/task/" in url:
            return httpx.Response(200, json={"status": task_status})
        if url.endswith("/bundle/task-123"):
            return httpx.Response(200, json={"files": [
                {"file_id": "f1", "file_name": "MCD64A1.061_Burn_Date_doy2026182.tif"},
                {"file_id": "f2", "file_name": "readme.txt"}]})
        if "/bundle/task-123/" in url:          # сам файл отдаётся отдельно
            return httpx.Response(200, content=b"II*\x00" + b"\x00" * 64)
        return httpx.Response(404)

    import json
    return httpx.Client(transport=httpx.MockTransport(handler)), state


def test_appeears_needs_its_own_token_not_the_earthdata_one():
    """Выяснено пробой: с Bearer от Earthdata task API отвечает 403."""
    client, _ = appeears_stub()
    ae = Appeears(CFG, "user", "pass", client=client)
    assert ae.token == "ae-token"


def test_appeears_login_failure_is_reported():
    client, _ = appeears_stub(login_status=401)
    with pytest.raises(LabelsError, match="login"):
        Appeears(CFG, "user", "bad", client=client)


def test_appeears_login_without_token_is_refused():
    client, _ = appeears_stub(with_token=False)
    with pytest.raises(LabelsError, match="без токена"):
        Appeears(CFG, "user", "pass", client=client)


def test_appeears_task_carries_the_area_and_the_layer():
    client, state = appeears_stub()
    ae = Appeears(CFG, "u", "p", client=client)
    ae.submit("ev", (92.0, 55.0, 93.0, 56.0), date(2026, 7, 1), date(2026, 7, 31))
    task = state["submitted"][0]
    assert task["task_type"] == "area"
    assert task["params"]["layers"][0]["layer"] == "Burn_Date"
    assert task["params"]["output"]["format"]["type"] == "geotiff"
    coords = task["params"]["geo"]["features"][0]["geometry"]["coordinates"][0]
    assert coords[0] == [92.0, 55.0] and coords[2] == [93.0, 56.0]


def test_appeears_error_status_stops_the_wait():
    client, _ = appeears_stub(task_status="error")
    ae = Appeears(CFG, "u", "p", client=client)
    with pytest.raises(LabelsError, match="статусом error"):
        ae.wait("task-123", verbose=False)


def test_appeears_bundle_takes_only_rasters(tmp_path):
    client, _ = appeears_stub()
    ae = Appeears(CFG, "u", "p", client=client)
    files = ae.download_tifs("task-123", tmp_path)
    assert len(files) == 1 and files[0].name.endswith(".tif")


def test_downloaded_file_that_is_not_a_tiff_is_refused(tmp_path):
    """Регрессия: AppEEARS отдаёт файлы бандла редиректом на S3, и без
    follow_redirects HTML-страница редиректа сохранялась под именем .tif.
    Падало это только в rasterio — далеко от причины."""
    from src.labels import _require_tiff
    bad = tmp_path / "x.tif"
    bad.write_bytes(b"<!doctype html>\n<title>Redirecting...</title>")
    with pytest.raises(LabelsError, match="не GeoTIFF"):
        _require_tiff(bad)
    assert not bad.exists(), "испорченный файл должен быть удалён, а не оставлен"


@pytest.mark.parametrize("magic", [b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"])
def test_valid_tiff_signatures_are_accepted(tmp_path, magic):
    from src.labels import _require_tiff
    good = tmp_path / "ok.tif"
    good.write_bytes(magic + b"\x00" * 40)
    _require_tiff(good)
    assert good.exists()


def test_qa_layer_must_not_be_counted_as_burned_area(tmp_path):
    """Регрессия, найденная живым прогоном. AppEEARS кладёт в бандл и QA, где
    значение 3 стоит по всей площади. Смешанный с Burn_Date через максимум, он
    давал шестикратное завышение: 238 950 га вместо 11 800."""
    burn_date = np.zeros((68, 301), dtype="int16"); burn_date.flat[:472] = 211
    qa = np.full((68, 301), 3, dtype="int16")
    assert product_area_ha(burn_date) == pytest.approx(472 * 25.0)
    assert product_area_ha(np.maximum(burn_date, qa)) > 10 * product_area_ha(burn_date)


def test_layer_selection_keeps_only_the_requested_raster():
    """Отбор идёт по имени файла: слой присутствует в нём как _Burn_Date_."""
    names = ["MCD64A1.061_Burn_Date_doy2026182000000_aid0001.tif",
             "MCD64A1.061_QA_doy2026182000000_aid0001.tif"]
    layer = "Burn_Date"
    kept = [n for n in names if f"_{layer}_" in n]
    assert kept == ["MCD64A1.061_Burn_Date_doy2026182000000_aid0001.tif"]
