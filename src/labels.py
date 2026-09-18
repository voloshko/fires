"""SPEC-11: контрольные продукты гарей как слабые метки и как валидация площади.

Проверено 18.09.2026, и обе проверки изменили спеку:

* **MCD64A1 актуален, но не там, где спека предполагала.** Продукт выпускается
  с задержкой около шести недель: гранулы за июль 2026 собраны 11 сентября.
  Зеркало на Planetary Computer при этом остановилось на августе 2024 —
  заявленный в каталоге открытый временной интервал вводит в заблуждение.
  Берём из первоисточника LP DAAC через AppEEARS: он отдаёт GeoTIFF, нарезанный
  по нужной области, и снимает нужду в драйвере HDF4, которого в этой сборке
  GDAL нет.
* **Fire_CCI не покрывает текущий период вовсе.** FireCCI51 заканчивается
  2020 годом, FireCCIS311 — 2024. Требование сверять площадь «с обоими
  продуктами» для пожаров 2026 года невыполнимо по существу, а не по доступу.
  Для исторических пожаров он остаётся применим как источник слабых меток.

Отдельно: разрешение MCD64A1 — 500 м, один его пиксель равен 25 гектарам.
Сравнивать с нашими 20 метрами напрямую на малых пожарах бессмысленно, поэтому
результат ниже порога помечается **несопоставимым**, а не выдаётся расхождением.
"""
from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import numpy as np

PRODUCT_RESOLUTION_M = 500.0
PRODUCT_PIXEL_HA = (PRODUCT_RESOLUTION_M ** 2) / 10_000.0     # 25 га

# Временное покрытие продуктов Fire_CCI, проверено 18.09.2026.
FIRE_CCI_COVERAGE = {"FireCCI51": (2001, 2020), "FireCCIS311": (2019, 2024),
                     "FireCCILT11": (1982, 2018)}


class LabelStatus(Enum):
    OK = "ok"
    NO_DATA = "no_data"                 # продукт не покрывает область или период
    NOT_COMPARABLE = "not_comparable"   # пожар мельче пикселя продукта
    UNAVAILABLE = "unavailable"         # продукт не существует для этого периода


class LabelsError(RuntimeError):
    """Работа с контрольным продуктом невозможна."""


@dataclass(frozen=True)
class LabelsConfig:
    appeears_url: str = "https://appeears.earthdatacloud.nasa.gov/api"
    product: str = "MCD64A1.061"
    layer: str = "Burn_Date"
    min_comparable_ha: float = 1000.0
    poll_seconds: int = 20
    poll_limit: int = 60
    cache_dir: Path = Path("data/labels")


def load_labels_config(config_path: str | Path) -> LabelsConfig:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8")).get("labels", {})
    return LabelsConfig(
        appeears_url=cfg.get("appeears_url", LabelsConfig.appeears_url),
        product=cfg.get("product", "MCD64A1.061"),
        layer=cfg.get("layer", "Burn_Date"),
        min_comparable_ha=float(cfg.get("min_comparable_ha", 1000.0)),
        poll_seconds=int(cfg.get("poll_seconds", 20)),
        poll_limit=int(cfg.get("poll_limit", 60)),
        cache_dir=Path(cfg.get("cache_dir", "data/labels")))


def fire_cci_covers(when: date, product: str = "FireCCIS311") -> bool:
    """Покрывает ли продукт Fire_CCI эту дату.

    Вынесено в функцию, а не в комментарий: невозможность сверки — часть
    результата, и она должна проверяться кодом, а не памятью автора.
    """
    span = FIRE_CCI_COVERAGE.get(product)
    if span is None:
        raise LabelsError(f"неизвестный продукт Fire_CCI: {product}")
    return span[0] <= when.year <= span[1]


# --- чистые вычисления ---

def burned_mask(burn_date: np.ndarray) -> np.ndarray:
    """Пиксели, помеченные продуктом как горевшие.

    В MCD64A1 значение Burn_Date — порядковый день года горения (1..366).
    Ноль — не горело, отрицательные — нет данных или вода.
    """
    return burn_date > 0


def product_area_ha(burn_date: np.ndarray, pixel_ha: float = PRODUCT_PIXEL_HA) -> float:
    return float(burned_mask(burn_date).sum()) * pixel_ha


@dataclass(frozen=True)
class Comparison:
    """Сопоставление нашей площади с контрольным продуктом.

    `ratio` присутствует только при `status == OK`: отношение, посчитанное для
    пожара мельче пикселя продукта, выглядело бы как измеренное расхождение,
    а является артефактом разрешения.
    """
    event_id: str
    status: LabelStatus
    our_ha: float
    product: str
    product_ha: float | None = None
    ratio: float | None = None
    product_pixels: int | None = None
    reason: str | None = None

    @property
    def comparable(self) -> bool:
        return self.status is LabelStatus.OK


def compare_area(event_id: str, our_ha: float, burn_date: np.ndarray | None,
                 config: LabelsConfig, product: str = "MCD64A1.061",
                 pixel_ha: float = PRODUCT_PIXEL_HA) -> Comparison:
    """Сопоставить нашу площадь с площадью продукта (AC-02, AC-03)."""
    if burn_date is None:
        return Comparison(event_id, LabelStatus.NO_DATA, our_ha, product,
                          reason="продукт не покрывает область или период события")
    if our_ha < config.min_comparable_ha:
        pix = int(burned_mask(burn_date).sum())
        return Comparison(
            event_id, LabelStatus.NOT_COMPARABLE, our_ha, product,
            product_ha=float(pix) * pixel_ha, product_pixels=pix,
            reason=(f"площадь {our_ha:.0f} га сопоставима с пикселем продукта "
                    f"({pixel_ha:.0f} га); порог сравнения {config.min_comparable_ha:.0f} га"))
    pix = int(burned_mask(burn_date).sum())
    p_ha = float(pix) * pixel_ha
    return Comparison(event_id, LabelStatus.OK, our_ha, product,
                      product_ha=p_ha, product_pixels=pix,
                      ratio=(our_ha / p_ha) if p_ha else None)


def summarise(comparisons) -> dict:
    """Распределение расхождений, а не единственное число (AC-04)."""
    ok = [c for c in comparisons if c.comparable and c.ratio]
    ratios = sorted(c.ratio for c in ok)
    by_status: dict[str, int] = {}
    for c in comparisons:
        by_status[c.status.value] = by_status.get(c.status.value, 0) + 1
    out = {"events": len(comparisons), "by_status": by_status, "comparable": len(ok)}
    if ratios:
        n = len(ratios)
        out["ratio"] = {
            "min": round(ratios[0], 3), "median": round(ratios[n // 2], 3),
            "max": round(ratios[-1], 3),
            "mean": round(sum(ratios) / n, 3),
            "our_larger": sum(1 for r in ratios if r > 1),
            "product_larger": sum(1 for r in ratios if r < 1)}
    return out


# --- слой AppEEARS: сеть ---

class Appeears:
    """Клиент AppEEARS. Токен Earthdata здесь не годится: сервис выдаёт свой
    по логину, и это выяснилось только пробой — с Bearer от Earthdata
    task API отвечает 403."""

    def __init__(self, config: LabelsConfig, username: str, password: str,
                 client=None):
        import httpx
        self.cfg = config
        # follow_redirects обязателен: файлы бандла отдаются 302-редиректом
        # на S3, и без этого HTML-страница редиректа сохранялась под
        # именем .tif, а падало это уже в rasterio.
        self.http = client or httpx.Client(timeout=300.0, follow_redirects=True)
        r = self.http.post(f"{config.appeears_url}/login", auth=(username, password))
        if r.status_code != 200:
            raise LabelsError(f"AppEEARS login: HTTP {r.status_code}")
        body = r.json()
        if "token" not in body:
            raise LabelsError(f"AppEEARS login без токена: {str(body)[:160]}")
        self.token = body["token"]
        self.expires = body.get("expiration")

    @property
    def _head(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def submit(self, name: str, bbox, start: date, end: date) -> str:
        west, south, east, north = bbox
        task = {
            "task_type": "area", "task_name": name,
            "params": {
                "dates": [{"startDate": start.strftime("%m-%d-%Y"),
                           "endDate": end.strftime("%m-%d-%Y")}],
                "layers": [{"product": self.cfg.product, "layer": self.cfg.layer}],
                "output": {"format": {"type": "geotiff"}, "projection": "geographic"},
                "geo": {"type": "FeatureCollection", "features": [{
                    "type": "Feature", "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": [[
                        [west, south], [east, south], [east, north],
                        [west, north], [west, south]]]}}]},
            },
        }
        r = self.http.post(f"{self.cfg.appeears_url}/task", json=task, headers=self._head)
        if r.status_code not in (200, 202):
            raise LabelsError(f"AppEEARS task: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["task_id"]

    def status(self, task_id: str) -> str:
        r = self.http.get(f"{self.cfg.appeears_url}/task/{task_id}", headers=self._head)
        if r.status_code != 200:
            raise LabelsError(f"AppEEARS status: HTTP {r.status_code}")
        return r.json().get("status", "unknown")

    def wait(self, task_id: str, verbose: bool = True) -> None:
        for i in range(self.cfg.poll_limit):
            st = self.status(task_id)
            if verbose:
                print(f"  задача {task_id[:8]}: {st} ({i * self.cfg.poll_seconds} с)")
            if st == "done":
                return
            if st in ("error", "expired"):
                raise LabelsError(f"AppEEARS задача завершилась со статусом {st}")
            time.sleep(self.cfg.poll_seconds)
        raise LabelsError(
            f"задача не завершилась за {self.cfg.poll_limit * self.cfg.poll_seconds} с; "
            f"AppEEARS обрабатывает асинхронно, повторите позже с тем же task_id")

    def download_tifs(self, task_id: str, out_dir: Path) -> list[Path]:
        r = self.http.get(f"{self.cfg.appeears_url}/bundle/{task_id}", headers=self._head)
        if r.status_code != 200:
            raise LabelsError(f"AppEEARS bundle: HTTP {r.status_code}")
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in r.json().get("files", []):
            if not f["file_name"].lower().endswith(".tif"):
                continue
            dst = out_dir / Path(f["file_name"]).name
            if not dst.exists():
                with self.http.stream(
                        "GET", f"{self.cfg.appeears_url}/bundle/{task_id}/{f['file_id']}",
                        headers=self._head, follow_redirects=True) as s:
                    with open(dst, "wb") as fh:
                        for chunk in s.iter_bytes():
                            fh.write(chunk)
                _require_tiff(dst)
            saved.append(dst)
        return saved


# Сигнатуры TIFF: little-endian, big-endian и BigTIFF.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _require_tiff(path: Path) -> None:
    """Убедиться, что скачан растр, а не страница редиректа или ошибки.

    AppEEARS отдаёт файлы бандла редиректом на S3. Без follow_redirects под
    именем .tif сохранялась HTML-страница, и падало это только в rasterio —
    далеко от причины. Проверка сигнатуры ловит подмену на месте.
    """
    head = path.read_bytes()[:4]
    if head not in _TIFF_MAGIC:
        preview = path.read_bytes()[:120].decode("utf-8", "replace").replace("\n", " ")
        path.unlink(missing_ok=True)
        raise LabelsError(
            f"{path.name}: скачан не GeoTIFF (сигнатура {head!r}); начало: {preview!r}")


def read_window(tif: Path, bbox) -> np.ndarray | None:
    """Прочитать окно продукта по bbox события. None — покрытия нет."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(tif) as src:
        try:
            win = from_bounds(*bbox, transform=src.transform)
        except Exception:
            return None
        data = src.read(1, window=win)
    return data if data.size else None


def main(argv=None) -> int:
    import argparse, os, sys
    from src import cache
    from src.burn import BurnStatus

    ap = argparse.ArgumentParser(prog="python3 -m src.labels")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare", help="сверить наши площади с контрольным продуктом")
    p.add_argument("--config", required=True)
    p.add_argument("--cache", default="data/demo")
    p.add_argument("--events", type=int, default=10)
    p.add_argument("--task-id", help="использовать готовую задачу AppEEARS")
    args = ap.parse_args(argv)

    cfg = load_labels_config(args.config)
    snap = cache.load(args.cache, args.config)
    ours = [(eid, r) for eid, r in snap.burns.items() if r.status is BurnStatus.OK]
    if not ours:
        print("в кеше нет ни одного посчитанного расчёта гари", file=sys.stderr)
        return 1
    print(f"расчётов гари в кеше: {len(ours)}")

    # период и область по всем событиям, для одной задачи AppEEARS
    evs = {e.id: e for e in snap.events}
    picked = sorted(ours, key=lambda kv: -kv[1].burned_ha)[:args.events]
    lo = min(evs[eid].first_seen.date() for eid, _ in picked)
    hi = max(evs[eid].last_seen.date() for eid, _ in picked)
    boxes = [evs[eid].bbox for eid, _ in picked]
    bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))
    print(f"период {lo}..{hi}, область {[round(v,2) for v in bbox]}")

    for eid, _ in picked:
        if not fire_cci_covers(evs[eid].last_seen.date(), "FireCCIS311"):
            print(f"Fire_CCI: покрытия на {evs[eid].last_seen:%Y} нет ни у одного продукта "
                  f"({', '.join(f'{k} {v[0]}-{v[1]}' for k,v in FIRE_CCI_COVERAGE.items())})")
            break

    user = os.environ.get("EARTHDATA_USERNAME", "")
    pwd = os.environ.get("EARTHDATA_PASSWORD", "")
    if not user or not pwd:
        print("EARTHDATA_USERNAME и EARTHDATA_PASSWORD не заданы", file=sys.stderr)
        return 2

    ae = Appeears(cfg, user, pwd)
    task_id = args.task_id or ae.submit(f"fires-{lo}-{hi}", bbox, lo, hi)
    print(f"задача AppEEARS: {task_id}")
    ae.wait(task_id)
    tifs = ae.download_tifs(task_id, cfg.cache_dir / task_id)
    print(f"получено растров: {len(tifs)}")
    if not tifs:
        print("AppEEARS не вернул ни одного растра", file=sys.stderr)
        return 1

    # AppEEARS кладёт в бандл не только запрошенный слой, но и QA. Смешивать их
    # нельзя: в QA значение 3 стоит по всей площади, и как «гарь» оно давало
    # шестикратное завышение. Берём строго файлы запрошенного слоя.
    layer_tifs = [t for t in tifs if f"_{cfg.layer}_" in t.name]
    other = [t.name for t in tifs if t not in layer_tifs]
    print(f"растров слоя {cfg.layer}: {len(layer_tifs)}"
          + (f"; отброшены посторонние слои: {', '.join(other)}" if other else ""))
    if not layer_tifs:
        print(f"в бандле нет ни одного растра слоя {cfg.layer}", file=sys.stderr)
        return 1

    comps = []
    for eid, r in picked:
        # пожар мог гореть на стыке месяцев: горело хотя бы в одном растре
        data = None
        for t in layer_tifs:
            w = read_window(t, evs[eid].bbox)
            if w is None or not w.size:
                continue
            data = w if data is None or w.shape != data.shape else np.maximum(data, w)
        comps.append(compare_area(eid, r.burned_ha, data, cfg, product=cfg.product))

    print(f"\n{'событие':22s}{'наша, га':>12s}{'продукт, га':>14s}{'отношение':>12s}  статус")
    for c in comps:
        ph = f"{c.product_ha:.0f}" if c.product_ha is not None else "—"
        rt = f"{c.ratio:.2f}" if c.ratio else "—"
        print(f"{c.event_id:22s}{c.our_ha:12.0f}{ph:>14s}{rt:>12s}  {c.status.value}")
    print("\n" + json.dumps(summarise(comps), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
