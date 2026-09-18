"""SPEC-4: фильтр временной персистентности и слой теплоисточников.

Газовые факелы Западной и Средней Сибири горят при ~1800 K — вдвое горячее
биомассы — и штатными алгоритмами MOD14/VNP14 отсеиваются плохо. Признак,
доступный без сырой радиометрии: **факел горит неделями в одной точке, пожар
движется**. Дополнительный признак — стабильность мощности: FRP факела почти
постоянна, FRP пожара скачет.

Три свойства, определяющие модуль:

* **«Недостаточно истории» — отдельный исход, а не пустой результат.** Пока
  накоплено меньше окна, фильтр обязан это сказать. Пустой список выглядел бы
  как «факелов нет» и создавал ложное впечатление работающей фильтрации.
* **Глубина истории публикуется как измеримая величина** (`history_days`):
  потребитель должен уметь узнать, работает фильтр или ещё нет.
* **Помеченные точки не удаляются.** Как и в SPEC-3/SPEC-5, фильтр является
  представлением; хранилище append-only.

Кластеризация — сеточный индекс с проверкой соседних ячеек: детекция
присоединяется к кластеру, если попадает в радиус от его центроида. Расстояние
считается равнопромежуточным приближением — на масштабе 500 м его погрешность
пренебрежима, а haversine здесь был бы лишней точностью.
"""
from __future__ import annotations

import math
import statistics
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

EARTH_M_PER_DEG = 111_320.0


@dataclass(frozen=True)
class PersistenceConfig:
    window_days: int = 14
    radius_m: float = 500.0
    min_distinct_days: int = 10
    max_frp_cv: float | None = 0.6

    def __post_init__(self):
        if self.min_distinct_days > self.window_days:
            raise ValueError(
                f"min_distinct_days ({self.min_distinct_days}) cannot exceed "
                f"window_days ({self.window_days})")


def load_persistence_config(config_path: str | Path) -> PersistenceConfig:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8")).get("persistence", {})
    return PersistenceConfig(
        window_days=int(cfg.get("window_days", 14)),
        radius_m=float(cfg.get("radius_m", 500.0)),
        min_distinct_days=int(cfg.get("min_distinct_days", 10)),
        max_frp_cv=cfg.get("max_frp_cv", 0.6),
    )


@dataclass
class Cluster:
    """Группа детекций, лежащих в одном месте."""
    latitude: float
    longitude: float
    detections: list = field(default_factory=list)

    @property
    def days(self) -> set[date]:
        return {d.acquired_at.date() for d in self.detections}

    @property
    def frp_values(self) -> list[float]:
        return [d.frp for d in self.detections if d.frp is not None]

    @property
    def frp_cv(self) -> float | None:
        """Коэффициент вариации FRP. None — если считать не из чего."""
        v = self.frp_values
        if len(v) < 2:
            return None
        mean = statistics.fmean(v)
        return statistics.stdev(v) / mean if mean else None

    def add(self, d) -> None:
        n = len(self.detections)
        self.latitude = (self.latitude * n + d.latitude) / (n + 1)
        self.longitude = (self.longitude * n + d.longitude) / (n + 1)
        self.detections.append(d)


@dataclass(frozen=True)
class PersistentSource:
    latitude: float
    longitude: float
    distinct_days: int
    detection_count: int
    frp_cv: float | None
    first_seen: datetime
    last_seen: datetime


@dataclass(frozen=True)
class PersistenceResult:
    """Итог анализа.

    `enough_history=False` означает «ответить нельзя», а НЕ «факелов нет».
    Различие обязано быть видимым: на нём стоит доверие ко всему слою (AC-03).
    """
    enough_history: bool
    history_days: float
    window_days: int
    sources: tuple[PersistentSource, ...]

    @property
    def status(self) -> str:
        return "ok" if self.enough_history else "insufficient_history"


def _meters_between(lat1, lon1, lat2, lon2) -> float:
    """Равнопромежуточное приближение: на масштабе сотен метров этого достаточно."""
    mid = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * EARTH_M_PER_DEG * math.cos(mid)
    dy = (lat2 - lat1) * EARTH_M_PER_DEG
    return math.hypot(dx, dy)


def cluster_detections(detections, radius_m: float) -> list[Cluster]:
    """Сгруппировать детекции по местоположению.

    Сеточный индекс со стороной radius: кандидаты ищутся только в 9 соседних
    ячейках, поэтому оценка остаётся линейной по числу детекций.
    """
    lat_step = radius_m / EARTH_M_PER_DEG
    grid: dict[tuple[int, int], list[Cluster]] = {}
    clusters: list[Cluster] = []

    def _lon_step(row: int) -> float:
        """Шаг по долготе для широтной полосы.

        Он обязан зависеть от НОМЕРА ПОЛОСЫ, а не от точной широты детекции:
        иначе две точки на одном меридиане получают чуть разный шаг, попадают
        в колонки, отличающиеся больше чем на единицу, и перестают видеть друг
        друга при проверке соседей. Сетка тогда перестаёт быть сеткой.
        """
        band_lat = row * lat_step
        return radius_m / (EARTH_M_PER_DEG * max(math.cos(math.radians(band_lat)), 1e-6))

    for d in sorted(detections, key=lambda x: x.acquired_at):
        row = int(d.latitude // lat_step)
        best, best_dist = None, radius_m
        # у каждой широтной полосы своя шкала долготы, поэтому колонка
        # пересчитывается для каждой просматриваемой полосы отдельно
        for dy in (-1, 0, 1):
            r = row + dy
            col = int(d.longitude // _lon_step(r))
            for dx in (-1, 0, 1):
                for c in grid.get((r, col + dx), ()):
                    dist = _meters_between(c.latitude, c.longitude, d.latitude, d.longitude)
                    if dist <= best_dist:
                        best, best_dist = c, dist
        if best is None:
            best = Cluster(d.latitude, d.longitude)
            clusters.append(best)
            grid.setdefault((row, int(d.longitude // _lon_step(row))), []).append(best)
        best.add(d)
    return clusters


def analyse(detections, config: PersistenceConfig | None = None,
            now: datetime | None = None) -> PersistenceResult:
    """Найти персистентные теплоисточники (AC-01, AC-03, AC-05)."""
    config = config or PersistenceConfig()
    if not detections:
        return PersistenceResult(False, 0.0, config.window_days, ())

    times = [d.acquired_at for d in detections]
    history_days = (max(times) - min(times)).total_seconds() / 86400
    if history_days < config.window_days:
        # истории не хватает — ответить нельзя, и это не «факелов нет»
        return PersistenceResult(False, history_days, config.window_days, ())

    cutoff = (now or max(times)) - timedelta(days=config.window_days)
    recent = [d for d in detections if d.acquired_at >= cutoff]

    sources = []
    for c in cluster_detections(recent, config.radius_m):
        if len(c.days) < config.min_distinct_days:
            continue
        cv = c.frp_cv
        if config.max_frp_cv is not None and cv is not None and cv > config.max_frp_cv:
            continue          # мощность скачет — похоже на пожар, а не на факел
        t = [d.acquired_at for d in c.detections]
        sources.append(PersistentSource(
            latitude=c.latitude, longitude=c.longitude,
            distinct_days=len(c.days), detection_count=len(c.detections),
            frp_cv=cv, first_seen=min(t), last_seen=max(t)))
    return PersistenceResult(True, history_days, config.window_days, tuple(sources))


def partition(detections, result: PersistenceResult,
              config: PersistenceConfig | None = None):
    """Разделить на основной слой и служебный слой теплоисточников (AC-04).

    Если истории не хватает, НИ ОДНА точка не отсеивается: фильтр, который не
    может ответить, не имеет права выбрасывать данные.
    """
    config = config or PersistenceConfig()
    if not result.enough_history or not result.sources:
        return list(detections), []
    normal, persistent = [], []
    for d in detections:
        near = any(_meters_between(s.latitude, s.longitude, d.latitude, d.longitude)
                   <= config.radius_m for s in result.sources)
        (persistent if near else normal).append(d)
    return normal, persistent


def backfill(region, sources, map_key, days: int, store=None, chunk: int = 5,
             verbose: bool = True):
    """Собрать историю глубже 5 суток, шагая параметром DATE у Area API.

    ТЗ и эта спека исходили из того, что окно в 14 суток требует архива FIRMS и
    NASA Earthdata Login. Это оказалось неверно: Area API принимает дату конца
    окна и отдаёт данные примерно за два месяца NRT-хранения, поэтому история
    собирается серией обычных запросов.
    """
    from datetime import date as _date
    from src import firms

    all_dets, failures = [], {}
    today = _date.today()
    for offset in range(0, days, chunk):
        when = (today - timedelta(days=offset)).isoformat()
        res = firms.fetch(region, sources, min(chunk, days - offset), map_key, date=when)
        all_dets.extend(res.detections)
        failures.update(res.failures)
        if verbose:
            print(f"  DATE={when} -> {len(res.detections):6d} детекций"
                  + (f"  СБОЙ: {list(res.failures)}" if res.failures else ""))
        if store is not None:
            start = datetime.combine(_date.fromisoformat(when), datetime.min.time(),
                                     tzinfo=times_tz()) - timedelta(days=chunk)
            end = start + timedelta(days=chunk)
            store.record_fetch(res, sources, start, end)
    return all_dets, failures


def times_tz():
    from datetime import timezone as _tz
    return _tz.utc


def main(argv=None) -> int:
    import argparse, os, sys
    from src import firms
    from src.confidence import load_thresholds, partition as confidence_partition

    ap = argparse.ArgumentParser(prog="python3 -m src.persistence")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyse", help="собрать историю и найти персистентные источники")
    p.add_argument("--config", required=True)
    p.add_argument("--days", type=int, default=25, help="глубина истории в сутках")
    p.add_argument("--top", type=int, default=15, help="сколько источников показать")
    args = ap.parse_args(argv)

    map_key = os.environ.get("FIRMS_MAP_KEY", "")
    if not map_key:
        print("FIRMS_MAP_KEY is not set in the environment", file=sys.stderr)
        return 2

    region, sources, _ = firms.load_region(args.config)
    cfg = load_persistence_config(args.config)
    print(f"регион: {region.name}  история: {args.days} сут  "
          f"окно: {cfg.window_days} сут  радиус: {cfg.radius_m:g} м  "
          f"мин. суток: {cfg.min_distinct_days}  max FRP CV: {cfg.max_frp_cv}")
    dets, failures = backfill(region, sources, map_key, args.days)
    print(f"собрано {len(dets)} детекций, сбойных источников: {len(failures)}")

    kept, _ = confidence_partition(dets, load_thresholds(args.config))
    result = analyse(kept, cfg)
    print(f"\nстатус: {result.status}  история: {result.history_days:.1f} сут "
          f"(нужно {result.window_days})")
    if not result.enough_history:
        print("истории недостаточно — фильтр не может ответить")
        return 0

    print(f"персистентных теплоисточников: {len(result.sources)}")
    for s in sorted(result.sources, key=lambda x: -x.detection_count)[:args.top]:
        cv = f"{s.frp_cv:.2f}" if s.frp_cv is not None else "  — "
        print(f"  {s.latitude:8.4f} {s.longitude:9.4f}  суток={s.distinct_days:3d} "
              f"детекций={s.detection_count:4d}  FRP CV={cv}  "
              f"{s.first_seen:%m-%d}..{s.last_seen:%m-%d}")
    normal, persistent = partition(kept, result, cfg)
    print(f"\nосновной слой: {len(normal)}, служебный: {len(persistent)} "
          f"({100*len(persistent)/len(kept):.2f}%)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
