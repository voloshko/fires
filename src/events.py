"""SPEC-6: кластеризация детекций в события пожаров.

Пожар наблюдается как облако отдельных пикселей от разных сенсоров и пролётов.
Пользователю нужен один очаг с историей, а этапу 2 — устойчивый идентификатор и
границы области, по которым искать сцены Sentinel-2.

Ключевое решение — **завершение события**. Наивный критерий «нет детекций
7 суток» неверен: пропуск опроса, сбой источника или сплошная облачность
потушили бы пожар в базе. Событие закрывается только при наличии
ПОДТВЕРЖДЁННЫХ наблюдений без детекций (ФТ-1.9), а журнал наблюдений из SPEC-2
существует ровно ради этого решения.

Идентификатор выводится из самой ранней детекции события и не меняется при
поступлении новых. Известный потолок: если задним числом подгрузить детекцию
РАНЬШЕ текущей первой, идентификатор события сменится. На штатном потоке (данные
приходят вперёд по времени) это не происходит; при массовом backfill возможно.
"""
from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from src.persistence import EARTH_M_PER_DEG, meters_between


class EventStatus(Enum):
    ACTIVE = "active"        # детекции продолжают поступать
    COMPLETE = "complete"    # тишина ПОДТВЕРЖДЕНА наблюдениями
    UNCONFIRMED = "unconfirmed"   # тишина есть, но наблюдений не было — молчать нельзя


@dataclass(frozen=True)
class EventConfig:
    radius_m: float = 3000.0
    time_gap_hours: float = 48.0
    completion_days: int = 7
    min_observations: int = 1


def load_event_config(config_path: str | Path) -> EventConfig:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8")).get("events", {})
    return EventConfig(
        radius_m=float(cfg.get("radius_m", 3000.0)),
        time_gap_hours=float(cfg.get("time_gap_hours", 48.0)),
        completion_days=int(cfg.get("completion_days", 7)),
        min_observations=int(cfg.get("min_observations", 1)),
    )


@dataclass
class Event:
    detections: list = field(default_factory=list)
    status: EventStatus = EventStatus.ACTIVE

    @property
    def history(self) -> list:
        """Вся история события в хронологическом порядке (AC-05)."""
        return sorted(self.detections, key=lambda d: d.acquired_at)

    @property
    def first_seen(self) -> datetime:
        return min(d.acquired_at for d in self.detections)

    @property
    def last_seen(self) -> datetime:
        return max(d.acquired_at for d in self.detections)

    @property
    def id(self) -> str:
        """Устойчивый идентификатор из самой ранней детекции (AC-02)."""
        d = self.history[0]
        raw = f"{d.latitude:.4f}|{d.longitude:.4f}|{d.acquired_at.isoformat()}"
        return "FIRE-" + hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """west, south, east, north — вход для поиска сцен в SPEC-7 (AC-04)."""
        lons = [d.longitude for d in self.detections]
        lats = [d.latitude for d in self.detections]
        return (min(lons), min(lats), max(lons), max(lats))

    @property
    def centroid(self) -> tuple[float, float]:
        n = len(self.detections)
        return (sum(d.longitude for d in self.detections) / n,
                sum(d.latitude for d in self.detections) / n)

    @property
    def total_frp(self) -> float:
        """Суммарная мощность излучения, МВт (AC-04)."""
        return sum(d.frp for d in self.detections if d.frp is not None)

    @property
    def max_frp(self) -> float | None:
        v = [d.frp for d in self.detections if d.frp is not None]
        return max(v) if v else None

    @property
    def sensors(self) -> list[str]:
        return sorted({d.sensor for d in self.detections})

    @property
    def duration_days(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds() / 86400

    def padded_bbox(self, pad_m: float) -> tuple[float, float, float, float]:
        """bbox с запасом в метрах: периметр гари шире облака термоточек."""
        w, s, e, n = self.bbox
        dlat = pad_m / EARTH_M_PER_DEG
        mid = math.radians((s + n) / 2)
        dlon = pad_m / (EARTH_M_PER_DEG * max(math.cos(mid), 1e-6))
        return (w - dlon, s - dlat, e + dlon, n + dlat)

    def _touches(self, d, config: EventConfig) -> bool:
        """Детекция примыкает к событию по месту И по времени."""
        gap = timedelta(hours=config.time_gap_hours)
        return any(
            abs((d.acquired_at - x.acquired_at)) <= gap
            and meters_between(x.latitude, x.longitude, d.latitude, d.longitude)
            <= config.radius_m
            for x in self.detections
        )


def build_events(detections, config: EventConfig | None = None) -> list[Event]:
    """Сгруппировать детекции в события по пространственно-временной близости (AC-01).

    Одно событие может поглотить другое: фронт, разошедшийся и снова сомкнувшийся,
    остаётся одним пожаром.
    """
    config = config or EventConfig()
    lat_step = config.radius_m / EARTH_M_PER_DEG
    grid: dict[tuple[int, int], set[int]] = {}
    events: list[Event | None] = []

    def _lon_step(row: int) -> float:
        # см. SPEC-4: шаг зависит от полосы, а не от точной широты точки
        return config.radius_m / (
            EARTH_M_PER_DEG * max(math.cos(math.radians(row * lat_step)), 1e-6))

    def _register(idx: int, d) -> None:
        row = int(d.latitude // lat_step)
        grid.setdefault((row, int(d.longitude // _lon_step(row))), set()).add(idx)

    for d in sorted(detections, key=lambda x: x.acquired_at):
        row = int(d.latitude // lat_step)
        candidates: set[int] = set()
        for dy in (-1, 0, 1):
            r = row + dy
            col = int(d.longitude // _lon_step(r))
            for dx in (-1, 0, 1):
                candidates |= grid.get((r, col + dx), set())

        matched = [i for i in sorted(candidates)
                   if events[i] is not None and events[i]._touches(d, config)]
        if not matched:
            events.append(Event(detections=[d]))
            _register(len(events) - 1, d)
            continue

        target = matched[0]
        for other in matched[1:]:              # детекция сомкнула два события
            events[target].detections.extend(events[other].detections)
            for x in events[other].detections:
                _register(target, x)
            events[other] = None
        events[target].detections.append(d)
        _register(target, d)

    return [e for e in events if e is not None]


def evaluate_completion(events, store, sources, config: EventConfig | None = None,
                        now: datetime | None = None) -> None:
    """Проставить статус завершения (AC-03).

    Три исхода, а не два. `UNCONFIRMED` — детекций нет, но и наблюдений не было:
    сказать, что пожар потух, нельзя. Схлопывание его в COMPLETE — это ровно тот
    способ, которым сервис объявил бы потухшим пожар, скрытый облачностью.
    """
    config = config or EventConfig()
    now = now or datetime.now(tz=events[0].last_seen.tzinfo) if events else now
    for e in events:
        silence = now - e.last_seen
        if silence < timedelta(days=config.completion_days):
            e.status = EventStatus.ACTIVE
            continue
        observations = sum(
            store.coverage(s, e.last_seen, now).observations for s in sources)
        e.status = (EventStatus.COMPLETE
                    if observations >= config.min_observations
                    else EventStatus.UNCONFIRMED)


def main(argv=None) -> int:
    import argparse, os, sys
    from datetime import timezone
    from src import firms
    from src.confidence import load_thresholds, partition as confidence_partition
    from src.persistence import (analyse, backfill, load_persistence_config,
                                 partition as persistence_partition)
    from src.store import Store

    ap = argparse.ArgumentParser(prog="python3 -m src.events")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="собрать историю и построить события")
    p.add_argument("--config", required=True)
    p.add_argument("--days", type=int, default=25)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--db", default=":memory:")
    args = ap.parse_args(argv)

    map_key = os.environ.get("FIRMS_MAP_KEY", "")
    if not map_key:
        print("FIRMS_MAP_KEY is not set in the environment", file=sys.stderr)
        return 2

    region, sources, _ = firms.load_region(args.config)
    cfg = load_event_config(args.config)
    print(f"регион: {region.name}  история: {args.days} сут  "
          f"радиус: {cfg.radius_m:g} м  разрыв: {cfg.time_gap_hours:g} ч  "
          f"завершение: {cfg.completion_days} сут")

    store = Store(args.db)
    dets, _ = backfill(region, sources, map_key, args.days, store=store, verbose=False)
    kept, _ = confidence_partition(dets, load_thresholds(args.config))
    pcfg = load_persistence_config(args.config)
    kept, flares = persistence_partition(kept, analyse(kept, pcfg), pcfg)
    print(f"детекций: получено {len(dets)}, после SPEC-3/SPEC-4 осталось {len(kept)} "
          f"(в служебный слой {len(flares)})")

    events = build_events(kept, cfg)
    now = max(d.acquired_at for d in kept)
    evaluate_completion(events, store, sources, cfg, now=now)

    by_status: dict[str, int] = {}
    for e in events:
        by_status[e.status.value] = by_status.get(e.status.value, 0) + 1
    print(f"\nсобытий: {len(events)}")
    for k, v in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v}")

    print(f"\nкрупнейшие по суммарному FRP:")
    for e in sorted(events, key=lambda x: -x.total_frp)[:args.top]:
        lon, lat = e.centroid
        w, s, ea, n = e.bbox
        span_km = max((n - s) * 111.32, (ea - w) * 111.32 * 0.5)
        print(f"  {e.id}  {lat:7.3f} {lon:8.3f}  детекций={len(e.detections):5d} "
              f"FRP={e.total_frp:9.1f} МВт  {e.duration_days:5.1f} сут  "
              f"~{span_km:5.1f} км  {e.status.value}")
    store.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
