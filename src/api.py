"""SPEC-8: REST API и веб-карта.

Тонкий слой поверх готовых конвейеров. Вся агрегация — на backend, фронтенд
только рисует ответы.

Три свойства, ради которых этот модуль существует именно в таком виде:

* **Голое число гектаров получить невозможно.** `masked_fraction`,
  `thresholds_version` и `status` — обязательные поля схемы ответа, а не
  опциональные. Площадь, посчитанная по области, наполовину закрытой облаком,
  внешне неотличима от честной, и возможность вернуть её без контекста — это
  способ, которым сервис начинает врать (REQ-002, REQ-004).
* **Площади помечены как непроверенные.** SPEC-7 закрыт статусом `partial`:
  перекрёстная проверка показала, что вычисленная площадь не выдерживает
  сопоставления с термоточками (receipt SPEC-7-LIVE-002, FAIL). Пока SPEC-9 не
  закрыта, каждый ответ с площадью несёт `validated: false` и ссылку на
  причину. Отдавать эти гектары молча было бы хуже, чем не отдавать вовсе.
* **`/health` различает «опрос прошёл» и «данные получены».** Это разные
  величины: опрос может идти успешно и возвращать пусто неделями, и их
  смешение скрывает деградацию источника.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import pathlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.burn import SEVERITY_ORDER, BurnResult, BurnStatus
from src.events import Event, EventStatus

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Пока SPEC-9 не закрыта, площади гари непроверены — см. SPEC-7-LIVE-002.
# SPEC-9 ввела перекрёстную проверку: флаг теперь берётся из результата, а не
# проставляется здесь. Непрошедший результат приходит со статусом `unvalidated`
# и причиной, и опубликовать его как проверенный нельзя.
UNVALIDATED_NOTE = (
    "Площадь не прошла перекрёстную проверку по термоточкам события (SPEC-9). "
    "Число посчитано, но пятно не совпадает с местом горения."
)


# --- схемы ответов: обязательность полей — часть контракта ---

class BurnAreas(BaseModel):
    unburnt: float
    low: float
    moderate_low: float
    moderate_high: float
    high: float


class BurnPayload(BaseModel):
    """Площадь гари. Ни одно из полей контекста не является опциональным."""
    event_id: str
    status: str
    area_ha: BurnAreas
    burned_ha: float
    masked_fraction: float = Field(..., ge=0.0, le=1.0)
    thresholds_version: str
    validated: bool
    validation_note: str | None = None
    doy_gap_days: int | None = None
    enrichment: float | None = None
    detections_checked: int | None = None
    mgrs_tile: str | None = None
    scene_before: str | None = None
    scene_after: str | None = None
    scene_before_date: str | None = None
    scene_after_date: str | None = None
    reason: str | None = None
    retry_after: str | None = None


class SourceHealth(BaseModel):
    """Две РАЗНЫЕ величины, не одна."""
    source: str
    last_successful_poll: str | None
    last_detection_received: str | None
    observations: int
    failures: int


class Health(BaseModel):
    region: str
    generated_at: str
    sources: list[SourceHealth]
    events_total: int
    events_deferred: int
    persistent_sources: int
    burn_results_validated: bool


@dataclass
class Snapshot:
    """Готовые данные для отдачи. Собирается вне HTTP-слоя — он остаётся тонким."""
    region_name: str
    bbox: tuple[float, float, float, float]
    events: list[Event] = field(default_factory=list)
    flares: list = field(default_factory=list)
    burns: dict[str, BurnResult] = field(default_factory=dict)
    store: object | None = None
    sources: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _feature(geometry: dict, props: dict) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": props}


def _fc(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def burn_payload(result: BurnResult) -> BurnPayload:
    return BurnPayload(
        event_id=result.event_id,
        status=result.status.value,
        area_ha=BurnAreas(**{k: result.area_ha.get(k, 0.0) for k in SEVERITY_ORDER}),
        burned_ha=result.burned_ha,
        masked_fraction=result.masked_fraction,
        thresholds_version=result.thresholds_version,
        validated=result.validated,
        validation_note=None if result.validated else (result.reason or UNVALIDATED_NOTE),
        doy_gap_days=result.doy_gap_days, enrichment=result.enrichment,
        detections_checked=result.detections_checked,
        mgrs_tile=result.mgrs_tile,
        scene_before=result.scene_before, scene_after=result.scene_after,
        scene_before_date=result.scene_before_date,
        scene_after_date=result.scene_after_date,
        reason=result.reason, retry_after=result.retry_after)


def create_app(snapshot: Snapshot) -> FastAPI:
    app = FastAPI(title="Оперативный мониторинг лесных пожаров",
                  version="0.1.0", description=__doc__)

    def _in_bbox(lon, lat, bbox):
        return bbox is None or (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3])

    def _parse_bbox(bbox: str | None):
        if not bbox:
            return None
        parts = [float(v) for v in bbox.split(",")]
        if len(parts) != 4:
            raise HTTPException(400, "bbox must be west,south,east,north")
        return parts

    @app.get("/api/v1/hotspots")
    def hotspots(bbox: str | None = Query(None), date_from: str | None = None,
                 date_to: str | None = None, min_confidence: str | None = None):
        box = _parse_bbox(bbox)
        lo = datetime.fromisoformat(date_from) if date_from else None
        hi = datetime.fromisoformat(date_to) if date_to else None
        feats = []
        for e in snapshot.events:
            lon, lat = e.centroid
            if not _in_bbox(lon, lat, box):
                continue
            if lo and e.last_seen < lo:
                continue
            if hi and e.first_seen > hi:
                continue
            feats.append(_feature(
                {"type": "Point", "coordinates": [lon, lat]},
                {"event_id": e.id, "status": e.status.value,
                 "detections": len(e.detections), "total_frp": round(e.total_frp, 1),
                 "max_frp": e.max_frp, "sensors": e.sensors,
                 "first_seen": e.first_seen.isoformat(),
                 "last_seen": e.last_seen.isoformat(),
                 "duration_days": round(e.duration_days, 2),
                 "bbox": list(e.bbox)}))
        return _fc(feats)

    @app.get("/api/v1/hotspots/{event_id}")
    def hotspot(event_id: str):
        e = next((x for x in snapshot.events if x.id == event_id), None)
        if e is None:
            raise HTTPException(404, f"event {event_id} not found")
        return {
            "event_id": e.id, "status": e.status.value, "bbox": list(e.bbox),
            "total_frp": round(e.total_frp, 1), "sensors": e.sensors,
            "first_seen": e.first_seen.isoformat(), "last_seen": e.last_seen.isoformat(),
            "history": _fc([
                _feature({"type": "Point", "coordinates": [d.longitude, d.latitude]},
                         {"acq_datetime": d.acquired_at.isoformat(), "sensor": d.sensor,
                          "frp": d.frp, "confidence": d.confidence,
                          "brightness": d.brightness, "daynight": d.daynight})
                for d in e.history]),
        }

    @app.get("/api/v1/burns")
    def burns(bbox: str | None = Query(None), severity_class: str | None = None):
        box = _parse_bbox(bbox)
        feats = []
        for e in snapshot.events:
            r = snapshot.burns.get(e.id)
            if r is None:
                continue
            lon, lat = e.centroid
            if not _in_bbox(lon, lat, box):
                continue
            p = burn_payload(r).model_dump()
            if severity_class:
                if severity_class not in SEVERITY_ORDER:
                    raise HTTPException(400, f"unknown severity_class {severity_class}")
                p["area_ha"] = {severity_class: p["area_ha"][severity_class]}
            w, s, ea, n = e.bbox
            feats.append(_feature(
                {"type": "Polygon", "coordinates": [[[w, s], [ea, s], [ea, n],
                                                     [w, n], [w, s]]]}, p))
        return _fc(feats)

    @app.get("/api/v1/burns/{event_id}", response_model=BurnPayload)
    def burn(event_id: str):
        r = snapshot.burns.get(event_id)
        if r is None:
            raise HTTPException(404, f"no burn result for event {event_id}")
        return burn_payload(r)

    @app.get("/api/v1/stats/area")
    def stats_area():
        totals = {k: 0.0 for k in SEVERITY_ORDER}
        counted = deferred = unvalidated = 0
        for r in snapshot.burns.values():
            if r.status is BurnStatus.UNVALIDATED:
                unvalidated += 1
                continue
            if r.status is not BurnStatus.OK:
                deferred += 1
                continue
            counted += 1
            for k in SEVERITY_ORDER:
                totals[k] += r.area_ha.get(k, 0.0)
        return {
            "region": snapshot.region_name,
            "area_ha": {k: round(v, 2) for k, v in totals.items()},
            "burned_ha": round(sum(v for k, v in totals.items() if k != "unburnt"), 2),
            "events_counted": counted, "events_without_result": deferred,
            "validated": counted > 0 and unvalidated == 0,
            "events_unvalidated": unvalidated,
            "validation_note": None if unvalidated == 0 else UNVALIDATED_NOTE,
        }

    @app.get("/api/v1/flares")
    def flares():
        """Служебный слой: без него фильтр SPEC-4 неаудируем."""
        return _fc([
            _feature({"type": "Point", "coordinates": [s.longitude, s.latitude]},
                     {"distinct_days": s.distinct_days, "detections": s.detection_count,
                      "frp_cv": s.frp_cv, "first_seen": s.first_seen.isoformat(),
                      "last_seen": s.last_seen.isoformat()})
            for s in snapshot.flares])

    @app.get("/api/v1/health", response_model=Health)
    def health():
        rows = []
        for src in snapshot.sources:
            last_poll = last_det = None
            obs = fails = 0
            if snapshot.store is not None:
                cur = snapshot.store.conn.execute(
                    "SELECT status, MAX(requested_at) FROM observations"
                    " WHERE source = ? GROUP BY status", (src,))
                for status, ts in cur.fetchall():
                    if status == "ok":
                        last_poll = ts
                    else:
                        fails += 1
                obs = snapshot.store.conn.execute(
                    "SELECT COUNT(*) FROM observations WHERE source=? AND status='ok'",
                    (src,)).fetchone()[0]
                fails = snapshot.store.conn.execute(
                    "SELECT COUNT(*) FROM observations WHERE source=? AND status='failed'",
                    (src,)).fetchone()[0]
                row = snapshot.store.conn.execute(
                    "SELECT MAX(acquired_at) FROM detections WHERE sensor=?",
                    (src,)).fetchone()
                last_det = row[0] if row else None
            rows.append(SourceHealth(
                source=src, last_successful_poll=last_poll,
                last_detection_received=last_det, observations=obs, failures=fails))
        return Health(
            region=snapshot.region_name,
            generated_at=snapshot.generated_at.isoformat(),
            sources=rows, events_total=len(snapshot.events),
            events_deferred=sum(1 for r in snapshot.burns.values()
                                if r.status is BurnStatus.DEFERRED),
            persistent_sources=len(snapshot.flares),
            # Непроверенной является ОПУБЛИКОВАННАЯ площадь, не прошедшая
            # перекрёстную проверку. Событие в статусе deferred площади не даёт
            # вовсе, и считать его непроверенным — значит поднимать тревогу
            # там, где сервис как раз повёл себя честно.
            burn_results_validated=not any(
                r.status is BurnStatus.UNVALIDATED for r in snapshot.burns.values()))

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


def build_snapshot(config_path: str, days: int = 25, burn_limit: int = 0,
                   db: str = ":memory:") -> Snapshot:
    """Собрать данные из всех конвейеров этапа 1 и (опционально) этапа 2."""
    import os
    from src import firms
    from src.burn import load_burn_config, run_for_event
    from src.confidence import load_thresholds, partition as cpart
    from src.events import build_events, evaluate_completion, load_event_config
    from src.persistence import (analyse, backfill, load_persistence_config,
                                 partition as ppart)
    from src.store import Store

    region, sources, _ = firms.load_region(config_path)
    store = Store(db)
    dets, _ = backfill(region, sources, os.environ["FIRMS_MAP_KEY"], days,
                       store=store, verbose=False)
    kept, _ = cpart(dets, load_thresholds(config_path))
    pcfg = load_persistence_config(config_path)
    result = analyse(kept, pcfg)
    kept, _ = ppart(kept, result, pcfg)

    ecfg = load_event_config(config_path)
    events = build_events(kept, ecfg)
    if events:
        evaluate_completion(events, store, sources, ecfg,
                            now=max(d.acquired_at for d in kept))

    burns: dict = {}
    if burn_limit:
        from datetime import timedelta
        from src.burn import aoi_extent_km
        bcfg = load_burn_config(config_path)
        now = max((d.acquired_at for d in kept), default=None)

        def mappable(e) -> bool:
            """Есть ли вообще шанс посчитать гарь для этого события.

            Отбор по одному только FRP тратил бюджет расчётов на крупнейшие
            пожары, которые либо шире max_aoi_km, либо ещё не отпустили окно
            поиска сцены «после». Пять расчётов подряд возвращали deferred и
            failed, и карта оставалась без единой площади.
            """
            if now is not None and (now - e.last_seen) < timedelta(
                    days=bcfg.post_window_days[0] + 30):
                return False
            return max(aoi_extent_km(e.padded_bbox(bcfg.aoi_pad_m))) <= bcfg.max_aoi_km

        candidates = [e for e in events if mappable(e)] or events
        for e in sorted(candidates, key=lambda x: -x.total_frp)[:burn_limit]:
            try:
                burns[e.id] = run_for_event(e, bcfg)
            except Exception as exc:                # сцена недоступна — не падаем
                print(f"  {e.id}: {type(exc).__name__}: {exc}")
    return Snapshot(region_name=region.name, bbox=region.bbox, events=events,
                    flares=list(result.sources), burns=burns, store=store,
                    sources=sources)


def main(argv=None) -> int:
    import argparse, os, sys
    import uvicorn

    ap = argparse.ArgumentParser(prog="python3 -m src.api")
    ap.add_argument("--config", required=True)
    ap.add_argument("--days", type=int, default=25)
    ap.add_argument("--burns", type=int, default=0,
                    help="для скольких крупнейших событий считать гарь")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cache", help="каталог кеша снимка (SPEC-14)")
    ap.add_argument("--rebuild", action="store_true",
                    help="пересобрать кеш, даже если он есть")
    args = ap.parse_args(argv)

    from time import perf_counter
    from src import cache

    snap = None
    root = pathlib.Path(args.cache) if args.cache else None
    if root and not args.rebuild and (root / "snapshot.json").exists():
        t0 = perf_counter()
        snap = cache.load(root, args.config)
        print(f"снимок загружен из кеша за {perf_counter() - t0:.1f} с "
              f"(собран {snap.generated_at:%Y-%m-%d %H:%M} UTC, сеть не использовалась)")

    if snap is None:
        if not os.environ.get("FIRMS_MAP_KEY"):
            print("FIRMS_MAP_KEY is not set in the environment", file=sys.stderr)
            return 2
        print(f"сборка данных за {args.days} суток…")
        t0 = perf_counter()
        snap = build_snapshot(args.config, args.days, args.burns)
        print(f"собрано за {perf_counter() - t0:.0f} с")
        if root:
            stats = cache.save(snap, root, args.config)
            print(f"кеш записан в {root}: детекций {stats['detections']}, "
                  f"{stats['total_bytes']/1e6:.1f} МБ всего "
                  f"(parquet {stats['detections_bytes']/1e6:.1f} + "
                  f"{stats['observations_bytes']/1e3:.0f} КБ, "
                  f"метаданные {stats['snapshot_bytes']/1e6:.2f} МБ)")
    print(f"событий {len(snap.events)}, теплоисточников {len(snap.flares)}, "
          f"расчётов гари {len(snap.burns)}")
    print(f"карта: http://127.0.0.1:{args.port}/")
    uvicorn.run(create_app(snap), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
