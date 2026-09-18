"""SPEC-14: локальный кеш снимка данных.

Сборка за 60 суток занимает около 13 минут — почти целиком сеть. Кеш позволяет
поднять сервер за секунды и без ключа FIRMS.

**Формат — Parquet со сжатием zstd, а не SQLite и не JSON.** Детекции
колоночные и крайне повторяемые: имя сенсора, спутник, день/ночь и класс
достоверности принимают единицы значений на сотни тысяч строк, а координаты и
временные отметки хорошо сжимаются дельтами. SQLite хранила 933 тысячи детекций в
268 МБ, а JSON с ключами событий — ещё в 53 МБ, и такой кеш в репозиторий не
кладётся. Принадлежность события хранится колонкой `event_id`, а не списком
строковых ключей: именно эти ключи и раздували JSON.

**Кеш несёт отпечаток конфигурации и отказывается загружаться при
несовпадении.** Снимок, собранный с одними порогами и прочитанный при других,
дал бы события, построенные по одной конфигурации, и площади — по другой.
Молчаливая подстройка здесь хуже отказа: расхождение было бы невидимым.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 2
COMPRESSION = "zstd"

DETECTION_FIELDS = ("sensor", "latitude", "longitude", "acquired_at", "brightness",
                    "frp", "confidence", "satellite", "daynight")
OBSERVATION_FIELDS = ("source", "requested_at", "window_start", "window_end",
                      "status", "detection_count", "error")


class CacheError(RuntimeError):
    """Кеш непригоден. Тихий откат к сети здесь запрещён: он превратил бы
    повреждённый кеш в тринадцатиминутную паузу без объяснения."""


def config_fingerprint(config_path: str | Path) -> str:
    """Хеш конфигурации: пороги влияют на каждое производное значение."""
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]


def _paths(root: str | Path):
    root = Path(root)
    return (root, root / "snapshot.json", root / "detections.parquet",
            root / "observations.parquet")


def save(snapshot, root: str | Path, config_path: str | Path) -> dict:
    """Записать кеш. Возвращает измеренные размеры."""
    root, meta_p, det_p, obs_p = _paths(root)
    root.mkdir(parents=True, exist_ok=True)

    # event_id хранится колонкой; детекции, не попавшие ни в одно событие
    # (отсеянные SPEC-3 и SPEC-4), сохраняются с пустым event_id — без них
    # /health показал бы неверное время последней детекции.
    owner: dict[int, str] = {}
    for e in snapshot.events:
        for d in e.detections:
            owner[id(d)] = e.id

    rows = {f: [] for f in DETECTION_FIELDS}
    rows["event_id"] = []
    seen: set[tuple] = set()

    def add(d, event_id):
        key = (d.sensor, round(d.latitude, 5), round(d.longitude, 5), d.acquired_at)
        if key in seen:
            return
        seen.add(key)
        rows["sensor"].append(d.sensor)
        rows["latitude"].append(d.latitude)
        rows["longitude"].append(d.longitude)
        rows["acquired_at"].append(d.acquired_at)
        rows["brightness"].append(d.brightness)
        rows["frp"].append(d.frp)
        rows["confidence"].append(d.confidence)
        rows["satellite"].append(d.satellite)
        rows["daynight"].append(d.daynight)
        rows["event_id"].append(event_id)

    for e in snapshot.events:
        for d in e.detections:
            add(d, e.id)
    if snapshot.store is not None:
        from src import firms
        for r in snapshot.store.detections_between(
                datetime(2000, 1, 1, tzinfo=timezone.utc),
                datetime(2100, 1, 1, tzinfo=timezone.utc)):
            add(firms.Detection(
                latitude=r["latitude"], longitude=r["longitude"],
                brightness=r["brightness"], frp=r["frp"], confidence=r["confidence"],
                acquired_at=datetime.fromisoformat(r["acquired_at"]),
                sensor=r["sensor"], satellite=r["satellite"],
                daynight=r["daynight"]), None)

    pq.write_table(pa.table(rows), det_p, compression=COMPRESSION)

    obs = {f: [] for f in OBSERVATION_FIELDS}
    if snapshot.store is not None:
        for r in snapshot.store.conn.execute(
                f"SELECT {', '.join(OBSERVATION_FIELDS)} FROM observations"):
            for f, v in zip(OBSERVATION_FIELDS, r):
                obs[f].append(v)
    pq.write_table(pa.table(obs), obs_p, compression=COMPRESSION)

    meta_p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(config_path),
        "built_at": snapshot.generated_at.isoformat(),
        "region_name": snapshot.region_name,
        "bbox": list(snapshot.bbox),
        "sources": list(snapshot.sources),
        "event_status": {e.id: e.status.value for e in snapshot.events},
        "flares": [{**asdict(s), "first_seen": s.first_seen.isoformat(),
                    "last_seen": s.last_seen.isoformat()} for s in snapshot.flares],
        "burns": {eid: {**asdict(r), "status": r.status.value}
                  for eid, r in snapshot.burns.items()},
    }, ensure_ascii=False), encoding="utf-8")

    return {"detections": len(rows["sensor"]), "observations": len(obs["source"]),
            "events": len(snapshot.events),
            "detections_bytes": det_p.stat().st_size,
            "observations_bytes": obs_p.stat().st_size,
            "snapshot_bytes": meta_p.stat().st_size,
            "total_bytes": sum(p.stat().st_size for p in (det_p, obs_p, meta_p))}


def load(root: str | Path, config_path: str | Path):
    """Восстановить снимок из кеша. Сеть не используется."""
    from src import firms
    from src.api import Snapshot
    from src.burn import BurnResult, BurnStatus
    from src.events import Event, EventStatus
    from src.persistence import PersistentSource
    from src.store import Store

    root, meta_p, det_p, obs_p = _paths(root)
    if not meta_p.exists():
        raise CacheError(f"кеш {meta_p} не найден")
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CacheError(f"кеш {meta_p} повреждён: {exc}") from None

    if meta.get("schema_version") != SCHEMA_VERSION:
        raise CacheError(
            f"кеш {meta_p} записан схемой версии {meta.get('schema_version')}, "
            f"ожидается {SCHEMA_VERSION}; пересоберите кеш")
    actual = config_fingerprint(config_path)
    if meta.get("config_fingerprint") != actual:
        raise CacheError(
            f"кеш {meta_p} собран при другой конфигурации "
            f"({meta.get('config_fingerprint')} против {actual}). Пороги влияют "
            f"на события и на площади; пересоберите кеш")
    for p in (det_p, obs_p):
        if not p.exists():
            raise CacheError(f"файл кеша {p} отсутствует")

    try:
        det = pq.read_table(det_p).to_pydict()
        obs = pq.read_table(obs_p).to_pydict()
    except Exception as exc:
        raise CacheError(f"файлы кеша нечитаемы: {type(exc).__name__}: {exc}") from None

    store = Store(":memory:")
    detections, grouped = [], {}
    for i in range(len(det["sensor"])):
        when = det["acquired_at"][i]
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        d = firms.Detection(
            latitude=det["latitude"][i], longitude=det["longitude"][i],
            brightness=det["brightness"][i], frp=det["frp"][i],
            confidence=det["confidence"][i], acquired_at=when,
            sensor=det["sensor"][i], satellite=det["satellite"][i],
            daynight=det["daynight"][i])
        detections.append(d)
        if det["event_id"][i]:
            grouped.setdefault(det["event_id"][i], []).append(d)
    store.add_detections(detections)

    if obs["source"]:
        store.conn.executemany(
            f"INSERT INTO observations ({', '.join(OBSERVATION_FIELDS)})"
            f" VALUES ({','.join('?' * len(OBSERVATION_FIELDS))})",
            list(zip(*(obs[f] for f in OBSERVATION_FIELDS))))
        store.conn.commit()

    statuses = meta["event_status"]
    events = []
    for eid, dets in grouped.items():
        e = Event(detections=dets, status=EventStatus(statuses.get(eid, "active")))
        if e.id != eid:
            raise CacheError(
                f"событие {eid} не воспроизводится из кеша (получилось {e.id}): "
                f"кеш и код разошлись в правиле построения идентификатора")
        events.append(e)

    flares = [PersistentSource(
        latitude=s["latitude"], longitude=s["longitude"],
        distinct_days=s["distinct_days"], detection_count=s["detection_count"],
        frp_cv=s["frp_cv"], first_seen=datetime.fromisoformat(s["first_seen"]),
        last_seen=datetime.fromisoformat(s["last_seen"])) for s in meta["flares"]]

    burns = {}
    for eid, r in meta["burns"].items():
        r = dict(r)
        burns[eid] = BurnResult(status=BurnStatus(r.pop("status")), **r)

    return Snapshot(
        region_name=meta["region_name"], bbox=tuple(meta["bbox"]), events=events,
        flares=flares, burns=burns, store=store, sources=meta["sources"],
        generated_at=datetime.fromisoformat(meta["built_at"]))
