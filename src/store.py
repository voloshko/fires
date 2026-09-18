"""SPEC-2: накопительное хранилище детекций и журнал наблюдений.

Хранилище обязательно с первого дня: FIRMS отдаёт максимум 5 суток за запрос
(SPEC-1), а фильтр персистентности требует окна ≥14 суток (SPEC-4). Без
накопления этот фильтр нереализуем в принципе.

Два инварианта, которые здесь не «соблюдаются кодом», а навязаны схемой:

* **append-only** — UPDATE и DELETE запрещены триггерами SQLite. Тихая правка
  истории сделала бы фильтр персистентности недоказуемым;
* **журнал наблюдений** — запись о каждом опросе, включая вернувшие ноль
  детекций. Без него «детекций нет» неотличимо от «опроса не было», и событие
  можно ошибочно объявить потухшим (ФТ-1.9, SPEC-6).

Движок — sqlite3 из stdlib. Переход на PostGIS предусмотрен ТЗ, но до появления
нагрузки это лишняя зависимость; вся работа идёт через SQL, поэтому миграция —
смена драйвера, а не переписывание логики.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.firms import Detection, FetchResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY,
    sensor        TEXT    NOT NULL,
    latitude      REAL    NOT NULL,
    longitude     REAL    NOT NULL,
    acquired_at   TEXT    NOT NULL,   -- ISO-8601 UTC
    brightness    REAL,
    frp           REAL,
    confidence    TEXT    NOT NULL,   -- сырое значение сенсора; шкалы в SPEC-3
    satellite     TEXT,
    daynight      TEXT,
    ingested_at   TEXT    NOT NULL,
    UNIQUE (sensor, latitude, longitude, acquired_at)
);

CREATE INDEX IF NOT EXISTS detections_acquired_at
    ON detections (acquired_at);
CREATE INDEX IF NOT EXISTS detections_sensor_acquired_at
    ON detections (sensor, acquired_at);

-- Журнал наблюдений. Строка появляется на КАЖДЫЙ опрос, в том числе
-- вернувший ноль детекций; отсутствие строки означает, что опроса не было.
CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY,
    source          TEXT    NOT NULL,
    requested_at    TEXT    NOT NULL,
    window_start    TEXT    NOT NULL,
    window_end      TEXT    NOT NULL,
    status          TEXT    NOT NULL CHECK (status IN ('ok', 'failed')),
    detection_count INTEGER NOT NULL,
    error           TEXT
);

CREATE INDEX IF NOT EXISTS observations_source_window
    ON observations (source, window_start, window_end);

-- Append-only: правка и удаление истории запрещены на уровне схемы.
CREATE TRIGGER IF NOT EXISTS detections_no_update
BEFORE UPDATE ON detections
BEGIN SELECT RAISE(ABORT, 'detections is append-only: UPDATE is forbidden'); END;

CREATE TRIGGER IF NOT EXISTS detections_no_delete
BEFORE DELETE ON detections
BEGIN SELECT RAISE(ABORT, 'detections is append-only: DELETE is forbidden'); END;

CREATE TRIGGER IF NOT EXISTS observations_no_update
BEFORE UPDATE ON observations
BEGIN SELECT RAISE(ABORT, 'observations is append-only: UPDATE is forbidden'); END;

CREATE TRIGGER IF NOT EXISTS observations_no_delete
BEFORE DELETE ON observations
BEGIN SELECT RAISE(ABORT, 'observations is append-only: DELETE is forbidden'); END;
"""


@dataclass(frozen=True)
class Coverage:
    """Ответ на вопрос «были ли наблюдения в окне» (AC-04).

    Три исхода спеки различимы явно, а не по значению счётчика:

    * `observed=False`                    — опросов не было;
    * `observed=True,  detections=0`      — опросы были, огня не видели;
    * `observed=True,  detections>0`      — опросы были, есть детекции.

    Схлопывать первые два в «нет данных» нельзя: на этом различии стоит
    решение о завершении события (ФТ-1.9).

    `detections` — число РАЗЛИЧНЫХ детекций в окне, взятое из таблицы
    `detections`, а не сумма счётчиков журнала. Сумма по журналу удваивалась бы
    при перекрывающихся опросах одного окна, а перекрытие — штатный режим
    SPEC-1 (`day_range=2` при опросе раз в 3 часа).
    """
    observed: bool
    observations: int
    detections: int
    failures: int


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime is not accepted: {dt!r}")
    return dt.astimezone(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        # check_same_thread=False: FastAPI выполняет синхронные обработчики в
        # пуле потоков, и /health читает журнал наблюдений из рабочего потока.
        # Без этого сервер падал бы с ProgrammingError (SPEC-8, DEF-01).
        # Потолок: безопасно, пока запись идёт из одного потока до начала
        # обслуживания запросов, а обработчики только читают. Появится
        # конкурентная запись — нужен пул соединений или переход на PostGIS.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Добавить колонки, появившиеся позже схемы базы.

        CREATE TABLE IF NOT EXISTS не трогает существующую таблицу, поэтому
        база, созданная до SPEC-10, осталась бы без version и fire_type.
        Триггеры append-only запрещают UPDATE и DELETE, но не ALTER.
        """
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(detections)")}
        for column, decl in (("version", "TEXT"), ("fire_type", "INTEGER")):
            if column not in have:
                self.conn.execute(f"ALTER TABLE detections ADD COLUMN {column} {decl}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # --- запись ---

    def add_detections(self, detections: list[Detection]) -> int:
        """Вставить детекции, вернуть число действительно новых (AC-01, AC-02).

        Повторная загрузка того же окна ничего не добавляет: UNIQUE по
        (сенсор, широта, долгота, время) + INSERT OR IGNORE.
        """
        now = _iso(datetime.now(timezone.utc))
        rows = [(d.sensor, d.latitude, d.longitude, _iso(d.acquired_at),
                 d.brightness, d.frp, d.confidence, d.satellite, d.daynight,
                 d.version, d.fire_type, now)
                for d in detections]
        before = self.count_detections()
        with self.conn:
            self.conn.executemany(
                "INSERT OR IGNORE INTO detections (sensor, latitude, longitude,"
                " acquired_at, brightness, frp, confidence, satellite, daynight,"
                " version, fire_type, ingested_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return self.count_detections() - before

    def record_observation(self, source: str, window_start: datetime,
                           window_end: datetime, *, status: str,
                           detection_count: int, error: str | None = None,
                           requested_at: datetime | None = None) -> None:
        """Зафиксировать факт опроса — даже если он вернул ноль (AC-03)."""
        if status not in ("ok", "failed"):
            raise ValueError(f"status must be 'ok' or 'failed', got {status!r}")
        requested_at = requested_at or datetime.now(timezone.utc)
        with self.conn:
            self.conn.execute(
                "INSERT INTO observations (source, requested_at, window_start,"
                " window_end, status, detection_count, error)"
                " VALUES (?,?,?,?,?,?,?)",
                (source, _iso(requested_at), _iso(window_start),
                 _iso(window_end), status, detection_count, error))

    def record_fetch(self, result: FetchResult, sources: list[str],
                     window_start: datetime, window_end: datetime,
                     requested_at: datetime | None = None) -> int:
        """Записать итог опроса SPEC-1: детекции + журнал по каждому источнику.

        Сбойный источник получает строку со `status='failed'` — иначе его
        молчание было бы неотличимо от «огня нет».
        """
        new = self.add_detections(result.detections)
        per_source: dict[str, int] = {s: 0 for s in sources}
        for d in result.detections:
            per_source[d.sensor] = per_source.get(d.sensor, 0) + 1
        for source in sources:
            failed = source in result.failures
            self.record_observation(
                source, window_start, window_end,
                status="failed" if failed else "ok",
                detection_count=0 if failed else per_source.get(source, 0),
                error=result.failures.get(source),
                requested_at=requested_at)
        return new

    # --- чтение ---

    def count_detections(self) -> int:
        with closing(self.conn.execute("SELECT COUNT(*) FROM detections")) as cur:
            return cur.fetchone()[0]

    def coverage(self, source: str, start: datetime, end: datetime) -> Coverage:
        """Были ли наблюдения источника `source` в окне [start, end] (AC-04).

        Наблюдение засчитывается, если его окно пересекается с запрошенным.
        """
        with closing(self.conn.execute(
            "SELECT status, detection_count FROM observations"
            " WHERE source = ? AND window_start <= ? AND window_end >= ?",
            (source, _iso(end), _iso(start)),
        )) as cur:
            rows = cur.fetchall()
        ok = [r for r in rows if r["status"] == "ok"]
        with closing(self.conn.execute(
            "SELECT COUNT(*) FROM detections"
            " WHERE sensor = ? AND acquired_at >= ? AND acquired_at <= ?",
            (source, _iso(start), _iso(end)),
        )) as cur:
            distinct = cur.fetchone()[0]
        return Coverage(
            observed=bool(ok),
            observations=len(ok),
            detections=distinct,
            failures=len(rows) - len(ok),
        )

    def detections_between(self, start: datetime, end: datetime,
                           source: str | None = None) -> list[sqlite3.Row]:
        """История детекций за окно — вход для SPEC-4 и SPEC-6."""
        sql = ("SELECT * FROM detections WHERE acquired_at >= ? AND acquired_at <= ?")
        params: list = [_iso(start), _iso(end)]
        if source:
            sql += " AND sensor = ?"
            params.append(source)
        sql += " ORDER BY acquired_at"
        with closing(self.conn.execute(sql, params)) as cur:
            return cur.fetchall()
