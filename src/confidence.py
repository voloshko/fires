"""SPEC-3: нормализация confidence и пороговый фильтр.

ТЗ требовало «исключать confidence=low», не заметив, что шкалы у сенсоров
несовместимы: VIIRS отдаёт категориальную `l`/`n`/`h`, MODIS — числовую 0-100.
Буквальная реализация не отфильтровала бы ни одной точки MODIS.

Два решения, определяющих модуль:

* **Границы MODIS — параметр конфигурации, а не константа.** Соответствие
  числовой шкалы категориальной не является тождеством: NASA документирует
  0-100 как «уверенность алгоритма», а не как три класса. Любое разбиение —
  наша интерпретация, и она подлежит явной фиксации и пересмотру.
* **Нормализация выполняется при чтении, а не при записи.** Хранилище
  append-only (SPEC-2), и записанный уровень заморозил бы пороги: смена границ
  задним числом сделала бы историю несопоставимой. Хранится сырое значение
  сенсора, уровень вычисляется под текущие пороги.

Неизвестное значение — ошибка, а не тихий `low`. Тихая деградация к самому
строгому классу выглядела бы как работающий фильтр и незаметно выбрасывала бы
реальные пожары.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class Level(IntEnum):
    """Упорядочен, чтобы порог сравнивался, а не перечислялся."""
    LOW = 0
    NOMINAL = 1
    HIGH = 2

    @classmethod
    def parse(cls, name: str) -> "Level":
        try:
            return cls[name.strip().upper()]
        except KeyError:
            raise UnknownConfidenceError(
                f"unknown confidence level {name!r}; "
                f"expected one of: {', '.join(l.name.lower() for l in cls)}"
            ) from None

    def __str__(self) -> str:
        return self.name.lower()


class UnknownConfidenceError(ValueError):
    """Значение confidence не распознано — молча деградировать нельзя."""


# VIIRS: категориальная шкала. Полный набор допустимых значений.
_VIIRS_SCALE = {"l": Level.LOW, "n": Level.NOMINAL, "h": Level.HIGH}


@dataclass(frozen=True)
class Thresholds:
    modis_low_below: int = 30
    modis_high_at_least: int = 80
    min_level: Level = Level.NOMINAL

    def __post_init__(self):
        if not 0 <= self.modis_low_below <= self.modis_high_at_least <= 100:
            raise ValueError(
                f"MODIS boundaries must satisfy 0 <= low_below <= high_at_least "
                f"<= 100, got {self.modis_low_below} and {self.modis_high_at_least}"
            )


def load_thresholds(config_path: str | Path) -> Thresholds:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8")).get("confidence", {})
    return Thresholds(
        modis_low_below=int(cfg.get("modis_low_below", 30)),
        modis_high_at_least=int(cfg.get("modis_high_at_least", 80)),
        min_level=Level.parse(cfg.get("min_level", "nominal")),
    )


def scale_of(sensor: str) -> str:
    """Какой шкалой пользуется сенсор. Неизвестный сенсор — ошибка."""
    s = sensor.upper()
    if s.startswith("VIIRS"):
        return "categorical"
    if s.startswith("MODIS"):
        return "numeric"
    raise UnknownConfidenceError(
        f"unknown sensor {sensor!r}: cannot tell which confidence scale it uses"
    )


def normalize(sensor: str, raw: str, thresholds: Thresholds | None = None) -> Level:
    """Привести confidence любого сенсора к общей трёхуровневой шкале (AC-01, AC-02)."""
    thresholds = thresholds or Thresholds()
    value = (raw or "").strip()
    if not value:
        raise UnknownConfidenceError(f"{sensor}: empty confidence value")

    if scale_of(sensor) == "categorical":
        level = _VIIRS_SCALE.get(value.lower())
        if level is None:
            raise UnknownConfidenceError(
                f"{sensor}: unknown categorical confidence {raw!r}; "
                f"expected one of: {', '.join(sorted(_VIIRS_SCALE))}"
            )
        return level

    try:
        n = int(value)
    except ValueError:
        raise UnknownConfidenceError(
            f"{sensor}: numeric confidence expected, got {raw!r}"
        ) from None
    if not 0 <= n <= 100:
        raise UnknownConfidenceError(
            f"{sensor}: numeric confidence out of documented range 0..100: {n}"
        )
    if n < thresholds.modis_low_below:
        return Level.LOW
    if n >= thresholds.modis_high_at_least:
        return Level.HIGH
    return Level.NOMINAL


def passes(level: Level, min_level: Level) -> bool:
    return level >= min_level


def partition(detections, thresholds: Thresholds | None = None):
    """Разделить детекции на прошедшие порог и отсеянные (AC-03, AC-04).

    Возвращает два списка, а не один: отсеянные не выбрасываются, а уходят в
    служебный слой. Хранилище при этом не трогается вовсе — оно append-only,
    и фильтрация здесь является представлением, а не мутацией.
    """
    thresholds = thresholds or Thresholds()
    kept, rejected = [], []
    for d in detections:
        level = normalize(d.sensor, d.confidence, thresholds)
        (kept if passes(level, thresholds.min_level) else rejected).append(d)
    return kept, rejected
