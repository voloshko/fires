"""Тесты порогового baseline активного горения (SPEC-17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.comp.af import (
    LC_BUILT_UP,
    NAMES,
    THRESHOLDS,
    AfChip,
    f1,
    features,
    predict_threshold,
    split_af,
)

SIZE = 64


def make_chip(i4_bg: float, landcover: float = 30.0, hot=None, dt: float = 12.0) -> AfChip:
    """Синтетический чип: ровный фон I4, при желании — одна горячая точка."""
    viirs = np.zeros((8, SIZE, SIZE), dtype=np.float32)
    viirs[3] = i4_bg              # I4
    viirs[4] = i4_bg - dt         # I5, чтобы ΔT был положительным
    viirs[7] = 1.0                # valid
    if hot is not None:
        row, col, delta = hot
        viirs[3, row, col] = i4_bg + delta
    aux = np.zeros((5, SIZE, SIZE), dtype=np.float32)
    aux[0] = landcover
    return AfChip(chip_id="AF_syn", viirs=viirs, aux=aux, mask=None)


@pytest.mark.parametrize("background", [305.0, 330.0])
def test_порог_контекстный_а_не_глобальный(background):
    """Одинаковая аномалия ловится и на холодном, и на тёплом фоне.

    Глобальная константа поймала бы обе точки на тёплом фоне и ни одной на
    холодном — ровно та ошибка, ради которой считается фон окна.
    """
    chip = make_chip(background, hot=(32, 32, 40.0))
    mask = predict_threshold(chip)
    assert mask[32, 32] == 1, f"аномалия +40 K не найдена на фоне {background} K"
    assert mask.sum() < 40, "аномалия размазалась — фон вычтен неверно"


def test_тёплый_фон_без_аномалии_не_детектируется():
    """Фон 340 K выше абсолютного пола, но аномалии над соседями нет."""
    assert predict_threshold(make_chip(340.0)).sum() == 0


def test_застройка_исключается_а_степь_нет():
    hot = (32, 32, 40.0)
    assert predict_threshold(make_chip(310.0, LC_BUILT_UP, hot)).sum() == 0
    assert predict_threshold(make_chip(310.0, 30.0, hot))[32, 32] == 1


def test_невалидные_пиксели_не_попадают_в_маску():
    chip = make_chip(310.0, hot=(32, 32, 40.0))
    chip.viirs[7, 32, 32] = 0.0
    assert predict_threshold(chip).sum() == 0


def test_пороги_версионированы():
    assert THRESHOLDS["version"].startswith("af-v")


def test_признаки_совпадают_по_числу_с_именами():
    chip = make_chip(310.0, hot=(32, 32, 40.0))
    stack = features(chip)
    assert stack.shape == (len(NAMES), SIZE, SIZE)
    assert np.isfinite(stack).all()
    # аномалия видна именно в контекстном признаке, а не только в сыром I4
    assert stack[NAMES.index("i4_anom")][32, 32] > 30


def test_f1_микро_усреднение():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    assert f1(truth, np.array([1, 0, 0, 0], dtype=bool))["f1"] == pytest.approx(2 / 3)
    assert f1(truth, np.zeros(4, dtype=bool))["f1"] == 0.0
    assert f1(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool))["f1"] == 0.0


def test_деление_не_пересекается_и_воспроизводится():
    meta = pd.DataFrame({"chip_id": [f"AF_tr_{i:06d}" for i in range(100)]})
    split = split_af(meta)
    parts = [set(split[k]) for k in ("train", "val", "holdout")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert sum(len(p) for p in parts) == 100
    assert split_af(meta) == split
    assert split_af(meta.iloc[::-1]) == split, "порядок строк не должен влиять"
    assert split_af(meta, seed=1)["holdout"] != split["holdout"]
