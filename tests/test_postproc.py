"""Тесты постобработки масок гари (SPEC-19)."""

import numpy as np
import pytest

from src.comp.postproc import MIN_BLOB, drop_small


def test_мелкое_пятно_выбрасывается_крупное_остаётся():
    mask = np.zeros((64, 64), np.uint8)
    mask[2:4, 2:4] = 2           # 4 пикселя — шум
    mask[20:40, 20:40] = 2       # 400 пикселей — гарь
    out = drop_small(mask, min_px=100)
    assert out[2:4, 2:4].sum() == 0
    assert (out[20:40, 20:40] == 2).all()


def test_смежные_классы_считаются_одним_пятном():
    """Пятно из классов 2 и 3 — один пожар. Если считать связность по классам
    раздельно, каждая половина окажется мелкой и обе пропадут."""
    mask = np.zeros((64, 64), np.uint8)
    mask[10:18, 10:18] = 2       # 64 пикселя
    mask[18:26, 10:18] = 3       # ещё 64, вплотную
    out = drop_small(mask, min_px=100)
    assert (out == mask).all(), "связное пятно из 128 пикселей не должно исчезнуть"


def test_нулевой_порог_ничего_не_меняет():
    mask = np.zeros((32, 32), np.uint8); mask[0, 0] = 1
    assert (drop_small(mask, 0) == mask).all()
    assert (drop_small(mask, -5) == mask).all()


def test_пустая_и_полная_маски():
    empty = np.zeros((32, 32), np.uint8)
    assert drop_small(empty).sum() == 0
    full = np.full((32, 32), 3, np.uint8)
    assert (drop_small(full) == full).all()


def test_исходная_маска_не_портится():
    mask = np.zeros((32, 32), np.uint8); mask[0, 0] = 1
    before = mask.copy()
    drop_small(mask, 10)
    assert (mask == before).all(), "функция обязана возвращать копию"


@pytest.mark.parametrize("size,survives", [(99, False), (100, True), (101, True)])
def test_порог_включительный(size, survives):
    mask = np.zeros((64, 64), np.uint8)
    flat = mask.reshape(-1)
    flat[:size] = 1              # одна строка подряд — связная область
    out = drop_small(mask.reshape(64, 64), min_px=MIN_BLOB)
    assert bool(out.sum() > 0) is survives   # np.bool_ не то же, что bool
