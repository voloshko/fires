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


def test_микро_и_макро_усреднение_дают_разное():
    """Чип с крошечной гарью в макро-среднем весит столько же, сколько чип, где
    выгорело всё. Тест закрепляет, что величины разные и путать их нельзя."""
    from src.comp.metric import score_bs, score_bs_micro

    big_t = np.zeros((10, 10), np.uint8); big_t[:, :] = 1          # вся площадь
    big_p = np.zeros((10, 10), np.uint8); big_p[:5, :] = 1         # угадали половину
    small_t = np.zeros((10, 10), np.uint8); small_t[0, 0] = 1      # один пиксель
    small_p = np.zeros((10, 10), np.uint8); small_p[0, 0] = 1      # угадали точно

    macro = np.mean([score_bs(big_t, big_p)["iou_burn"],
                     score_bs(small_t, small_p)["iou_burn"]])
    micro = score_bs_micro([big_t, small_t], [big_p, small_p])["iou_burn"]
    assert macro == pytest.approx(0.75)      # (0.5 + 1.0) / 2
    assert micro == pytest.approx(51 / 101)  # пул: 51 совпадение из 101
    assert abs(macro - micro) > 0.2


def test_drop_far_убирает_дальнее_пятно_и_оставляет_ближнее():
    from src.comp.postproc import drop_far
    pred = np.zeros((300, 300), np.uint8)
    pred[10:60, 10:60] = 2          # главное пятно
    pred[70:75, 70:75] = 1          # рядом (≈14 пикс.) — остаётся
    pred[280:290, 280:290] = 3      # далеко (>125) — чужой пожар
    out = drop_far(pred, 125)
    assert out[20, 20] == 2 and out[72, 72] == 1 and out[285, 285] == 0
    assert drop_far(pred, 0) is pred
