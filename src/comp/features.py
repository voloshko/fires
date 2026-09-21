"""Пиксельные признаки BS-чипа (SPEC-18/19).

Шкала dNBR, принятая USGS, выведена для хвойных лесов; в степи запас биомассы
вдесятеро меньше, поэтому landcover подаётся признаком, а не игнорируется.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import uniform_filter

from .chips import BsChip

NAMES_BASE = (
    "dnbr", "rbr", "nbr_pre", "nbr_post",
    "ndvi_pre", "ndvi_post", "dndvi",
    "nbr2_post", "b12_post", "b8a_post",
    "dnbr_win5", "dnbr_win15", "dnbr_std5", "dnbr_chip",
    "landcover", "slope", "dem", "vv", "vh",
)
# Радар «после» и его изменение: единственный вход, который видит сквозь
# облака, — а под маской SCL лежит 23 % истинной гари.
NAMES_S1 = NAMES_BASE + ("vv_post", "vh_post", "dvv", "dvh")
# Набор по умолчанию для НОВЫХ моделей. Обученные модели несут свой набор имён и
# грузятся по нему, так что смена умолчания старые модели не ломает.
# SPEC-47: SWIR-индексы для бледной гари. MIRBI = 10·B12 − 9.8·B11 + 2 (Trigg & Flasse),
# в единицах отражения 0..1; NBR2 = (B11 − B12)/(B11 + B12). Среди базовых каналов
# B11 не было вовсе — десять пожаров с dNBR ≈ 0.15 сети не видели, по dMIRBI они
# отделяются от фона на всех десяти.
NAMES_SWIR = NAMES_BASE + ("mirbi_pre", "mirbi_post", "dmirbi", "nbr2_pre", "dnbr2", "b11_post")
NAMES = {"s1": NAMES_S1, "swir": NAMES_SWIR}.get(os.environ.get("FEATURES", "base"), NAMES_BASE)


def normalise_post(chip: BsChip, dnbr_tol: float = 0.05) -> BsChip:
    """Полосы «после» линейно приводятся к «до» по устойчивым пикселям
    (|dNBR| < dnbr_tol, обе сцены валидны): убирает разницу освещения и дымки
    между датами до расчёта индексов (SPEC-30). SCL не трогается. Без сцены
    «после» или без устойчивых пикселей чип возвращается как есть."""
    if not chip.post.size:
        return chip
    pre, post = chip.pre.astype(np.float32), chip.post.astype(np.float32)
    dnbr = _ratio(pre[6], pre[8]) - _ratio(post[6], post[8])
    stable = chip.valid() & (np.abs(dnbr) < dnbr_tol)
    if stable.sum() < 500:
        return chip
    out = post.copy()
    for b in range(9):
        a, c = np.polyfit(post[b][stable], pre[b][stable], 1)
        out[b] = a * post[b] + c
    from dataclasses import replace
    return replace(chip, post=np.clip(out, 0, 65535).astype(chip.post.dtype))


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    return np.divide(a - b, total, out=np.zeros_like(total), where=total != 0)


def stack(chip: BsChip, names: tuple[str, ...] = None) -> np.ndarray:
    """(len(names), H, W) float32; порядок слоёв — порядок `names`."""
    names = tuple(names) if names is not None else NAMES
    pre, post = chip.pre.astype(np.float32), chip.post.astype(np.float32)
    if not post.size:
        post = pre
    nbr_pre, nbr_post = _ratio(pre[6], pre[8]), _ratio(post[6], post[8])
    dnbr = nbr_pre - nbr_post
    # RBR гасит зависимость от предпожарного состояния (Parks et al.).
    rbr = np.divide(dnbr, nbr_pre + 1.001, out=np.zeros_like(dnbr), where=True)
    ndvi_pre, ndvi_post = _ratio(pre[6], pre[2]), _ratio(post[6], post[2])
    sar_post = chip.sar_post if chip.sar_post is not None else chip.sar   # нет радара после — изменение 0
    vv, vh = chip.sar[0].astype(np.float32) / 1000.0, chip.sar[1].astype(np.float32) / 1000.0
    vv_post, vh_post = sar_post[0].astype(np.float32) / 1000.0, sar_post[1].astype(np.float32) / 1000.0
    layers = {
        "dnbr": dnbr,
        "rbr": rbr,
        "nbr_pre": nbr_pre,
        "nbr_post": nbr_post,
        "ndvi_pre": ndvi_pre,
        "ndvi_post": ndvi_post,
        "dndvi": ndvi_pre - ndvi_post,
        "nbr2_post": _ratio(post[7], post[8]),
        "nbr2_pre": _ratio(pre[7], pre[8]),
        "dnbr2": _ratio(pre[7], pre[8]) - _ratio(post[7], post[8]),
        "mirbi_pre": 10.0 * pre[8] / 10000.0 - 9.8 * pre[7] / 10000.0 + 2.0,
        "mirbi_post": 10.0 * post[8] / 10000.0 - 9.8 * post[7] / 10000.0 + 2.0,
        "dmirbi": (10.0 * post[8] - 9.8 * post[7] - 10.0 * pre[8] + 9.8 * pre[7]) / 10000.0,
        "b11_post": post[7] / 10000.0,
        "b12_post": post[8] / 10000.0,
        "b8a_post": post[6] / 10000.0,
        "dnbr_win5": uniform_filter(dnbr, 5, mode="nearest"),
        "dnbr_win15": uniform_filter(dnbr, 15, mode="nearest"),
        "dnbr_std5": np.sqrt(np.maximum(uniform_filter(dnbr * dnbr, 5, mode="nearest")
                           - uniform_filter(dnbr, 5, mode="nearest") ** 2, 0.0)),
        "dnbr_chip": np.full_like(dnbr, float(np.median(dnbr[chip.valid()])) if chip.valid().any() else 0.0),
        "landcover": chip.aux[2].astype(np.float32),
        "slope": chip.aux[1].astype(np.float32),
        "dem": chip.aux[0].astype(np.float32) / 1000.0,
        "vv": vv,
        "vh": vh,
        "vv_post": vv_post, "vh_post": vh_post,
        "dvv": vv_post - vv, "dvh": vh_post - vh,
    }
    # Комментарии к слоям: контекстные окна dNBR — гарь связное пятно, на AF-чипах
    # поправка на окно подняла F1 с 0.027 до 0.595; dnbr_chip — сдвиг всей сцены.
    return np.stack([layers[n] for n in names]).astype(np.float32)
