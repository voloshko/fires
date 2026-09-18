"""Пиксельные признаки BS-чипа (SPEC-18/19).

Шкала dNBR, принятая USGS, выведена для хвойных лесов; в степи запас биомассы
вдесятеро меньше, поэтому landcover подаётся признаком, а не игнорируется.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .chips import BsChip

NAMES = (
    "dnbr", "rbr", "nbr_pre", "nbr_post",
    "ndvi_pre", "ndvi_post", "dndvi",
    "nbr2_post", "b12_post", "b8a_post",
    "dnbr_win5", "dnbr_win15", "dnbr_std5", "dnbr_chip",
    "landcover", "slope", "dem", "vv", "vh",
)


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    return np.divide(a - b, total, out=np.zeros_like(total), where=total != 0)


def stack(chip: BsChip) -> np.ndarray:
    """(n_features, H, W) float32."""
    pre, post = chip.pre.astype(np.float32), chip.post.astype(np.float32)
    if not post.size:
        post = pre
    nbr_pre, nbr_post = _ratio(pre[6], pre[8]), _ratio(post[6], post[8])
    dnbr = nbr_pre - nbr_post
    # RBR гасит зависимость от предпожарного состояния (Parks et al.).
    rbr = np.divide(dnbr, nbr_pre + 1.001, out=np.zeros_like(dnbr), where=True)
    ndvi_pre, ndvi_post = _ratio(pre[6], pre[2]), _ratio(post[6], post[2])
    layers = [
        dnbr, rbr, nbr_pre, nbr_post,
        ndvi_pre, ndvi_post, ndvi_pre - ndvi_post,
        _ratio(post[7], post[8]),          # NBR2: гарь в SWIR
        post[8] / 10000.0, post[6] / 10000.0,
        # Контекст: гарь — связное пятно, а не россыпь пикселей. На AF-чипах
        # ровно эта поправка на окно подняла F1 с 0.027 до 0.595.
        uniform_filter(dnbr, 5, mode="nearest"),
        uniform_filter(dnbr, 15, mode="nearest"),
        np.sqrt(np.maximum(uniform_filter(dnbr * dnbr, 5, mode="nearest")
                           - uniform_filter(dnbr, 5, mode="nearest") ** 2, 0.0)),
        # Сдвиг всей сцены: у разных пар до/после свой уровень dNBR.
        np.full_like(dnbr, float(np.median(dnbr[chip.valid()])) if chip.valid().any() else 0.0),
        chip.aux[2].astype(np.float32),    # landcover
        chip.aux[1].astype(np.float32),    # slope
        chip.aux[0].astype(np.float32) / 1000.0,
        chip.sar[0].astype(np.float32) / 1000.0,
        chip.sar[1].astype(np.float32) / 1000.0,
    ]
    return np.stack(layers).astype(np.float32)
