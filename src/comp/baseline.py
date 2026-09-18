"""Пороговый baseline по dNBR (SPEC-18).

Пороги USGS выведены для хвойных лесов и здесь заведомо не оптимальны —
это и есть смысл baseline: показать, сколько даёт шкала «из учебника»,
прежде чем обучать что-либо. Пороги зафиксированы ДО замера и не двигаются
после того, как результат увиден.
"""

from __future__ import annotations

import numpy as np

from .chips import BsChip
from .features import stack

# USGS: low 0.10-0.27, moderate 0.27-0.66, high > 0.66.
USGS = (0.10, 0.27, 0.66)


def predict(chip: BsChip, thresholds: tuple[float, float, float] = USGS) -> np.ndarray:
    low, moderate, high = thresholds
    dnbr = stack(chip)[0]
    out = np.zeros(dnbr.shape, dtype=np.uint8)
    out[dnbr >= low] = 1
    out[dnbr >= moderate] = 2
    out[dnbr >= high] = 3
    out[~chip.valid()] = 0
    return out
