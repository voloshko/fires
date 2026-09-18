"""Ансамбль сети и бустинга по гари (SPEC-19, SPEC-20).

Модели ошибаются по-разному: бустинг видит пиксель и его окно, сеть — форму
пятна целиком. Усреднение вероятностей даёт больше, чем любая из них, причём
сильнее всего — по слабому классу, самому спорному: 0.387 против 0.363 у сети
и 0.311 у бустинга.

Вес найден на 35 настроечных чипах в микро-шкале, как считает проверяющая
система. Плато широкое — от 0.5 до 0.7 результат меняется на 0.003, — поэтому
0.6 не является точкой, подогнанной под замер.

Ансамбль дважды отвергался ошибочно: сперва мерился со слабой сетью, потом в
макро-шкале. История в evidence/hypotheses.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.comp.chips import BsChip
from src.comp.features import stack
from src.comp.model import feature_names
from src.comp.postproc import drop_small

NET_WEIGHT = 0.6

# Фильтр мелких пятен для ансамбля ОТКЛЮЧЁН. Он давал +0.027, пока сеть была
# мелкой и сорила ложными пятнами; на глубине 6 перестал помогать, на глубине 7
# в ансамбле мешает: 0.5961/0.5806 без него против 0.5937/0.5776 с ним. Он
# компенсировал нехватку контекста — когда контекста хватает, он режет
# настоящие мелкие гари.
MIN_BLOB_ENSEMBLE = 0


def predict(net_model, boost_model, chip: BsChip,
            net_weight: float = NET_WEIGHT,
            min_blob: int = MIN_BLOB_ENSEMBLE,
            tta: bool = True) -> np.ndarray:
    """Маска степеней поражения по смеси вероятностей.

    `net_model` — одна сеть или список сетей одной роли (сидовый ансамбль):
    их вероятности усредняются до смешивания с бустингом. `net_weight=1.0`
    вырождается в одни сети, `0.0` — в один бустинг; обе вырожденные точки
    измерены и хуже смеси.
    """
    if not 0.0 <= net_weight <= 1.0:
        raise ValueError(f"вес сети вне [0, 1]: {net_weight}")

    from src.comp import unet

    nets = net_model if isinstance(net_model, (list, tuple)) and not hasattr(net_model[0], "eval") else [net_model]
    p_net = np.mean([unet.probs(m, chip, tta) for m in nets], axis=0)

    names = feature_names(boost_model)
    feats = np.nan_to_num(stack(chip, names), posinf=0.0, neginf=0.0).astype(np.float32)
    p_boost = boost_model.predict_proba(
        feats.reshape(len(names), -1).T).reshape(*chip.shape, p_net.shape[2])

    mixed = net_weight * p_net + (1.0 - net_weight) * p_boost
    pred = mixed.argmax(2).astype(np.uint8)
    # Под маской облаков (SCL) фон НЕ ставится принудительно. Истина размечена и
    # под облаками — 23 % истинной гари настроечных чипов лежит под маской, — а
    # маска SCL консервативна: дымка, тень, cirrus. Сеть видит сырые отражения
    # и контекст пятна и угадывает там лучше, чем гарантированный пропуск:
    # IoU_burn 0.5961 → 0.7178, mIoU_sev 0.5806 → 0.6737 на 35 чипах вне обучения.
    # Бустинг под маской слабее сети (0.7103 при смеси), поэтому там — сеть одна.
    blind = ~chip.valid()
    pred[blind] = p_net.argmax(2).astype(np.uint8)[blind]
    # …кроме классов, под которыми сам разметчик ставил ноль (тень, плотное
    # облако, нет данных): там истины нет по построению, а предсказанная гарь —
    # чистый ложный положительный. 0.7178/0.6737 → 0.7337/0.6907.
    pred[chip.label_zero()] = 0
    return drop_small(pred, min_blob)


def available(net_path: str | Path, boost_path: str | Path) -> bool:
    """Ансамбль возможен, только если на диске есть обе модели и стоит torch."""
    from src.comp import unet

    return Path(net_path).exists() and Path(boost_path).exists() and unet.available()
