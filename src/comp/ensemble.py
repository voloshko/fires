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
from src.comp.features import NAMES, stack
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

    `net_weight=1.0` вырождается в одну сеть, `0.0` — в один бустинг; обе
    вырожденные точки измерены и хуже смеси.
    """
    if not 0.0 <= net_weight <= 1.0:
        raise ValueError(f"вес сети вне [0, 1]: {net_weight}")

    import torch

    net, mean, std, device = net_model
    feats = np.nan_to_num(stack(chip), posinf=0.0, neginf=0.0).astype(np.float32)

    with torch.no_grad():
        x = (torch.from_numpy(feats).unsqueeze(0).to(device)
             - torch.as_tensor(mean, device=device).view(1, -1, 1, 1)) \
            / torch.as_tensor(std, device=device).view(1, -1, 1, 1)
        logits = net(x).float()
        if tta:
            for dims in ([2], [3], [2, 3]):
                logits = logits + torch.flip(net(torch.flip(x, dims)).float(), dims)
            logits = logits / 4
        p_net = logits.softmax(1)[0].permute(1, 2, 0).cpu().numpy()

    p_boost = boost_model.predict_proba(
        feats.reshape(len(NAMES), -1).T).reshape(*chip.shape, p_net.shape[2])

    mixed = net_weight * p_net + (1.0 - net_weight) * p_boost
    pred = mixed.argmax(2).astype(np.uint8)
    pred[~chip.valid()] = 0
    return drop_small(pred, min_blob)


def available(net_path: str | Path, boost_path: str | Path) -> bool:
    """Ансамбль возможен, только если на диске есть обе модели и стоит torch."""
    from src.comp import unet

    return Path(net_path).exists() and Path(boost_path).exists() and unet.available()
