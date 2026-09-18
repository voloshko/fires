"""Применение обученного U-Net к чипам гари (SPEC-19, SPEC-20).

Отдельный модуль, потому что torch нужен только здесь: если его нет, точка
входа обязана продолжить работу на бустинге, а не упасть. Сети с диска
доверяем ровно настолько, насколько доверяем файлу модели — как и pickle
бустинга; сетевого доступа модуль не открывает.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.comp.chips import BsChip
from src.comp.features import NAMES, stack
from src.comp.postproc import MIN_BLOB, drop_small


def available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _modernise(state: dict) -> dict:
    """Приводит веса сетей, сохранённых до параметризации глубины, к нынешним
    именам. Переобучать их только ради имён — расточительство."""
    if any(k.startswith("down.") for k in state):
        return state
    renames = {"d1": "down.0", "d2": "down.1", "d3": "down.2", "d4": "down.3",
               "u3": "up.0", "u2": "up.1", "u1": "up.2",
               "c3": "conv.0", "c2": "conv.1", "c1": "conv.2"}
    out = {}
    for key, value in state.items():
        head, _, rest = key.partition(".")
        out[f"{renames[head]}.{rest}" if head in renames else key] = value
    return out


def load(path: str | Path):
    """Возвращает (сеть в режиме eval, среднее, разброс, устройство)."""
    import torch

    from scripts.train_unet import UNet

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(bundle["names"]) != tuple(NAMES):
        raise ValueError(
            "набор признаков модели не совпадает с текущим: "
            f"{len(bundle['names'])} против {len(NAMES)}"
        )
    state = _modernise(bundle["state"])
    # Ширина и глубина читаются из самих весов: файл модели не обязан их нести,
    # а разойтись с кодом они не должны.
    width = state["down.0.0.weight"].shape[0]
    depth = sum(1 for k in state if k.startswith("down.") and k.endswith(".0.weight"))
    net = UNet(len(NAMES), w=width, depth=depth)
    net.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device).eval()
    return net, bundle["mean"], bundle["std"], device


def predict(model, chip: BsChip, min_blob: int = MIN_BLOB, tta: bool = True) -> np.ndarray:
    """`tta` усредняет ответ по четырём отражениям чипа.

    На четырёхуровневой сети приём давал +0.0002 и был отвергнут; на
    пятиуровневой даёт +0.006 по IoU_burn. Результат зависит от глубины, поэтому
    флаг оставлен: переносить его между конфигурациями вслепую нельзя.
    """
    import torch

    net, mean, std, device = model
    feats = np.nan_to_num(stack(chip), posinf=0.0, neginf=0.0).astype(np.float32)
    x = (feats - mean[:, None, None]) / std[:, None, None]
    with torch.no_grad():
        batch = torch.from_numpy(x).unsqueeze(0).to(device)
        logits = net(batch).float()
        if tta:
            for dims in ([2], [3], [2, 3]):
                logits = logits + torch.flip(net(torch.flip(batch, dims)).float(), dims)
            logits = logits / 4
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred[~chip.valid()] = 0
    return drop_small(pred, min_blob)
