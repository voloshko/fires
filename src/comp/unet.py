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
    names = tuple(bundle["names"])
    state = _modernise(bundle["state"])
    # Ширина и глубина читаются из самих весов: файл модели не обязан их нести,
    # а разойтись с кодом они не должны.
    width, cin = state["down.0.0.weight"].shape[:2]
    depth = sum(1 for k in state if k.startswith("down.") and k.endswith(".0.weight"))
    net = UNet(cin, w=width, depth=depth)
    net.load_state_dict(state)
    # Лишний входной канал сверх признаков — маска валидности (MASKCH=1 в стенде).
    net.names = names            # сеть считает признаки по СВОЕМУ набору имён
    net.maskch = cin - len(names)
    if net.maskch not in (0, 1):
        raise ValueError(f"у сети {cin} входов при {len(names)} признаках")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device).eval()
    return net, bundle["mean"], bundle["std"], device


def probs(model, chip: BsChip, tta: bool = True) -> np.ndarray:
    """(H, W, 4) вероятностей одной сети, с усреднением по отражениям."""
    import torch

    net, mean, std, device = model
    feats = np.nan_to_num(stack(chip, getattr(net, "names", NAMES)), posinf=0.0, neginf=0.0).astype(np.float32)
    if getattr(net, "maskch", 0):
        feats = np.concatenate([feats, chip.valid()[None].astype(np.float32)])
    x = (feats - mean[:, None, None]) / std[:, None, None]
    with torch.no_grad():
        batch = torch.from_numpy(x).unsqueeze(0).to(device)
        logits = net(batch).float()
        if tta:
            for dims in ([2], [3], [2, 3]):
                logits = logits + torch.flip(net(torch.flip(batch, dims)).float(), dims)
            logits = logits / 4
        return logits.softmax(1)[0].permute(1, 2, 0).cpu().numpy()


def predict(model, chip: BsChip, min_blob: int = MIN_BLOB, tta: bool = True) -> np.ndarray:
    """`tta` усредняет ответ по четырём отражениям чипа: на глубокой сети +0.006.

    Под маской облаков не обнуляем: истина размечена и под ними, см. ensemble.predict.
    """
    pred = probs(model, chip, tta).argmax(2).astype(np.uint8)
    pred[chip.label_zero()] = 0
    return drop_small(pred, min_blob)
