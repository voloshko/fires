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
    state = bundle["state"]
    # Ширина и глубина читаются из самих весов: файл модели не обязан их нести,
    # а разойтись с кодом они не должны.
    width = state["down.0.0.weight"].shape[0]
    depth = sum(1 for k in state if k.startswith("down.") and k.endswith(".0.weight"))
    net = UNet(len(NAMES), w=width, depth=depth)
    net.load_state_dict(bundle["state"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device).eval()
    return net, bundle["mean"], bundle["std"], device


def predict(model, chip: BsChip, min_blob: int = MIN_BLOB) -> np.ndarray:
    import torch

    net, mean, std, device = model
    feats = np.nan_to_num(stack(chip), posinf=0.0, neginf=0.0).astype(np.float32)
    x = (feats - mean[:, None, None]) / std[:, None, None]
    with torch.no_grad():
        out = net(torch.from_numpy(x).unsqueeze(0).to(device))
        pred = out.argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred[~chip.valid()] = 0
    return drop_small(pred, min_blob)
