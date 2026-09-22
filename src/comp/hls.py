"""SPEC-69: разбиение HLS Burn Scars по тайлам, аугментация и 8 преобразований при инференсе."""
import hashlib, re
import torch


def tile_of(name):
    return re.search(r'HLS\.S30\.(T[0-9A-Z]+)\.', name).group(1)


def tile_holdout(names, frac=0.2):
    """True — сцена в отложенной части: 20 % тайлов по хешу, тайл целиком по одну сторону."""
    return [int(hashlib.sha256(f'hls-hold:{tile_of(n)}'.encode()).hexdigest(), 16) % 1000 < frac * 1000 for n in names]


def augment(x, y, rng, rot=False, gain=0.0):
    """x (B,C,H,W), y (B,H,W): отражения, повороты на 90°, усиление каналов ×[1-gain, 1+gain]."""
    if rng.random() < 0.5: x, y = x.flip(3), y.flip(2)
    if rng.random() < 0.5: x, y = x.flip(2), y.flip(1)
    if rot:
        k = int(rng.integers(4)); x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if gain:
        g = torch.as_tensor(rng.uniform(1 - gain, 1 + gain, (x.shape[0], x.shape[1], 1, 1)), dtype=x.dtype, device=x.device)
        x = x * g   # ponytail: усиление в нормализованном пространстве, не в отражении — сдвига среднего нет
    return x, y


def tta8(net, x):
    """Средние логиты по 8 преобразованиям (4 поворота × отражение), каждое возвращено обратно."""
    acc = 0
    for f in (False, True):
        xf = x.flip(3) if f else x
        for k in range(4):
            lg = torch.rot90(net(torch.rot90(xf, k, (2, 3))).float(), -k, (2, 3))
            acc = acc + (lg.flip(3) if f else lg)
    return acc / 8
