"""Стенд перебора гипотез по сети (SPEC-19).

Обучение на 144 чипах, замер на 35 настроечных. Отложенные 45 не трогаются.
Одна обученная сеть меряется сразу четырьмя способами постобработки, потому что
обучение стоит минуты, а замер — секунды: платить за него дважды незачем.

Проверяемые гипотезы:
  вес фона        — потеря IoU идёт от ложной гари на фоне (4.3 % фона = 43 %
                    площади гари), значит фон надо штрафовать сильнее;
  отражения       — усреднение по четырём отражениям на замере;
  связные области — выбрасывание пятен гари мельче порога.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_unet import (  # noqa: E402
    BATCH, CROP, SEED, UNet, cache, gpu_cache, iou_scores, load_split, make_batch, normalise,
)
from src.comp.features import NAMES  # noqa: E402

import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BG = float(os.environ.get("BG_WEIGHT", 0.25))
EPOCHS = int(os.environ.get("EPOCHS", 200))
CROPSZ = int(os.environ.get("CROP", CROP))
MIN_BLOB = int(os.environ.get("MIN_BLOB", 200))
DEPTH = int(os.environ.get("DEPTH", 4))
WIDTH = int(os.environ.get("WIDTH", 48))
TAG = os.environ.get("TAG", "x")
RUNSEED = int(os.environ.get("SEED", SEED))
MASKCH = int(os.environ.get("MASKCH", 0))   # канал валидности на входе
LOSS = os.environ.get("LOSS", "dice")        # dice | lovasz — что добавляется к CE
BOUNDARY = float(os.environ.get("BOUNDARY", 0))   # >0: вес пикселей у кромки истинной гари (×(1+BOUNDARY))


def boundary_weight(y, k=3):
    """Кромка истинной гари: пиксели, где в окне (2k+1) есть и гарь, и фон.
    Слабый класс — кольцо на кромке; ошибка «где кончается гарь» — главная."""
    burn = (y > 0).float().unsqueeze(1)
    dil = F.max_pool2d(burn, 2*k+1, 1, k); ero = 1 - F.max_pool2d(1 - burn, 2*k+1, 1, k)
    return 1.0 + BOUNDARY * (dil - ero).squeeze(1)


def lovasz_grad(gt_sorted):
    """Градиент расширения Ловаса для Жаккара (Berman et al., 2018)."""
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_softmax(probas, labels, classes=(1, 2, 3)):
    """Пул по ВСЕМУ батчу, а не среднее по изображениям: метрика кейса — микро.
    Классы гари; фон получает свой сигнал через CE."""
    p = probas.permute(0, 2, 3, 1).reshape(-1, probas.shape[1])
    y = labels.reshape(-1)
    losses = []
    for c in classes:
        fg = (y == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - p[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    # и Жаккар по гари целиком — это IoU_burn
    fg = (y > 0).float(); pb = 1 - p[:, 0]
    errors_sorted, perm = torch.sort((fg - pb).abs(), 0, descending=True)
    losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


def drop_small(pred: np.ndarray, min_px: int) -> np.ndarray:
    """Пятно гари меньше порога — почти наверняка ложное срабатывание.

    Настоящая гарь связна и велика; одиночные пиксели среди фона возникают там,
    где сцена шумит, а не там, где горело.
    """
    from scipy.ndimage import label

    burn = pred > 0
    marks, count = label(burn)
    if not count:
        return pred
    sizes = np.bincount(marks.reshape(-1))
    small = np.isin(marks, np.flatnonzero(sizes < min_px))
    out = pred.copy()
    out[small] = 0
    return out


def infer(net, x16, mean_t, std_t, tta: bool) -> np.ndarray:
    x = (torch.from_numpy(x16.astype(np.float32)).unsqueeze(0).to(DEV) - mean_t) / std_t
    logits = net(x).float()
    if tta:
        for dims in ([2], [3], [2, 3]):
            logits = logits + torch.flip(net(torch.flip(x, dims)).float(), dims)
        logits = logits / 4
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def main():
    torch.manual_seed(RUNSEED); np.random.seed(RUNSEED)
    d, fit, tune = load_split("data/comp/train/bs", "data/comp/split_bs.json")
    print(f"[{TAG}] вес фона {BG}, эпох {EPOCHS}, вырезка {CROPSZ}, "
          f"глубина {DEPTH}, ширина {WIDTH}, "
          f"обучение {len(fit)}, замер {len(tune)}", flush=True)
    xtr, ytr, oktr = cache(d, fit)
    xva, yva, okva = cache(d, tune)
    mean, std = normalise(xtr)
    if MASKCH:
        # Сеть не отличает «нет гари» от «облако»: признаки под маской — сырые
        # отражения через SCL. Канал валидности говорит ей, где она слепа.
        xtr = [np.concatenate([x, ok[None].astype(np.float16)]) for x, ok in zip(xtr, oktr)]
        xva = [np.concatenate([x, ok[None].astype(np.float16)]) for x, ok in zip(xva, okva)]
        mean = np.append(mean, 0.5).astype(np.float32); std = np.append(std, 0.5).astype(np.float32)
    cin = len(NAMES) + MASKCH

    net = UNet(cin, w=WIDTH, depth=DEPTH).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = EPOCHS * max(1, len(fit) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    weight = torch.tensor([BG, 1.0, 1.0, 1.0], device=DEV)
    scaler = torch.amp.GradScaler(DEV)
    mean_t = torch.tensor(mean, device=DEV).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=DEV).view(1, -1, 1, 1)
    rng = np.random.default_rng(RUNSEED)
    t0 = time.time()

    X, Y = gpu_cache(xtr, ytr, mean, std); del xtr
    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = rng.permutation(len(fit))
        for k in range(0, len(order) - BATCH + 1, BATCH):
            x, y = make_batch(X, Y, torch.as_tensor(order[k:k + BATCH], device=DEV), rng, CROPSZ)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV):
                out = net(x)
                if BOUNDARY > 0:
                    loss = (F.cross_entropy(out, y, weight=weight, reduction="none") * boundary_weight(y)).mean()
                else:
                    loss = F.cross_entropy(out, y, weight=weight)
                if LOSS == "lovasz":
                    loss = loss + lovasz_softmax(out.float().softmax(1), y)
                else:
                    p_burn = 1 - out.softmax(1)[:, 0]
                    t_burn = (y > 0).float()
                    loss = loss + 1 - (2 * (p_burn * t_burn).sum() + 1) / (p_burn.sum() + t_burn.sum() + 1)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched.last_epoch < steps - 1: sched.step()

    net.eval()
    print(f"[{TAG}] обучено за {time.time()-t0:.0f}с", flush=True)
    with torch.no_grad(), torch.amp.autocast(DEV):
        for tta in (False, True):
            raw = [infer(net, x16, mean_t, std_t, tta) for x16 in xva]
            for min_px, zero in ((0, False), (0, True), (MIN_BLOB, False)):
                tt, pp = [], []
                for pred, y, ok in zip(raw, yva, okva):
                    p = drop_small(pred, min_px) if min_px else pred.copy()
                    if zero: p[~ok] = 0      # старое правило — для сравнения с прежними числами
                    tt.append(y.reshape(-1)); pp.append(p.reshape(-1))
                burn, miou, per = iou_scores(np.concatenate(tt), np.concatenate(pp))
                how = ("отражения " if tta else "обычно    ") + (f"фильтр {min_px}" if min_px else "без фильтра") + (" ноль под маской" if zero else "")
                print(f"[{TAG}] {how:42s} IoU_burn {burn:.4f}  mIoU_sev {miou:.4f}  "
                      f"[{per[1]:.3f} {per[2]:.3f} {per[3]:.3f}]", flush=True)
    torch.save({"state": net.state_dict(), "mean": mean, "std": std, "names": NAMES,
                "bg_weight": BG, "epochs": EPOCHS, "crop": CROPSZ,
                "depth": DEPTH, "width": WIDTH, "maskch": MASKCH, "seed": RUNSEED, "loss": LOSS, "boundary": BOUNDARY}, f"models/exp_{TAG}.pt")


if __name__ == "__main__":
    main()
