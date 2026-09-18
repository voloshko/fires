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
    BATCH, CROP, SEED, UNet, cache, iou_scores, load_split, normalise,
)
from src.comp.features import NAMES  # noqa: E402

import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BG = float(os.environ.get("BG_WEIGHT", 0.25))
EPOCHS = int(os.environ.get("EPOCHS", 200))
CROPSZ = int(os.environ.get("CROP", CROP))
MIN_BLOB = int(os.environ.get("MIN_BLOB", 200))
TAG = os.environ.get("TAG", "x")


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
    torch.manual_seed(SEED); np.random.seed(SEED)
    d, fit, tune = load_split("data/comp/train/bs", "data/comp/split_bs.json")
    print(f"[{TAG}] вес фона {BG}, эпох {EPOCHS}, вырезка {CROPSZ}, "
          f"обучение {len(fit)}, замер {len(tune)}", flush=True)
    xtr, ytr, _ = cache(d, fit)
    xva, yva, okva = cache(d, tune)
    mean, std = normalise(xtr)

    net = UNet(len(NAMES), w=48).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = EPOCHS * max(1, len(fit) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    weight = torch.tensor([BG, 1.0, 1.0, 1.0], device=DEV)
    scaler = torch.amp.GradScaler(DEV)
    mean_t = torch.tensor(mean, device=DEV).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=DEV).view(1, -1, 1, 1)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = rng.permutation(len(fit))
        for k in range(0, len(order) - BATCH + 1, BATCH):
            xb, yb = [], []
            for i in order[k:k + BATCH]:
                h, w = ytr[i].shape
                r, c = rng.integers(0, h - CROPSZ + 1), rng.integers(0, w - CROPSZ + 1)
                patch = xtr[i][:, r:r + CROPSZ, c:c + CROPSZ].astype(np.float32)
                label_ = ytr[i][r:r + CROPSZ, c:c + CROPSZ]
                if rng.random() < 0.5: patch, label_ = patch[:, :, ::-1], label_[:, ::-1]
                if rng.random() < 0.5: patch, label_ = patch[:, ::-1], label_[::-1]
                xb.append(np.ascontiguousarray(patch)); yb.append(np.ascontiguousarray(label_))
            x = (torch.from_numpy(np.stack(xb)).to(DEV) - mean_t) / std_t
            y = torch.from_numpy(np.stack(yb)).to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV):
                out = net(x)
                loss = F.cross_entropy(out, y, weight=weight)
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
            for min_px in (0, MIN_BLOB):
                tt, pp = [], []
                for pred, y, ok in zip(raw, yva, okva):
                    p = drop_small(pred, min_px) if min_px else pred.copy()
                    p[~ok] = 0
                    tt.append(y.reshape(-1)); pp.append(p.reshape(-1))
                burn, miou, per = iou_scores(np.concatenate(tt), np.concatenate(pp))
                how = ("отражения " if tta else "обычно    ") + (f"фильтр {min_px}" if min_px else "без фильтра")
                print(f"[{TAG}] {how:26s} IoU_burn {burn:.4f}  mIoU_sev {miou:.4f}  "
                      f"[{per[1]:.3f} {per[2]:.3f} {per[3]:.3f}]", flush=True)
    torch.save({"state": net.state_dict(), "mean": mean, "std": std, "names": NAMES,
                "bg_weight": BG, "epochs": EPOCHS}, f"models/exp_{TAG}.pt")


if __name__ == "__main__":
    main()
