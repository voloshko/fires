"""U-Net по чипам гари (эксперимент к SPEC-19). Запускается на машине с GPU.

Попиксельный бустинг не знает о форме пятна: он видит пиксель и его окно, но не
границу гари целиком. Свёрточная сеть видит. Проверяется на той же настроечной
части, что и бустинг, — 45 отложенных чипов не трогаются.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset            # noqa: E402
from src.comp.features import NAMES, stack      # noqa: E402

SEED = 20260918
CROP = 256


def _arg(pos, default, cast=int):
    """Терпимый разбор: модуль импортируется инференсом ради класса UNet,
    и чужой argv не должен его ронять."""
    try:
        return cast(sys.argv[pos])
    except (IndexError, ValueError):
        return default


EPOCHS = _arg(1, 60)
WIDTH = _arg(2, 48)
TAG = _arg(3, "a", str)
USE_ALL = TAG.startswith("final")
BATCH = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin: int, classes: int = 4, w: int = 48):
        super().__init__()
        self.d1, self.d2, self.d3, self.d4 = block(cin, w), block(w, 2*w), block(2*w, 4*w), block(4*w, 8*w)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8*w, 4*w, 2, 2); self.c3 = block(8*w, 4*w)
        self.u2 = nn.ConvTranspose2d(4*w, 2*w, 2, 2); self.c2 = block(4*w, 2*w)
        self.u1 = nn.ConvTranspose2d(2*w, w, 2, 2);   self.c1 = block(2*w, w)
        self.head = nn.Conv2d(w, classes, 1)

    def forward(self, x):
        a = self.d1(x); b = self.d2(self.pool(a)); c = self.d3(self.pool(b)); d = self.d4(self.pool(c))
        x = self.c3(torch.cat([self.u3(d), c], 1))
        x = self.c2(torch.cat([self.u2(x), b], 1))
        x = self.c1(torch.cat([self.u1(x), a], 1))
        return self.head(x)


def load_split(root: str, split_path: str, use_all: bool = False):
    """use_all: обучение на ВСЕХ чипах — режим финальной модели.

    Настроечная часть тогда входит в обучение, и замер на ней перестаёт что-либо
    означать: лучшая эпоха не выбирается, берётся последняя. Так и должно быть —
    конфигурация выбрана заранее, а 25 % данных простаивать не должны.
    """
    d = BsDataset(root)
    if use_all:
        every = sorted(c for c in d.chip_ids() if d.has_post(c))
        return d, every, []
    split = json.load(open(split_path))
    ids = [c for c in split["train"] if d.has_post(c)]
    rank = sorted(ids, key=lambda c: hashlib.sha256(f"tune:{c}".encode()).hexdigest())
    return d, sorted(rank[35:]), sorted(rank[:35])


def cache(dataset, chip_ids):
    """Признаки в float16: 19 каналов × 512² × 2 байта ≈ 10 МБ на чип."""
    xs, ys, ok = [], [], []
    for n, cid in enumerate(chip_ids, 1):
        chip = dataset.load(cid)
        xs.append(np.nan_to_num(stack(chip), posinf=0, neginf=0).astype(np.float16))
        ys.append(chip.mask.astype(np.int64))
        ok.append(chip.valid())
        if n % 25 == 0:
            print(f"  загружено {n}/{len(chip_ids)}", flush=True)
    return xs, ys, ok


def normalise(xs):
    """Поканальная стандартизация по обучающей части; параметры сохраняются."""
    flat = np.concatenate([x.reshape(len(NAMES), -1)[:, ::37].astype(np.float32) for x in xs], 1)
    mean = flat.mean(1).astype(np.float32)
    std = np.maximum(flat.std(1), 1e-3).astype(np.float32)
    return mean, std


def iou_scores(truth: np.ndarray, pred: np.ndarray):
    burn_t, burn_p = truth > 0, pred > 0
    union = (burn_t | burn_p).sum()
    iou_burn = float((burn_t & burn_p).sum() / union) if union else np.nan
    per = {}
    for c in (1, 2, 3):
        t, p = truth == c, pred == c
        u = (t | p).sum()
        per[c] = float((t & p).sum() / u) if u else np.nan
    return iou_burn, float(np.nanmean(list(per.values()))), per


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d, fit, tune = load_split("data/comp/train/bs", "data/comp/split_bs.json", USE_ALL)
    print(f"эпох {EPOCHS}, ширина {WIDTH}, метка {TAG}"); print(f"устройство {DEV}, обучение {len(fit)} чипов, настроечная часть {len(tune)}", flush=True)
    t0 = time.time()
    xtr, ytr, _ = cache(d, fit)
    xva, yva, okva = cache(d, tune)
    mean, std = normalise(xtr)
    print(f"данные в памяти за {time.time()-t0:.0f}с", flush=True)

    net = UNet(len(NAMES), w=WIDTH).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * max(1, len(fit) // BATCH))
    # Фон подавляется весом: он занимает подавляющую часть пикселей.
    weight = torch.tensor([0.25, 1.0, 1.0, 1.0], device=DEV)
    scaler = torch.amp.GradScaler(DEV)
    mean_t = torch.tensor(mean, device=DEV).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=DEV).view(1, -1, 1, 1)
    rng = np.random.default_rng(SEED)
    best = (-1.0, -1.0)

    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = rng.permutation(len(fit))
        total = 0.0
        for k in range(0, len(order) - BATCH + 1, BATCH):
            xb, yb = [], []
            for i in order[k:k + BATCH]:
                h, w = ytr[i].shape
                r, c = rng.integers(0, h - CROP + 1), rng.integers(0, w - CROP + 1)
                patch = xtr[i][:, r:r + CROP, c:c + CROP].astype(np.float32)
                label = ytr[i][r:r + CROP, c:c + CROP]
                if rng.random() < 0.5: patch, label = patch[:, :, ::-1], label[:, ::-1]
                if rng.random() < 0.5: patch, label = patch[:, ::-1], label[::-1]
                xb.append(np.ascontiguousarray(patch)); yb.append(np.ascontiguousarray(label))
            x = (torch.from_numpy(np.stack(xb)).to(DEV) - mean_t) / std_t
            y = torch.from_numpy(np.stack(yb)).to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV):
                out = net(x)
                loss = F.cross_entropy(out, y, weight=weight)
                # мягкий Dice по гари: метрика — IoU, а не точность попикселя
                p_burn = 1 - out.softmax(1)[:, 0]
                t_burn = (y > 0).float()
                loss = loss + 1 - (2 * (p_burn * t_burn).sum() + 1) / (p_burn.sum() + t_burn.sum() + 1)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched.last_epoch < sched.total_steps - 1: sched.step()
            total += float(loss)

        if epoch % 5 and epoch != EPOCHS:
            continue
        if not tune:                     # финальный режим: сверять не с чем
            print(f"эпоха {epoch:3d}  loss {total/max(1,len(order)//BATCH):.4f}", flush=True)
            if epoch == EPOCHS:
                torch.save({"state": net.state_dict(), "mean": mean, "std": std,
                            "names": NAMES, "epochs": EPOCHS, "chips": len(fit)},
                           f"models/bs_unet_{TAG}.pt")
                print(f"сохранена последняя эпоха: {len(fit)} чипов, выбор эпохи не производился")
            continue
        net.eval(); tt, pp = [], []
        with torch.no_grad(), torch.amp.autocast(DEV):
            for x16, y, ok in zip(xva, yva, okva):
                x = (torch.from_numpy(x16.astype(np.float32)).unsqueeze(0).to(DEV) - mean_t) / std_t
                pred = net(x).argmax(1)[0].cpu().numpy().astype(np.uint8)
                pred[~ok] = 0
                tt.append(y.reshape(-1)); pp.append(pred.reshape(-1))
        burn, miou, per = iou_scores(np.concatenate(tt), np.concatenate(pp))
        print(f"эпоха {epoch:3d}  loss {total/max(1,len(order)//BATCH):.4f}  "
              f"IoU_burn {burn:.4f}  mIoU_sev {miou:.4f}  "
              f"[{per[1]:.3f} {per[2]:.3f} {per[3]:.3f}]", flush=True)
        if burn > best[0]:
            best = (burn, miou)
            torch.save({"state": net.state_dict(), "mean": mean, "std": std,
                        "names": NAMES, "iou_burn": burn, "miou_sev": miou}, f"models/bs_unet_{TAG}.pt")
    print(f"лучшее: IoU_burn {best[0]:.4f} mIoU_sev {best[1]:.4f} за {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
