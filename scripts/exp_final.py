"""Очная ставка кандидатов в ОДНОЙ шкале — микро, как считает проверяющая
система (SPEC-19).

Предыдущие замеры сравнивали бустинг (макро, среднее IoU по чипам) с сетью
(микро, пул пикселей) — величины разные, и сравнение было недействительным.
Здесь всё меряется микро, на одних и тех же 35 настроечных чипах.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs_micro
from src.comp.model import train, sample_chip, SEED
from src.comp.postproc import drop_small
from src.comp.unet import load as load_net

NET = sys.argv[1] if len(sys.argv) > 1 else 'models/exp_d5.pt'

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])

t0=time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
boost = train(np.concatenate(xs), np.concatenate(ys))
print(f'бустинг обучен за {time.time()-t0:.0f}с', flush=True)

net, mean_np, std_np, DEV = load_net(NET)
mean = torch.tensor(mean_np, device=DEV).view(1,-1,1,1)
std = torch.tensor(std_np, device=DEV).view(1,-1,1,1)
print(f'сеть {NET} загружена на {DEV}', flush=True)

truths, pb_all, pn_all, oks = [], [], [], []
with torch.no_grad():
    for c in tune:
        ch = d.load(c)
        feats = np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32)
        flat = feats.reshape(len(NAMES), -1).T
        pb_all.append(boost.predict_proba(flat).reshape(*ch.shape, 4))
        x = (torch.from_numpy(feats).unsqueeze(0).to(DEV) - mean) / std
        logits = net(x).float()
        for dims in ([2], [3], [2, 3]):          # отражения: на глубокой сети дают +0.006
            logits = logits + torch.flip(net(torch.flip(x, dims)).float(), dims)
        pn_all.append((logits / 4).softmax(1)[0].permute(1,2,0).cpu().numpy())
        truths.append(ch.mask); oks.append(ch.valid())
print('вероятности посчитаны', flush=True)

def measure(w, min_blob):
    preds = []
    for pb, pn, ok in zip(pb_all, pn_all, oks):
        p = (1-w)*pb + w*pn
        pred = p.argmax(2).astype(np.uint8); pred[~ok] = 0
        preds.append(drop_small(pred, min_blob) if min_blob else pred)
    return score_bs_micro(truths, preds)

print('\nвес сети  фильтр  IoU_burn  mIoU_sev   кл1    кл2    кл3   взвеш.')
for w in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
    for mb in (0, 100):
        r = measure(w, mb)
        weighted = (0.35*r['iou_burn'] + 0.30*r['miou_sev']) / 0.65
        print('%-9.1f %-7s %8.4f %9.4f %6.3f %6.3f %6.3f %8.4f' % (
            w, mb or '—', r['iou_burn'], r['miou_sev'],
            r['per_class'][1], r['per_class'][2], r['per_class'][3], weighted), flush=True)
