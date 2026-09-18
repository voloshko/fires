"""Радар «после» в бустинге (SPEC-19): базовые 19 признаков против 23 с
Sentinel-1 post и изменением VV/VH. Замер на 35 настроечных чипах, оба правила
под маской. Быстрая проверка на CPU; сеть с теми же признаками — в exp_unet.py
с FEATURES=s1."""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp import features as F
from src.comp.metric import score_bs_micro
from src.comp.model import train, sample_chip, SEED

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
chips_fit = [d.load(c) for c in fit]; chips_tune = [d.load(c) for c in tune]
for label, names in (('base 19', F.NAMES_BASE), ('s1 23', F.NAMES_S1)):
    F.NAMES = names                       # sample_chip/stack берут умолчание модуля
    import src.comp.model as M; M.NAMES = names
    t0 = time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
    for ch in chips_fit:
        a, b = sample_chip(ch, rng); xs.append(a); ys.append(b)
    m = train(np.concatenate(xs), np.concatenate(ys))
    preds_zero, preds_all, truths = [], [], []
    for ch in chips_tune:
        p = m.predict(F.stack(ch, names).reshape(len(names), -1).T).astype(np.uint8).reshape(ch.shape)
        z = p.copy(); z[~ch.valid()] = 0
        preds_zero.append(z); preds_all.append(p); truths.append(ch.mask)
    r0 = score_bs_micro(truths, preds_zero); r1 = score_bs_micro(truths, preds_all)
    print(f'{label:8s} ноль под маской: {r0["iou_burn"]:.4f}/{r0["miou_sev"]:.4f}   везде: {r1["iou_burn"]:.4f}/{r1["miou_sev"]:.4f} '
          f'кл1 {r1["per_class"][1]:.3f}  ({time.time()-t0:.0f}с)', flush=True)
