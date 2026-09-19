"""Псевдоразметка тестовых чипов (SPEC-19). Последний резерв из research-findings.
Метки для 89 тестовых чипов даёт ансамбль сетей, обученных на 144 чипах, и
бустинг на тех же 144 — чтобы замер на 35 настроечных остался честным (модели
на 224 видели настроечные чипы, их предсказания на тесте несли бы утечку).
Пишет data/comp/pseudo/<chip>.npy: uint8 маска и float16 уверенность."""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.model import train, sample_chip, SEED
from src.comp.unet import load as load_net, probs

NETS = sys.argv[1:] or [f'models/exp_{t}.pt' for t in ('d7w32','d7s1','d7s2','d7fast','d7rot')]
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
fit = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[35:])
t0 = time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
boost = train(np.concatenate(xs), np.concatenate(ys)); print(f'бустинг на {len(fit)} за {time.time()-t0:.0f}с', flush=True)
nets = [load_net(n) for n in NETS]
test = BsDataset('data/comp/test/bs'); out = Path('data/comp/pseudo'); out.mkdir(exist_ok=True)
for c in test.chip_ids():
    ch = test.load(c)
    pn = np.mean([probs(m, ch) for m in nets], 0)
    feats = np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32)
    pb = boost.predict_proba(feats.reshape(len(NAMES), -1).T).reshape(*ch.shape, 4)
    p = 0.6*pn + 0.4*pb; blind = ~ch.valid(); p[blind] = pn[blind]
    mask = p.argmax(2).astype(np.uint8); mask[ch.label_zero()] = 0
    np.save(out / f'{c}.npy', mask); np.save(out / f'{c}_conf.npy', p.max(2).astype(np.float16))
print(f'псевдоразметка {len(test)} чипов за {time.time()-t0:.0f}с → {out}')
