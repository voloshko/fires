"""SPEC-42: переопределение фона уверенным бустингом по пикселю. Обе шкалы."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import label
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def big(mask, n):
    if n <= 0 or not mask.any(): return mask
    m, k = label(mask); sz = np.bincount(m.reshape(-1)); keep = np.flatnonzero(sz >= n); keep = keep[keep > 0]
    return np.isin(m, keep)
def predict(P, pn, PB, OK, ZERO, agree, tau=None, n=0):
    burn = P.argmax(2) > 0; burn[~OK] = (pn.argmax(2) > 0)[~OK]
    if tau is not None:
        over = (~burn) & ((1 - PB[..., 0]) > tau); burn = burn | big(over, n)
    out = np.where(burn, P[..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return drop_far(out, anchor=agree)
V = {'v19': {}}
for tau in (0.8, 0.9, 0.95):
    for n in (0, 500, 2000): V[f'τ={tau} N={n}'] = dict(tau=tau, n=n)
acc = {k: {'sel': ([], []), 'chk': ([], [])} for k in V}; lost = {k: 0 for k in V}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB, PO, PS1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    PS = 0.5 * (PS1 + np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32))
    pn = 0.5 * PO + 0.5 * PS; P = 0.6 * pn + 0.4 * PB
    for i, c in enumerate(chips):
        OK = c.valid(); ZERO = c.label_zero(); agree = (PO[i].argmax(2) > 0) & (PS[i].argmax(2) > 0); t = c.mask > 0
        for name, kw in V.items():
            out = predict(P[i], pn[i], PB[i], OK, ZERO, agree, **kw); p = out > 0
            lost[name] += ((t & p).sum() / max((t | p).sum(), 1)) < 0.3
            part = 'sel' if f <= 2 else 'chk'; acc[name][part][0].append(c.mask); acc[name][part][1].append(out)
def pooled(Ts, Os): return w(score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os])))
b = None; best = None
print('ГРУППОВЫЕ ФОЛДЫ (144 чипа)')
for name, a in acc.items():
    s, c = pooled(*a['sel']), pooled(*a['chk']); p = pooled(a['sel'][0] + a['chk'][0], a['sel'][1] + a['chk'][1]); b = b or (s, c, p)
    if name != 'v19' and (best is None or s > best[1]): best = (name, s)
    print(f'{name:14s} выбор {s:.4f} ({s-b[0]:+.4f}) проверка {c:.4f} ({c-b[1]:+.4f}) пул {p:.4f} ({p-b[2]:+.4f}) | потеряно {lost[name]:2d}')
print(f'лучший по выбору: {best[0]}')
# 35 настроечных чипов
z = np.load('models/tune_proba_19.npz'); PB35, T35, OK35 = z['pb'].astype(np.float32), z['t'], z['ok']
s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO35 = np.stack([c.label_zero() for c in chips])
OPT = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0)
SIAM = np.mean([np.load(HYP / f'research/{r}/probabilities.npy').astype(np.float32) for r in ('bs-siam-20260918-v3','bs-siam-20260919-v3')], 0)
pn = 0.5 * OPT + 0.5 * SIAM; P = 0.6 * pn + 0.4 * PB35; agree = (OPT.argmax(3) > 0) & (SIAM.argmax(3) > 0)
print('35 НАСТРОЕЧНЫХ ЧИПОВ')
for name in ('v19', best[0]):
    kw = V[name]; out = np.stack([predict(P[i], pn[i], PB35[i], OK35[i], ZERO35[i], agree[i], **kw) for i in range(35)])
    r = score_bs(T35.reshape(-1), out.reshape(-1)); print(f'{name:14s} {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f}')
