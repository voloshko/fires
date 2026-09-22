"""SPEC-67: сколько даёт число сидов радарного сиама. Фолды: 1 / 2 / 3 сида (30, 40, 50) в рецепте v22;
35 чипов: 2 over-сида (v21) против 2 и 4 sar-сидов (18, 19, 20, 21)."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def rec(PB, PO, PS, OK, ZERO):
    pn = 0.5 * PO + 0.5 * PS; P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, (PO.argmax(3) > 0) & (PS.argmax(3) > 0))])
VAR = {'1 сид (v22)': ['sar'], '2 сида': ['sar', 'sar40'], '3 сида': ['sar', 'sar40', 'sar50']}; T_all, O = [], {k: [] for k in VAR}; F = {k: [] for k in VAR}; L = {k: 0 for k in VAR}
for f in range(5):
    if not all((OWN / f'bs-confirm-siam-f{f}-{t}-v1/summary.json').exists() for t in ('sar40', 'sar50')): continue
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]; T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips]); T_all.append(T)
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32)
    S = {t: np.load(OWN / f'bs-confirm-siam-f{f}-{t}-v1/probabilities.npy').astype(np.float32) for t in ('sar', 'sar40', 'sar50')}
    for k, ts in VAR.items():
        o = rec(PB, PO, np.mean([S[t] for t in ts], 0), OK, ZERO); O[k].append(o); F[k].append(w(score_bs(T.reshape(-1), o.reshape(-1))))
        for tt, oo in zip(T, o): L[k] += ((((tt > 0) & (oo > 0)).sum() / max(((tt > 0) | (oo > 0)).sum(), 1)) < 0.3)
if T_all:
    Tc = np.concatenate([t.reshape(-1) for t in T_all]); base = np.array(F['1 сид (v22)'])
    for k in VAR: r = w(score_bs(Tc, np.concatenate([o.reshape(-1) for o in O[k]]))); print(f'фолды ({len(T_all)} ф.) {k:14s} пул {r:.4f} | Δ к 1 сиду по фолдам {np.round(np.array(F[k])-base, 4).tolist()} | потеряно {L[k]}')
s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]; tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0); PB = np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32)
L35 = lambda r: np.load(f'research/{r}/probabilities.npy').astype(np.float32)
sets = {'over 2 сида (v21)': ['bs-siam-over-screen-20260918', 'bs-siam-over-screen-20260919'], 'sar 2 сида (v22)': ['bs-siam-sar-screen-20260918', 'bs-siam-sar-screen-20260919'], 'sar 4 сида': [f'bs-siam-sar-screen-2026091{k}' for k in (8, 9)] + [f'bs-siam-sar-screen-2026092{k}' for k in (0, 1)]}
for k, rs in sets.items():
    if not all(Path(f'research/{r}/summary.json').exists() for r in rs): print(f'35 чипов {k}: ещё нет'); continue
    o = rec(PB, PO, np.mean([L35(r) for r in rs], 0), OK, ZERO); r = score_bs(T.reshape(-1), o.reshape(-1)); p = o > 0; t = T > 0
    print(f'35 чипов {k:18s} взв {w(r):.4f} | чистое небо {(t&p&OK).sum()/((t|p)&OK).sum():.4f} | под маской {(t&p&~OK).sum()/max(((t|p)&~OK).sum(),1):.4f}')
