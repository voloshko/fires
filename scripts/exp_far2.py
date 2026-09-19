"""Мягкий вариант фильтра чужих пожаров (SPEC-19): для компонент в зоне
неопределённости (по расстоянию до главного пятна) требовать более высокой
средней уверенности в гари, чем для ближних. Честно на половинах."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import label, distance_transform_edt as edt
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
PN = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
P = 0.4*PB + 0.6*PN; burn = P.argmax(3) > 0; burn[~OK] = (PN.argmax(3) > 0)[~OK]
base = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); base[ZERO] = 0
PBURN = 1 - P[..., 0]; PBURN[~OK] = (1 - PN[..., 0])[~OK]
comps = []   # (chip, mask, dist, conf, size)
for i in range(35):
    marks, n = label(base[i] > 0)
    if n < 2: continue
    sizes = np.bincount(marks.ravel()); sizes[0] = 0; main = sizes.argmax(); dist = edt(marks != main)
    for k in range(1, n + 1):
        if k == main: continue
        m = marks == k; comps.append((i, m, float(dist[m].min()), float(PBURN[i][m].mean()), int(m.sum())))
def apply(D_hard, D_soft, tau):
    out = base.copy()
    for i, m, dist, conf, size in comps:
        if dist > D_hard or (dist > D_soft and conf < tau): out[i][m] = 0
    return out
def sc(pred, idx=None):
    idx = range(35) if idx is None else idx
    q = score_bs_micro([T[i] for i in idx], [pred[i] for i in idx]); return (0.35*q['iou_burn']+0.30*q['miou_sev'])/0.65, q['iou_burn'], q['miou_sev']
print('жёсткий 125:', '%.4f (%.4f/%.4f)' % sc(apply(125, 9e9, 0)))
grid = [(Dh, Ds, tau) for Dh in (125, 150, 200, 9e9) for Ds in (50, 75, 100) for tau in (0.6, 0.7, 0.8, 0.9)]
res = sorted(((sc(apply(*g))[0], g) for g in grid), reverse=True)[:6]
for v, g in res: print(f'  Dh={g[0]:<5} Ds={g[1]:<4} tau={g[2]}: {v:.4f}')
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed); perm = rng.permutation(35); halves = (perm[:17], perm[17:])
    for fit_h, ev_h in (halves, halves[::-1]):
        best = max(grid, key=lambda g: sc(apply(*g), fit_h)[0])
        print(f'честно (сид {seed}): {best} → {sc(apply(*best), ev_h)[0]:.4f} против жёсткого 125: {sc(apply(125, 9e9, 0), ev_h)[0]:.4f}')
