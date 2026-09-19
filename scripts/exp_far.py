"""Фильтр компонент по расстоянию до главного пятна (SPEC-19), честно на
половинах 17/18: порог подбирается на одной, меряется на другой."""
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
def drop_far(pred, D, n_main=1):
    out = pred.copy(); marks, n = label(pred > 0)
    if n <= n_main: return out
    sizes = np.bincount(marks.ravel()); sizes[0] = 0
    mains = np.argsort(sizes)[::-1][:n_main]
    dist = edt(~np.isin(marks, mains))
    far = [k for k in range(1, n + 1) if k not in mains and dist[marks == k].min() > D]
    out[np.isin(marks, far)] = 0; return out
def sc(pred, idx=None):
    idx = range(35) if idx is None else idx
    q = score_bs_micro([T[i] for i in idx], [pred[i] for i in idx]); return q['iou_burn'], q['miou_sev'], (0.35*q['iou_burn']+0.30*q['miou_sev'])/0.65
cache = {}
def pred_for(D, nm): 
    if (D, nm) not in cache: cache[(D, nm)] = np.stack([drop_far(b, D, nm) for b in base])
    return cache[(D, nm)]
print('порог D  главных  IoU_burn  mIoU_sev  взвеш.')
for nm in (1, 2):
    for D in (100, 125, 150, 175, 200, 250):
        r = sc(pred_for(D, nm)); print(f'{D:6d}  {nm:6d}   {r[0]:.4f}   {r[1]:.4f}   {r[2]:.4f}')
b = sc(base); print(f'база                {b[0]:.4f}   {b[1]:.4f}   {b[2]:.4f}')
grid = [(D, nm) for nm in (1, 2) for D in (100, 125, 150, 175, 200, 250)]
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed); perm = rng.permutation(35); halves = (perm[:17], perm[17:])
    for fit_h, ev_h in (halves, halves[::-1]):
        best = max(grid, key=lambda g: sc(pred_for(*g), fit_h)[2])
        print(f'честно (сид {seed}): подобрано D={best[0]}, главных {best[1]} → на другой половине {sc(pred_for(*best), ev_h)[2]:.4f} против базы {sc(base, ev_h)[2]:.4f}  ({sc(pred_for(*best), ev_h)[2]-sc(base, ev_h)[2]:+.4f})')
# по чипам: где фильтр D=150 помогает/вредит
q = pred_for(150, 1)
gain = []
for i in range(35):
    t = T[i] > 0; b0 = base[i] > 0; b1 = q[i] > 0
    gain.append(((t & b1).sum() - (t & b0).sum(), (~t & b0).sum() - (~t & b1).sum()))
gain = np.array(gain); print('чипов с потерей истины > 500 пикс:', int((gain[:, 0] < -500).sum()), ' | суммарно снято fp', int(gain[:, 1].sum()), ' потеряно tp', int(-gain[:, 0].sum()))
