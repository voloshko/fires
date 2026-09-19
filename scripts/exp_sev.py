"""Разные веса смеси для двух решений (SPEC-19): «гарь/фон» — где сеть сильнее
(форма пятна), «степень внутри гари» — где сильнее бустинг (спектр пикселя).
Под маской степень — только сеть (признаки бустинга там шум). Честность —
на половинах 17/18."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
PN = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
P = 0.4*PB + 0.6*PN; P[~OK] = PN[~OK]; BURN = P.argmax(3) > 0
def rule(w_sev):
    S = (1-w_sev)*PB + w_sev*PN; S[~OK] = PN[~OK]
    out = np.where(BURN, S[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; return out
def sc(pred, idx=None):
    idx = range(35) if idx is None else idx
    r = score_bs_micro([T[i] for i in idx], [pred[i] for i in idx]); return r['iou_burn'], r['miou_sev'], r['per_class'], (0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65
print('вес сети для степени  mIoU_sev   кл1    кл2    кл3   взвеш.')
for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    r = sc(rule(w)); print(f'{w:4.1f}                  {r[1]:.4f}   {r[2][1]:.3f}  {r[2][2]:.3f}  {r[2][3]:.3f}  {r[3]:.4f}')
rng = np.random.default_rng(0); perm = rng.permutation(35); halves = (perm[:17], perm[17:])
for fit_h, ev_h in (halves, halves[::-1]):
    best = max((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), key=lambda w: sc(rule(w), fit_h)[3])
    print(f'честно: подобрано {best} на одной половине → на другой {sc(rule(best), ev_h)[3]:.4f} против 0.6: {sc(rule(0.6), ev_h)[3]:.4f}')
