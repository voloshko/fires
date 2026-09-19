"""Ещё решающие правила из кэша (SPEC-19): медианное сглаживание (истина —
растр полигона, кромка гладкая), отдельные веса смеси для решения «гарь/фон»
и для степени, доля бустинга под маской 8/10."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import median_filter
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
PN = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
def sc(pred):
    r = score_bs_micro(list(T), list(pred)); return f"{r['iou_burn']:.4f}/{r['miou_sev']:.4f} [{r['per_class'][1]:.3f} {r['per_class'][2]:.3f} {r['per_class'][3]:.3f}] взв {(0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65:.4f}"
def rule(w_out=0.6, w_mask=1.0, w_sev=None):
    P = (1-w_out)*PB + w_out*PN; Pm = (1-w_mask)*PB + w_mask*PN; P[~OK] = Pm[~OK]
    burn = P.argmax(3) > 0
    if w_sev is None: sev = P[..., 1:].argmax(3) + 1
    else:
        S = (1-w_sev)*PB + w_sev*PN; sev = S[..., 1:].argmax(3) + 1
    out = np.where(burn, sev, 0).astype(np.uint8); out[ZERO] = 0; return out
base = rule(); print('база:', sc(base))
for w in (0.7, 0.8, 0.9): print(f'под маской вес сети {w}:', sc(rule(w_mask=w)))
for w in (0.4, 0.8, 1.0): print(f'степень по смеси с весом сети {w}:', sc(rule(w_sev=w)))
for k in (3, 5):
    q = np.stack([median_filter(b, k) for b in base]); q[ZERO] = 0; print(f'медиана {k}×{k} по карте классов:', sc(q))
    burn = np.stack([median_filter((b > 0).astype(np.uint8), k) for b in base]) > 0
    sev = np.where(base > 0, base, 0)
    from scipy.ndimage import distance_transform_edt as edt
    q = base.copy()
    for i in range(len(q)):
        add = burn[i] & (base[i] == 0)
        if add.any():
            idx = edt(base[i] == 0, return_distances=False, return_indices=True); q[i][add] = base[i][tuple(idx)][add]
        q[i][~burn[i]] = 0
    q[ZERO] = 0; print(f'медиана {k}×{k} по маске гари:', sc(q))
