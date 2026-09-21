"""Сидовый ансамбль внутри передискретизованной сиамской ветви (фолды, кэши)."""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs, score_bs_micro
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def run(PB, PO, PS, chips):
    pn = 0.5 * PO + 0.5 * PS; P = 0.4 * PB + 0.6 * pn; OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    agree = (PO.argmax(3) > 0) & (PS.argmax(3) > 0)
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, agree)]), OK
POOL, FOLDS, LOST, CLR = {}, {}, {}, {}
for f in range(5):
    o2 = OWN / f'bs-confirm-siam-f{f}-over2-v1'
    if not (o2 / 'probabilities.npy').exists(): continue
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips])
    PB, PO = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical'))
    P1 = np.load(OWN / f'bs-confirm-siam-f{f}-over-v1/probabilities.npy').astype(np.float32); P2 = np.load(o2 / 'probabilities.npy').astype(np.float32)
    for name, ps in (('v20: передискр. сид A', P1), ('передискр. сид B один', P2), ('передискр. A+B (v21?)', 0.5 * (P1 + P2))):
        out, OK = run(PB, PO, ps, chips); FOLDS.setdefault(name, []).append(w(score_bs_micro(list(T), list(out))))
        e = POOL.setdefault(name, ([], [])); e[0].append(T); e[1].append(out)
        t = T > 0; p = out > 0; c = CLR.setdefault(name, [0, 0]); c[0] += (t & p & OK).sum(); c[1] += ((t | p) & OK).sum()
        for tt, o in zip(T, out):
            LOST[name] = LOST.get(name, 0) + ((((tt > 0) & (o > 0)).sum() / max(((tt > 0) | (o > 0)).sum(), 1)) < 0.3)
base = np.array(FOLDS['v20: передискр. сид A'])
for name, v in FOLDS.items():
    Ts, Os = POOL[name]; r = score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os]))
    print(f'{name:26s} пул {len(v)} ф: {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f} | чистое небо {CLR[name][0]/CLR[name][1]:.4f} | Δ к A {np.round(np.array(v)-base,4).tolist()} | потеряно {LOST[name]}')
