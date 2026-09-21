"""SPEC-51: сиам с разностной фьюжн (diff) и двухэтапной потерей (2st) против передискр. сида, рецепт v21."""
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
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, (PO.argmax(3) > 0) & (PS.argmax(3) > 0))]), OK
POOL, FOLDS, LOST, CLR, ALONE = {}, {}, {}, {}, {}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]; T = np.stack([c.mask for c in chips])
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(base['optical'] / 'probabilities.npy').astype(np.float32)
    variants = [('v21: сиам передискр. (over)', np.load(OWN / f'bs-confirm-siam-f{f}-over-v1/probabilities.npy').astype(np.float32))]
    for tag, name in (('diff', 'сиам diff-фьюжн (SPEC-51.1)'), ('2st', 'сиам двухэтапная потеря (SPEC-51.2)'), ('soft', 'сиам мягкие метки у кромки (SPEC-52)')):
        q = OWN / f'bs-confirm-siam-f{f}-{tag}-v1/probabilities.npy'
        if q.exists(): variants.append((name, np.load(q).astype(np.float32)))
    for name, ps in variants:
        ao = np.where(ps.argmax(3) > 0, ps[..., 1:].argmax(3) + 1, 0); e = ALONE.setdefault(name, ([], [])); e[0].append(T); e[1].append(ao)
        out, OK = run(PB, PO, ps, chips); FOLDS.setdefault(name, []).append(w(score_bs_micro(list(T), list(out))))
        e = POOL.setdefault(name, ([], [])); e[0].append(T); e[1].append(out)
        t = T > 0; p = out > 0; c = CLR.setdefault(name, [0, 0]); c[0] += (t & p & OK).sum(); c[1] += ((t | p) & OK).sum()
        for tt, o in zip(T, out): LOST[name] = LOST.get(name, 0) + ((((tt > 0) & (o > 0)).sum() / max(((tt > 0) | (o > 0)).sum(), 1)) < 0.3)
def pooled(e): return score_bs(np.concatenate([t.reshape(-1) for t in e[0]]), np.concatenate([o.reshape(-1) for o in e[1]]))
base = np.array(FOLDS['v21: сиам передискр. (over)'])
for name, v in FOLDS.items():
    r = pooled(POOL[name]); ra = pooled(ALONE[name]); n = len(v)
    print(f'{name:40s} сеть одна {w(ra):.4f} | рецепт пул {n} ф: {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f} | чистое небо {CLR[name][0]/CLR[name][1]:.4f} | Δ {np.round(np.array(v)-base[:n],4).tolist()} | потеряно {LOST[name]}')
