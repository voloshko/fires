"""Ресёч №2, блок 2: логит-усреднение (геометрическое среднее) вместо арифметического.
Проверка на 5 групповых фолдах и на 35 чипах, в нашем рецепте. Гипотеза: «смелый»
передискретизованный сид перестанет терять прибавку в смеси с обычными."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def gmean(ps, wts=None):
    L = np.log(np.clip(np.stack(ps), 1e-4, 1)); wts = np.ones(len(ps)) / len(ps) if wts is None else np.asarray(wts) / np.sum(wts)
    g = np.exp(np.tensordot(wts, L, 1)); return g / g.sum(-1, keepdims=True)
def run(PB, pn, chips, boost_geo=False):
    P = gmean([pn, PB], [0.6, 0.4]) if boost_geo else 0.4 * PB + 0.6 * pn
    OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return out
POOL = {}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips])
    PB, PO, S1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    S2 = np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32); SV = np.load(OWN / f'bs-confirm-siam-f{f}-over-v1/probabilities.npy').astype(np.float32)
    V = {
        'v19 арифм.: опт + (S1+S2)/2': 0.5 * PO + 0.25 * (S1 + S2),
        'v19 геом.': gmean([PO, S1, S2], [0.5, 0.25, 0.25]),
        'v20 арифм.: опт + SV': 0.5 * PO + 0.5 * SV,
        'v20 геом.': gmean([PO, SV], [0.5, 0.5]),
        'опт + (S1+SV)/2 арифм.': 0.5 * PO + 0.25 * (S1 + SV),
        'опт + (S1+SV)/2 геом.': gmean([PO, S1, SV], [0.5, 0.25, 0.25]),
        'опт + (S1+S2+SV)/3 геом.': gmean([PO, S1, S2, SV], [0.5, 1/6, 1/6, 1/6]),
    }
    for name, pn in V.items():
        for bg in (False, True):
            out = run(PB, pn, chips, bg)
            agree_ps = pn  # якорь согласия: оптика и сиамская часть; для простоты — оптика и смесь сетей
            out = np.stack([drop_far(o, anchor=a) for o, a in zip(out, (PO.argmax(3) > 0) & (pn.argmax(3) > 0))])
            e = POOL.setdefault(name + (' | бустинг геом.' if bg else ''), ([], [])); e[0].append(T); e[1].append(out)
for name, (Ts, Os) in POOL.items():
    r = score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os]))
    print(f'{name:44s} пул 144: {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f}')
