"""Бледные гари: порог гари и чиповый переключатель на бустинг (фолды, кэши)."""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def variant(P, pn, PB, OK, ZERO, agree, th=0.5, gate=None, gate_min=2000):
    pb = 1 - P[..., 0]; pbn = 1 - pn[..., 0]
    burn = pb > th; burn[~OK] = (pbn > th)[~OK]
    if gate is not None:
        nb = (pn.argmax(2) > 0).sum(); bb = (PB.argmax(2) > 0)
        if bb.sum() >= gate_min and nb < gate * bb.sum():
            burn = bb.copy()
    out = np.where(burn, P[..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return drop_far(out, anchor=agree)
V = {'p_burn>0.5': {}, 'порог 0.44': dict(th=0.44), 'порог 0.42': dict(th=0.42), 'порог 0.40': dict(th=0.40), 'порог 0.38': dict(th=0.38), 'порог 0.36': dict(th=0.36)}
acc = {k: {'sel': ([], []), 'chk': ([], [])} for k in V}; fired = {k: [] for k in V}; lost = {k: 0 for k in V}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB, PO, PS1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    PS = 0.5 * (PS1 + np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32))
    pn = 0.5 * PO + 0.5 * PS; P = 0.6 * pn + 0.4 * PB
    for i, c in enumerate(chips):
        OK = c.valid(); ZERO = c.label_zero(); agree = (PO[i].argmax(2) > 0) & (PS[i].argmax(2) > 0); t = c.mask > 0
        for name, kw in V.items():
            out = variant(P[i], pn[i], PB[i], OK, ZERO, agree, **kw)
            if kw.get('gate') is not None:
                nb = (pn[i].argmax(2) > 0).sum(); bb = (PB[i].argmax(2) > 0).sum()
                if bb >= 2000 and nb < kw['gate'] * bb: fired[name].append(c.chip_id)
            p = out > 0; iou = (t & p).sum() / max((t | p).sum(), 1); lost[name] += iou < 0.3
            part = 'sel' if f <= 2 else 'chk'; acc[name][part][0].append(c.mask); acc[name][part][1].append(out)
def pooled(Ts, Os): return w(score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os])))
b = None
for name, a in acc.items():
    s, c = pooled(*a['sel']), pooled(*a['chk']); p = pooled(a['sel'][0] + a['chk'][0], a['sel'][1] + a['chk'][1]); b = b or (s, c, p)
    print(f'{name:32s} выбор {s:.4f} ({s-b[0]:+.4f}) проверка {c:.4f} ({c-b[1]:+.4f}) пул {p:.4f} ({p-b[2]:+.4f}) | потеряно чипов {lost[name]:2d}' + (f' | сработал на {len(fired[name])} чипах' if fired[name] else ''))
for k, v in fired.items():
    if v: print(f'  {k}: {v}')
