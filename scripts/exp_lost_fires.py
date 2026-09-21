"""Потерянные пожары (IoU<0.3) на фолдах: слепота сети или постобработка? + порог гари."""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
BAD = {'BS_tr_000177','BS_tr_000207','BS_tr_000113','BS_tr_000170','BS_tr_000091','BS_tr_000218','BS_tr_000190','BS_tr_000145','BS_tr_000075','BS_tr_000179','BS_tr_000124','BS_tr_000167','BS_tr_000027','BS_tr_000140','BS_tr_000062','BS_tr_000205'}
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
TH = {'argmax (v19)': 0.0, 'p_burn>0.4': 0.1, 'p_burn>0.3': 0.2, 'p_burn>0.2': 0.3}
acc = {k: {'sel': ([], []), 'chk': ([], [])} for k in TH}
dn_ok, dn_bad = [], []
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB, PO, PS1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    PS = 0.5 * (PS1 + np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32))
    pn = 0.5 * PO + 0.5 * PS; P = 0.6 * pn + 0.4 * PB
    for i, c in enumerate(chips):
        OK = c.valid(); t = c.mask > 0; agree = (PO[i].argmax(2) > 0) & (PS[i].argmax(2) > 0)
        pre, post = c.pre.astype(np.float32), c.post.astype(np.float32)
        nbr = lambda s: (s[6] - s[8]) / (s[6] + s[8] + 1e-6)
        dnbr = nbr(pre) - nbr(post); (dn_bad if c.chip_id in BAD else dn_ok).append(float(np.median(dnbr[t])) if t.any() else np.nan)
        if c.chip_id in BAD:
            raw = pn[i].argmax(2) > 0; mixed = P[i].argmax(2) > 0; burn = mixed.copy(); burn[~OK] = raw[~OK]
            out = np.where(burn, 1, 0).astype(np.uint8); z = c.label_zero(); out2 = out.copy(); out2[z] = 0; out3 = drop_far(out2, anchor=agree)
            print(f'{c.chip_id} ф{f} истина {t.sum():6d} | сети {raw.sum():6d} (на истине {raw[t].mean():.2f}) бустинг {(PB[i].argmax(2)>0).sum():6d} (на истине {(PB[i].argmax(2)>0)[t].mean():.2f}) | смесь {out.sum():6d} → label_zero {out2.sum():6d} → far {out3.sum():6d} | p_burn сети на истине {1-pn[i][...,0][t].mean():.2f}, доля истины под label_zero {z[t].mean():.2f}, медиана dNBR истины {np.median(dnbr[t]):+.3f}, фона {np.median(dnbr[~t]):+.3f}')
        for name, dl in TH.items():
            burn = (1 - P[i][..., 0]) > 0.5 - dl if dl else P[i].argmax(2) > 0
            burn[~OK] = ((1 - pn[i][..., 0]) > 0.5 - dl)[~OK] if dl else (pn[i].argmax(2) > 0)[~OK]
            out = np.where(burn, P[i][..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[c.label_zero()] = 0; out = drop_far(out, anchor=agree)
            part = 'sel' if f <= 2 else 'chk'; acc[name][part][0].append(c.mask); acc[name][part][1].append(out)
print(f'медиана dNBR истины: потерянные {np.nanmedian(dn_bad):+.3f} (n={len(dn_bad)}), остальные {np.nanmedian(dn_ok):+.3f}; квартили остальных {np.round(np.nanpercentile(dn_ok,[25,75]),3)}')
def pooled(Ts, Os): return w(score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os])))
b = None
for name, a in acc.items():
    s, c = pooled(*a['sel']), pooled(*a['chk']); p = pooled(a['sel'][0] + a['chk'][0], a['sel'][1] + a['chk'][1]); b = b or (s, c, p)
    print(f'{name:16s} выбор {s:.4f} ({s-b[0]:+.4f}) проверка {c:.4f} ({c-b[1]:+.4f}) пул {p:.4f} ({p-b[2]:+.4f})')
