"""Ресёч №2, блок 4: асимметричная кромка — дилатация предсказания на 1–2 пикс. и
пониженный порог только у кромки. Фолды (выбор ф0–2 / проверка ф3–4) + 35 чипов, рецепт v20."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import binary_dilation, generate_binary_structure
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
ST = generate_binary_structure(2, 1)
def predict(PB, PO, PS, OK, ZERO, dil=0, edge_th=None):
    pn = 0.5 * PO + 0.5 * PS; P = 0.4 * PB + 0.6 * pn
    burn = P.argmax(2) > 0; burn[~OK] = (pn.argmax(2) > 0)[~OK]
    if edge_th is not None:   # пониженный порог только в кольце 2 пикс. вокруг предсказанной гари
        ring = binary_dilation(burn, ST, 2) & ~burn; burn = burn | (ring & ((1 - P[..., 0]) > edge_th))
    sev = (P[..., 1:].argmax(2) + 1).astype(np.uint8)
    out = np.where(burn, sev, 0).astype(np.uint8); out[ZERO] = 0
    out = drop_far(out, anchor=(PO.argmax(2) > 0) & (PS.argmax(2) > 0))
    if dil:   # дилатация после фильтра: новые пиксели получают степень ближайшей (по смеси), ноль под label_zero
        grown = binary_dilation(out > 0, ST, dil) & ~(out > 0) & ~ZERO; out = out.copy(); out[grown] = sev[grown]
    return out
V = {'v20': {}, 'дилатация 1 пикс': dict(dil=1), 'дилатация 2 пикс': dict(dil=2), 'порог у кромки 0.4': dict(edge_th=0.4), 'порог у кромки 0.3': dict(edge_th=0.3), 'порог у кромки 0.3 + дилатация 1': dict(edge_th=0.3, dil=1)}
acc = {k: {'sel': ([], []), 'chk': ([], [])} for k in V}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB, PO = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical')); SV = np.load(OWN / f'bs-confirm-siam-f{f}-over-v1/probabilities.npy').astype(np.float32)
    for i, c in enumerate(chips):
        OK = c.valid(); ZERO = c.label_zero()
        for name, kw in V.items():
            out = predict(PB[i], PO[i], SV[i], OK, ZERO, **kw); part = 'sel' if f <= 2 else 'chk'; acc[name][part][0].append(c.mask); acc[name][part][1].append(out)
def pooled(Ts, Os): return w(score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os])))
b = None
for name, a in acc.items():
    s, c = pooled(*a['sel']), pooled(*a['chk']); p = pooled(a['sel'][0] + a['chk'][0], a['sel'][1] + a['chk'][1]); b = b or (s, c, p)
    print(f'{name:34s} выбор {s:.4f} ({s-b[0]:+.4f}) проверка {c:.4f} ({c-b[1]:+.4f}) пул {p:.4f} ({p-b[2]:+.4f})')
z = np.load('models/tune_proba_19.npz'); PB35, T35, OK35 = z['pb'].astype(np.float32), z['t'], z['ok']
s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
PO35 = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0); OV35 = np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
b = None
for name, kw in V.items():
    out = np.stack([predict(PB35[i], PO35[i], OV35[i], OK35[i], chips[i].label_zero(), **kw) for i in range(35)]); r = score_bs(T35.reshape(-1), out.reshape(-1)); v = w(r); b = b or v
    print(f'35 чипов {name:34s} взв {v:.4f} ({v-b:+.4f})')
