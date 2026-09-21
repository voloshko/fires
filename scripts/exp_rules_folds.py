"""Перепроверка решающих правил BS на пяти групповых фолдах (кэши, CPU).

Правила подбирались на 35 случайных чипах, делящих пожары с обучением. Здесь —
144 чипа, целые пожары отложены. Одна ось за раз от рецепта v19 (оптика 0.5 +
два сида сиама 0.5, бустинг 0.4, согласие как якорь). Выбор по фолдам 0–2,
проверка на 3–4, пул по всем.
"""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset, SCL_LABEL_ZERO
from src.comp.metric import score_bs
from src.comp.postproc import drop_far, drop_small
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65

def recipe(PB, PO, PS, OK, ZERO, boost=0.4, mask='net', far=125, blob=0, anchor=True):
    pn = 0.5 * PO + 0.5 * PS; P = (1 - boost) * pn + boost * PB
    burn = P.argmax(3) > 0
    if mask == 'net': burn[~OK] = (pn.argmax(3) > 0)[~OK]
    elif mask == 'zero': burn[~OK] = False
    # 'mixed': смесь решает и под маской
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    agree = (PO.argmax(3) > 0) & (PS.argmax(3) > 0)
    return np.stack([drop_far(drop_small(o, blob), far, anchor=a if anchor else None) for o, a in zip(out, agree)])

AXES = {
    'v19 (база)': {},
    'бустинг 0.0 (только сети)': dict(boost=0.0), 'бустинг 0.2': dict(boost=0.2), 'бустинг 0.6': dict(boost=0.6),
    'под маской: ноль': dict(mask='zero'), 'под маской: смесь': dict(mask='mixed'),
    'far_px 0 (выкл)': dict(far=0), 'far_px 75': dict(far=75), 'far_px 200': dict(far=200),
    'min_blob 100': dict(blob=100),
    'label_zero: нет': dict(zero='none'), 'label_zero без тени (3)': dict(zero=(0, 1, 9, 11)), 'label_zero + cirrus (10)': dict(zero=(0, 1, 3, 9, 10, 11)), 'label_zero + облако ср. (8)': dict(zero=(0, 1, 3, 8, 9, 11)),
}
acc = {k: {'sel': ([], []), 'chk': ([], [])} for k in AXES}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips])
    SCL = [(c.pre[9], c.post[9]) for c in chips]
    PB, PO, PS1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    PS = 0.5 * (PS1 + np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32))
    for name, kw in AXES.items():
        kw = dict(kw); zs = kw.pop('zero', SCL_LABEL_ZERO)
        ZERO = np.zeros_like(OK) if zs == 'none' else np.stack([np.isin(a, zs) | np.isin(b, zs) for a, b in SCL])
        out = recipe(PB, PO, PS, OK, ZERO, **kw)
        part = 'sel' if f <= 2 else 'chk'; acc[name][part][0].append(T); acc[name][part][1].append(out)
    print(f'фолд {f} готов', file=sys.stderr, flush=True)
def pooled(Ts, Os): return w(score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os])))
print(f'{"ось":32s} {"выбор ф0-2":>10s} {"проверка ф3-4":>13s} {"пул":>7s} {"Δ пул":>7s}')
b = None
for name, a in acc.items():
    s, c = pooled(*a['sel']), pooled(*a['chk']); p = pooled(a['sel'][0] + a['chk'][0], a['sel'][1] + a['chk'][1])
    if b is None: b = (s, c, p)
    print(f'{name:32s} {s:10.4f} {c:13.4f} {p:7.4f} {p-b[2]:+7.4f}   (Δ выбор {s-b[0]:+.4f}, Δ проверка {c-b[1]:+.4f})')
