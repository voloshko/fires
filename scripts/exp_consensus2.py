"""SPEC-39, продолжение: варианты якоря согласия на кэшах пяти групповых фолдов.

Протокол против подгонки: вариант выбирается по пулу фолдов 0–2, проверяется
на фолдах 3–4. Всё считается по сохранённым вероятностям — сети не учатся.
"""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import label
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far, FAR_PX
HYP = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / 'fires-hypotheses'))
d = BsDataset('data/comp/train/bs')

def drop_far_soft(pred, weight, far_px=FAR_PX):
    """Как drop_far, но главное пятно — с наибольшей суммой `weight` (мягкое согласие)."""
    from scipy.ndimage import distance_transform_edt
    marks, count = label(pred > 0)
    if count < 2: return pred
    mass = np.bincount(marks.reshape(-1), weights=weight.reshape(-1), minlength=count + 1); mass[0] = 0
    sizes = np.bincount(marks.reshape(-1), minlength=count + 1); sizes[0] = 0
    main = mass.argmax() if mass.max() > 0 else sizes.argmax()
    dist = distance_transform_edt(marks != main)
    far = [k for k in range(1, count + 1) if k != main and dist[marks == k].min() > far_px]
    out = pred.copy(); out[np.isin(marks, far)] = 0; return out

def largest(mask):
    marks, count = label(mask)
    if not count: return mask
    sizes = np.bincount(marks.reshape(-1)); sizes[0] = 0
    return marks == sizes.argmax()

VARIANTS = {
    'v17 без согласия': lambda po, ps: ('none', None),
    'v18: argmax обеих': lambda po, ps: ('hard', (po.argmax(2) > 0) & (ps.argmax(2) > 0)),
    'min p_burn > 0.3': lambda po, ps: ('hard', np.minimum(1 - po[..., 0], 1 - ps[..., 0]) > 0.3),
    'min p_burn > 0.7': lambda po, ps: ('hard', np.minimum(1 - po[..., 0], 1 - ps[..., 0]) > 0.7),
    'мягкое: масса min p_burn': lambda po, ps: ('soft', np.minimum(1 - po[..., 0], 1 - ps[..., 0])),
    'мягкое: масса произведения': lambda po, ps: ('soft', (1 - po[..., 0]) * (1 - ps[..., 0])),
    'argmax обеих, иначе крупнейшие пятна обеих': lambda po, ps: ('hard', ((po.argmax(2) > 0) & (ps.argmax(2) > 0)) if ((po.argmax(2) > 0) & (ps.argmax(2) > 0)).any() else (largest(po.argmax(2) > 0) | largest(ps.argmax(2) > 0))),
}
acc = {k: {'sel': ([], []), 'chk': ([], [])} for k in VARIANTS}
for f in range(5):
    dirs = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(dirs['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    PB, PO, PS = (np.load(dirs[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    pn = 0.5 * PO + 0.5 * PS; P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    base = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); base[ZERO] = 0
    for name, fn in VARIANTS.items():
        outs = []
        for i in range(len(ids)):
            kind, a = fn(PO[i], PS[i])
            outs.append(drop_far(base[i]) if kind == 'none' else drop_far(base[i], anchor=a) if kind == 'hard' else drop_far_soft(base[i], a))
        part = 'sel' if f <= 2 else 'chk'
        acc[name][part][0].append(T); acc[name][part][1].append(np.stack(outs))
def w(Ts, Os):
    r = score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os]))
    return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
print(f'{"вариант":46s} {"выбор ф0-2":>10s} {"проверка ф3-4":>13s} {"пул 0-4":>8s}')
for name, a in acc.items():
    s, c = w(*a['sel']), w(*a['chk']); p = w(a['sel'][0] + a['chk'][0], a['sel'][1] + a['chk'][1])
    print(f'{name:46s} {s:10.4f} {c:13.4f} {p:8.4f}')
