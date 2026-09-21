"""SPEC-50: перебор весов смеси после SWIR-бустинга на двух шкалах (5 групповых фолдов и 35 чипов).

Сетка предзаявлена в спеке: вес бустинга wb, доля сиама s внутри сетей, среднее сетей
арифметическое или геометрическое (логит). Выбор — по пулу фолдов 0–2, проверка — фолды 3–4
и 35 чипов; принимается только минимакс по обеим шкалам."""
import sys, json, hashlib, itertools, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
WB = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55); S = (0.4, 0.5, 0.6, 0.7); GEO = (False, True)
V21 = (0.4, 0.5, False)
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def nets(PO, PS, s, geo):
    if not geo: return (1 - s) * PO + s * PS
    g = np.exp((1 - s) * np.log(PO + 1e-4) + s * np.log(PS + 1e-4)); return g / g.sum(3, keepdims=True)
def recipe(PB, PO, PS, OK, ZERO, anchor, wb, s, geo):
    pn = nets(PO, PS, s, geo); P = wb * PB + (1 - wb) * pn
    burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, anchor)])
def load_fold(f):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32)
    PO = np.load(base['optical'] / 'probabilities.npy').astype(np.float32)
    PS = np.mean([np.load(OWN / f'bs-confirm-siam-f{f}-{t}-v1/probabilities.npy').astype(np.float32) for t in ('over', 'over2')], 0)
    return chips, PB, PO, PS
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
    assert json.load(open('research/bs-boost-swir-screen-v1/data_manifest.json'))['evaluation'] == tune
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    PS = np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
    PB = np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32)
    return chips, PB, PO, PS
def measure(sets, combo):
    """sets: list of (chips, PB, PO, PS); returns pooled weighted, clear-sky IoU, lost chips."""
    T, O, ok_i, ok_u, lost = [], [], 0, 0, 0
    for chips, PB, PO, PS in sets:
        OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips]); truth = np.stack([c.mask for c in chips])
        anchor = (PO.argmax(3) > 0) & (PS.argmax(3) > 0)
        out = recipe(PB, PO, PS, OK, ZERO, anchor, *combo); T.append(truth.reshape(-1)); O.append(out.reshape(-1))
        t = truth > 0; p = out > 0; ok_i += (t & p & OK).sum(); ok_u += ((t | p) & OK).sum()
        for tt, o in zip(truth, out): lost += ((((tt > 0) & (o > 0)).sum() / max(((tt > 0) | (o > 0)).sum(), 1)) < 0.3)
    return w(score_bs(np.concatenate(T), np.concatenate(O))), ok_i / ok_u, int(lost)
folds = [load_fold(f) for f in range(5)]; s35 = [load_35()]
combos = list(itertools.product(WB, S, GEO)); rows = {}
for c in combos:
    sel = measure(folds[:3], c)[0]; chk = measure(folds[3:], c)[0]; pool, clr, lost = measure(folds, c); m35, clr35, _ = measure(s35, c)
    rows[c] = (sel, chk, pool, clr, lost, m35, clr35)
b = rows[V21]
print(f'{"wb":>5} {"s":>4} {"geo":>4} | {"выбор ф0-2":>10} {"провер ф3-4":>11} {"пул 5ф":>7} {"чист":>6} {"потер":>5} | {"35 чипов":>8} {"чист35":>6}')
for c in sorted(combos, key=lambda c: -rows[c][0]):
    r = rows[c]; tag = ' <- v21' if c == V21 else ''
    print(f'{c[0]:5.2f} {c[1]:4.1f} {"лог" if c[2] else "ариф":>4} | {r[0]-b[0]:+10.4f} {r[1]-b[1]:+11.4f} {r[2]:7.4f} {r[3]:6.4f} {r[4]:5d} | {r[5]-b[5]:+8.4f} {r[6]:6.4f}{tag}')
best = max(combos, key=lambda c: rows[c][0]); r = rows[best]
print(f'\nv21 (0.40, 0.5, ариф): пул {b[2]:.4f}, чистое небо {b[3]:.4f}, потеряно {b[4]}, 35 чипов {b[5]:.4f}')
print(f'лучший по выбору ф0-2: wb={best[0]} s={best[1]} {"лог" if best[2] else "ариф"}: выбор {r[0]-b[0]:+.4f}, проверка {r[1]-b[1]:+.4f}, пул 5ф {r[2]:.4f} ({r[2]-b[2]:+.4f}), 35 чипов {r[5]-b[5]:+.4f}')
ok = (r[2] - b[2] > 0.004) and (r[1] - b[1] >= 0) and (r[5] - b[5] >= 0)
print('КРИТЕРИЙ SPEC-50:', 'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН')
json.dump({str(k): v for k, v in rows.items()}, open('research/mix_weights_spec50.json', 'w'), indent=1, default=float)
