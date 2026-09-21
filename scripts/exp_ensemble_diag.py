"""Бриф №3 / ресёч, Stage 1: калибровка (ECE) и разнообразие (double-fault, Q) членов смеси на 144 групповых чипах;
плюс отрицательные веса в смеси (Stage 2.4) на двух шкалах."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def ece(p, t, ok, bins=15):
    """бинарная ECE по чипу: p — вероятность гари, t — истина гари, ok — чистое небо"""
    p, t = p[ok], t[ok]; e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any(): e += m.mean() * abs(p[m].mean() - t[m].mean())
    return e
MEMBERS = {'бустинг SWIR': ('own', 'bs-confirm-boost-f{f}-swir-v1'), 'оптика (v21)': ('hyp', 'bs-confirm-optical-f{f}-v1'),
           'сиам over (v21)': ('own', 'bs-confirm-siam-f{f}-over-v1'), 'сиам over2 (v21)': ('own', 'bs-confirm-siam-f{f}-over2-v1'),
           'сиам база s2': ('own', 'bs-confirm-siam-f{f}-s2-v1'),
           'оптика +SWIR (47, откл.)': ('own', 'bs-confirm-optical-f{f}-swir-v1'), 'оптика +карта буст. (48, откл.)': ('own', 'bs-confirm-optical-f{f}-bch-v1'),
           'оптика +лок.контраст (49, откл.)': ('own', 'bs-confirm-optical-f{f}-lz-v1'), 'сиам diff (51.1, откл.)': ('own', 'bs-confirm-siam-f{f}-diff-v1'),
           'сиам 2-этап (51.2, откл.)': ('own', 'bs-confirm-siam-f{f}-2st-v1'), 'сиам мягкая кромка (52)': ('own', 'bs-confirm-siam-f{f}-soft-v1')}
stats = {k: dict(ece=[], sharp=[], err=[], acc=[]) for k in MEMBERS}; errs = {k: [] for k in MEMBERS}; truths = []; oks = []
for f in range(5):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips]) > 0; OK = np.stack([c.valid() for c in chips]); truths.append(T); oks.append(OK)
    for k, (src, pat) in MEMBERS.items():
        q = (HYP if src == 'hyp' else OWN) / 'research' / pat.format(f=f) / 'probabilities.npy' if src == 'hyp' else OWN / pat.format(f=f) / 'probabilities.npy'
        if not q.exists(): errs[k].append(None); continue
        P = np.load(q).astype(np.float32); pb = 1 - P[..., 0]
        for i in range(len(chips)):
            if OK[i].sum() < 100: continue
            stats[k]['ece'].append(ece(pb[i], T[i], OK[i])); stats[k]['sharp'].append(np.abs(pb[i][OK[i]] - 0.5).mean() * 2)
        errs[k].append((pb > 0.5) != T)
print(f'{"член":34s} {"ECE(чип, 15 бинов)":>18s} {"резкость":>9s} | {"double-fault с бустингом":>24s} {"Q с бустингом":>13s} | {"DF с оптикой v21":>16s} {"DF с сиамом over":>16s}')
def pair(ea, eb):
    df, qs = [], []
    for A, B, OK in zip(ea, eb, oks):
        if A is None or B is None: continue
        for a, b, ok in zip(A, B, OK):
            if ok.sum() < 100: continue
            a, b = a[ok], b[ok]; dd = (a & b).mean(); aa = (~a & ~b).mean(); bb = (~a & b).mean(); cc = (a & ~b).mean()
            df.append(dd); den = aa * dd + bb * cc; qs.append((aa * dd - bb * cc) / den if den > 0 else 0)
    return (np.mean(df), np.mean(qs)) if df else (np.nan, np.nan)
for k in MEMBERS:
    if all(e is None for e in errs[k]): continue
    n = sum(e is not None for e in errs[k]); dfb, qb = pair(errs[k], errs['бустинг SWIR']); dfo, _ = pair(errs[k], errs['оптика (v21)']); dfs, _ = pair(errs[k], errs['сиам over (v21)'])
    print(f'{k:34s} {np.mean(stats[k]["ece"]):18.4f} {np.mean(stats[k]["sharp"]):9.3f} | {dfb*100:23.2f}% {qb:13.3f} | {dfo*100:15.2f}% {dfs*100:15.2f}%   ({n} ф.)')
import sys as _s; _s.exit(0)
# --- отрицательные веса
print('\nОтрицательные веса (рецепт v21, бустинг 0.4 фикс.; сети: (1−s)·оптика + s·сиам):')
def recipe(PB, PO, PS, OK, ZERO, wb, s):
    pn = (1 - s) * PO + s * PS; P = wb * PB + (1 - wb) * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, (PO.argmax(3) > 0) & (PS.argmax(3) > 0))])
def load_fold(f):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PS = np.mean([np.load(OWN / f'bs-confirm-siam-f{f}-{t}-v1/probabilities.npy').astype(np.float32) for t in ('over', 'over2')], 0)
    return chips, np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32), np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32), PS
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    return [d.load(c) for c in tune], np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32), PO, np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
def measure(sets, wb, s):
    T, O = [], []
    for chips, PB, PO, PS in sets:
        OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips]); T.append(np.stack([c.mask for c in chips]).reshape(-1)); O.append(recipe(PB, PO, PS, OK, ZERO, wb, s).reshape(-1))
    return w(score_bs(np.concatenate(T), np.concatenate(O)))
folds = [load_fold(f) for f in range(5)]; s35 = [load_35()]
b_f = measure(folds, 0.4, 0.5); b_35 = measure(s35, 0.4, 0.5)
for wb in (0.4, 0.5, 0.6):
    for s in (-0.3, -0.15, 0.5, 1.15, 1.3):
        print(f'  wb={wb:.1f} s={s:+.2f}: фолды {measure(folds, wb, s)-b_f:+.4f}  35 чипов {measure(s35, wb, s)-b_35:+.4f}')
