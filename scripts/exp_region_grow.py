"""SPEC-59: условное расширение гари — ядро (выход смеси) растёт в примыкающие компоненты бустинга,
если их средняя p гари бустинга > τ (и, вариант, p сиама > s). В отличие от SPEC-40 (один сигнал на
двух порогах) и SPEC-42 (переопределение по пикселю без связности) сигналы ядра и роста разные, а
решает связность. Выбор (τ, s) — фолды 0–2, проверка — фолды 3–4 и 35 чипов. Плюс аудит Stage 4:
доля ложных компонент по разнесению дат пары."""
import sys, json, hashlib, itertools, numpy as np, pandas as pd; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import label, binary_dilation
from src.comp.chips import BsDataset
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs'); META = d.meta.set_index('chip_id')
def conf(t, p): return np.bincount((t.astype(int) * 4 + p.astype(int)).ravel(), minlength=16).reshape(4, 4)
def metric(C):
    tot = C.sum(); burn = C[1:, 1:].sum() / max(tot - C[0, 0], 1); ious = []
    for k in (1, 2, 3):
        u = C[k].sum() + C[:, k].sum() - C[k, k]
        if u > 0: ious.append(C[k, k] / u)
    return (0.35 * burn + 0.30 * np.mean(ious)) / 0.65
def recipe(pb, po, ps, ok, zero):
    pn = 0.5 * po + 0.5 * ps; P = 0.4 * pb + 0.6 * pn; burn = P.argmax(2) > 0; burn[~ok] = (pn.argmax(2) > 0)[~ok]
    out = np.where(burn, P[..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[zero] = 0
    return drop_far(out, anchor=(po.argmax(2) > 0) & (ps.argmax(2) > 0))
def comps(chip, pb, ps, out):
    zero = chip.label_zero(); bb = (pb.argmax(2) > 0) & ~zero; lab, n = label(bb); near = binary_dilation(out > 0, iterations=2); T = chip.mask > 0; res = []
    for i in range(1, n + 1):
        m = lab == i; a = int(m.sum())
        if a < 20 or (m & (out > 0)).any(): continue
        res.append(dict(mask=m, adj=bool((m & near).any()), pb=float((1 - pb[..., 0])[m].mean()), ps=float((1 - ps[..., 0])[m].mean()), pos=T[m].mean() > 0.5, area=a, sev=np.where(m, pb[..., 1:].argmax(2) + 1, 0).astype(np.uint8)))
    return res
def load_fold(f):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']
    return ids, np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32), np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32), np.load(OWN / f'bs-confirm-siam-f{f}-sar-v1/probabilities.npy').astype(np.float32)
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    return tune, np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32), PO, np.load('research/bs-siam-sar-screen-20260918/probabilities.npy').astype(np.float32)
def table(ids, PB, PO, PS):
    rows = []
    for i, c in enumerate(ids):
        chip = d.load(c); out = recipe(PB[i], PO[i], PS[i], chip.valid(), chip.label_zero()); m = META.loc[c]
        gap = (pd.Timestamp(m.date_post) - pd.Timestamp(m.date_pre)).days if isinstance(m.date_post, str) else -1
        rows.append(dict(T=chip.mask, out=out, comps=comps(chip, PB[i], PS[i], out), gap=gap, lc=chip.aux[2]))
    return rows
def apply(rows, rule):
    C = np.zeros((4, 4), int); Cb = np.zeros((4, 4), int); lost = 0
    for r in rows:
        o = r['out'].copy()
        for k in r['comps']:
            if rule(k): o = np.where(k['mask'], k['sev'], o)
        C += conf(r['T'], o); Cb += conf(r['T'], r['out']); t = r['T'] > 0; p = o > 0; lost += (((t & p).sum() / max((t | p).sum(), 1)) < 0.3)
    return metric(C) - metric(Cb), int(lost), metric(Cb)
folds = [table(*load_fold(f)) for f in range(5)]; allr = sum(folds, []); r35 = table(*load_35())
allc = [k for r in allr for k in r['comps']]; pos = np.array([k['pos'] for k in allc]); adj = np.array([k['adj'] for k in allc])
print(f'компонент {len(allc)}, истинных {pos.sum()}; примыкают к ядру (≤ 2 пикс.): истинных {adj[pos].mean():.0%}, ложных {adj[~pos].mean():.0%}; среди примыкающих истинных {pos[adj].mean():.0%} (n={adj.sum()})')
base, lost0, _ = apply(allr, lambda k: False); o_adj, l_adj, _ = apply(allr, lambda k: k['pos'] and k['adj']); o_all, l_all, _ = apply(allr, lambda k: k['pos'])
print(f'фолды: рецепт потеряно {lost0} | оракул примыкающих {o_adj:+.4f}, потеряно {l_adj} | оракул всех истинных {o_all:+.4f}, потеряно {l_all} | принять все примыкающие {apply(allr, lambda k: k["adj"])[0]:+.4f}')
TAU = (0.5, 0.6, 0.7, 0.8, 0.9); S = (0.0, 0.01, 0.03, 0.05, 0.1)
sel = {(t, s): apply(sum(folds[:3], []), lambda k, t=t, s=s: k['adj'] and k['pb'] > t and k['ps'] > s)[0] for t, s in itertools.product(TAU, S)}
print('выбор на фолдах 0–2 (τ бустинга × s сиама), Δ взв:')
for t in TAU: print(f'  τ={t:.1f}: ' + '  '.join(f's={s:.2f} {sel[(t, s)]:+.4f}' for s in S))
best = max(sel, key=sel.get); t, s = best
chk, lchk, _ = apply(sum(folds[3:], []), lambda k: k['adj'] and k['pb'] > t and k['ps'] > s); pool, lpool, _ = apply(allr, lambda k: k['adj'] and k['pb'] > t and k['ps'] > s); g35, l35, b35 = apply(r35, lambda k: k['adj'] and k['pb'] > t and k['ps'] > s)
pf = [apply(folds[f], lambda k: k['adj'] and k['pb'] > t and k['ps'] > s)[0] for f in range(5)]
print(f'лучшее τ={t} s={s}: выбор {sel[best]:+.4f}, проверка ф3–4 {chk:+.4f}, пул 5 ф {pool:+.4f} (по фолдам {np.round(pf,4).tolist()}), потеряно {lost0} → {lpool} | 35 чипов {g35:+.4f}, потеряно {apply(r35, lambda k: False)[1]} → {l35}')
# без сиама (только связность + τ)
sel2 = {t: apply(sum(folds[:3], []), lambda k, t=t: k['adj'] and k['pb'] > t)[0] for t in TAU}; t2 = max(sel2, key=sel2.get)
chk2, _, _ = apply(sum(folds[3:], []), lambda k: k['adj'] and k['pb'] > t2); pool2, l2, _ = apply(allr, lambda k: k['adj'] and k['pb'] > t2); g352, l352, _ = apply(r35, lambda k: k['adj'] and k['pb'] > t2)
print(f'только связность + τ={t2}: выбор {sel2[t2]:+.4f}, проверка {chk2:+.4f}, пул {pool2:+.4f}, потеряно {l2} | 35 чипов {g352:+.4f}, потеряно {l352}')
ok = pool >= 0.004 and chk >= 0 and g35 >= 0
print('КРИТЕРИЙ SPEC-59:', 'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН')
# Stage 4: аудит по разнесению дат
print('\nаудит Stage 4: разнесение дат пары → компоненты (n, доля ложных, доля пашни среди ложных)')
for lo, hi in ((0, 14), (15, 30), (31, 60), (61, 400)):
    ks = [(k, r) for r in allr for k in r['comps'] if lo <= r['gap'] <= hi]
    if not ks: continue
    f = np.array([not k['pos'] for k, r in ks]); crop = np.array([np.bincount(r['lc'][k['mask']].astype(int), minlength=100)[40] / k['area'] > 0.5 for k, r in ks if not k['pos']])
    print(f'  {lo:3d}–{hi:3d} дней: n={len(ks):4d}, ложных {f.mean():.0%}, пашня среди ложных {crop.mean():.0%}, чипов {len({id(r) for k, r in ks})}')
print('распределение разнесения дат по 144 чипам:', np.percentile([r['gap'] for r in allr], [10, 50, 90]).round().tolist())
