"""SPEC-56: гейт по чипу, вторая попытка (ресёч №3, дополнение, Stage 1–2 + Stage 4).

К 8 признакам SPEC-53 добавляются: статистики карты разногласия оптики и сиама и взаимная
информация членов (Stage 1a), новизна входа — Махаланобис сводки 25 признаков бустинга к
обучающим чипам фолда (1b), качество пары — дни между сценами, смещение dNBR негоревшего,
сдвиг освещения (1c); отдельно — плотность границы масок (Stage 2). Бутстрэп по чипам для
оракула и потолка степени (Stage 4)."""
import sys, json, hashlib, numpy as np, pandas as pd; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import binary_erosion, label
from sklearn.ensemble import HistGradientBoostingRegressor
from src.comp.chips import BsDataset
from src.comp.features import stack, NAMES_SWIR
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs'); META = d.meta.set_index('chip_id')
GRID = np.array([0.3, 0.4, 0.5, 0.6, 0.7]); WB = 0.4; rng = np.random.default_rng(56)
def conf(t, p):
    return np.bincount((t.astype(int) * 4 + p.astype(int)).ravel(), minlength=16).reshape(4, 4)
def metric(C):
    tot = C.sum(); burn = C[1:, 1:].sum() / max(tot - C[0, 0], 1); ious = []
    for k in (1, 2, 3):
        u = C[k].sum() + C[:, k].sum() - C[k, k]
        if u > 0: ious.append(C[k, k] / u)
    return (0.35 * burn + 0.30 * np.mean(ious)) / 0.65
def predict(pb, po, ps, ok, zero, s):
    pn = (1 - s) * po + s * ps; P = WB * pb + (1 - WB) * pn; burn = P.argmax(2) > 0; burn[~ok] = (pn.argmax(2) > 0)[~ok]
    out = np.where(burn, P[..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[zero] = 0
    return drop_far(out, anchor=(po.argmax(2) > 0) & (ps.argmax(2) > 0))
def H(p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(p * np.log(p) + (1 - p) * np.log(1 - p))
def shape_feats(m):
    a = m.sum()
    if a == 0: return [0.0, 0.0, 0.0]
    per = (m & ~binary_erosion(m)).sum(); n = label(m)[1]
    return [4 * np.pi * a / max(per, 1) ** 2, per / a, float(n)]
def summary_vec(chip):
    X = stack(chip, NAMES_SWIR); ok = chip.valid()
    if ok.sum() < 100: ok = np.ones_like(ok)
    v = X[:, ok][:, ::7]; return np.concatenate([np.median(v, 1), np.percentile(v, 90, 1) - np.percentile(v, 10, 1)])
SUMMARY = {}
def summ(c):
    if c not in SUMMARY: SUMMARY[c] = summary_vec(d.load(c))
    return SUMMARY[c]
def mahal(v, ref):
    R = np.stack(ref); mu = R.mean(0); S = np.cov(R.T) + 1e-3 * np.eye(len(mu)) * np.trace(np.cov(R.T)) / len(mu)
    dlt = v - mu; return float(np.sqrt(dlt @ np.linalg.solve(S, dlt)))
def features(c, chip, pb, po, ps, ps_each, fit_ids):
    ok = chip.valid(); bo, bs_ = po.argmax(2) > 0, ps.argmax(2) > 0; u = bo | bs_; dn = chip.dnbr(); pbb = pb.argmax(2) > 0
    base = [ok.mean(), bo.mean(), bs_.mean(), (bo & bs_).sum() / max(u.sum(), 1), pb[..., 1:].sum(2)[pbb].mean() if pbb.any() else 0,
            float(np.median(dn[u])) if u.any() else 0, po.max(2).mean(), ps.max(2).mean()]
    # 1a: карта разногласия и взаимная информация
    D = np.abs((1 - po[..., 0]) - (1 - ps[..., 0])); hi = D > 0.3
    members = [1 - po[..., 0]] + [1 - p[..., 0] for p in ps_each] + [1 - pb[..., 0]]; pm = np.mean(members, 0)
    MI = H(pm) - np.mean([H(m) for m in members], 0); anyb = u | pbb
    s1a = [D[ok].mean() if ok.any() else 0, hi[ok].mean() if ok.any() else 0, float(np.percentile(D[ok], 95)) if ok.any() else 0,
           (hi & anyb).sum() / max(hi.sum(), 1), MI[ok].mean() if ok.any() else 0, MI[anyb].mean() if anyb.any() else 0]
    # 1b: новизна входа
    s1b = [mahal(summ(c), [summ(x) for x in fit_ids])]
    # 1c: качество пары
    m = META.loc[c]; days = (pd.Timestamp(m.date_post) - pd.Timestamp(m.date_pre)).days if isinstance(m.date_post, str) else 0
    unb = ok & ~anyb; offset = float(np.median(dn[unb])) if unb.any() else 0
    pre, post = chip.pre.astype(np.float32), chip.post.astype(np.float32) if chip.post.size else chip.pre.astype(np.float32)
    stable = ok & (np.abs(dn) < 0.05); illum = float(np.log(post[:9][:, stable].mean() / max(pre[:9][:, stable].mean(), 1))) if stable.sum() > 100 else 0
    s1c = [float(days), offset, illum]
    s2 = shape_feats(bo) + shape_feats(bs_)
    return base, base + s1a + s1b + s1c, base + s1a + s1b + s1c + s2
def load_fold(f):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json')); ev, fit = ids['evaluation'], ids['fit']
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32)
    PSe = [np.load(OWN / f'bs-confirm-siam-f{f}-{t}-v1/probabilities.npy').astype(np.float32) for t in ('over', 'over2')]
    return ev, fit, PB, PO, PSe
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); fit = [c for c in ids if c not in tune]
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    return tune, fit, np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32), PO, [np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)]
def table(ev, fit, PB, PO, PSe):
    rows = []; PS = np.mean(PSe, 0)
    for i, c in enumerate(ev):
        chip = d.load(c); ok, zero = chip.valid(), chip.label_zero(); T = chip.mask
        confs = {s: conf(T, predict(PB[i], PO[i], PS[i], ok, zero, s)) for s in GRID}; per = np.array([metric(confs[s]) for s in GRID])
        best = 0.5 if np.ptp(per) < 1e-9 else float(GRID[per.argmax()])
        out = predict(PB[i], PO[i], PS[i], ok, zero, 0.5); ideal = np.where(out > 0, np.where(T > 0, T, out), 0)
        f8, f1, f2 = features(c, chip, PB[i], PO[i], PS[i], [p[i] for p in PSe], fit)
        rows.append(dict(x8=f8, x1=f1, x2=f2, confs=confs, best=best, ideal=conf(T, ideal)))
    return rows
def pooled(rows, choice): return metric(sum(r['confs'][s] for r, s in zip(rows, choice)))
def snap(v): return GRID[np.abs(GRID[None] - np.asarray(v)[:, None]).argmin(1)]
folds = [table(*load_fold(f)) for f in range(5)]; allrows = sum(folds, [])
fixed = pooled(allrows, [0.5] * len(allrows)); oracle = pooled(allrows, [r['best'] for r in allrows])
print(f'фолды: фикс. 0.5 {fixed:.4f} | оракул {oracle:.4f} (+{oracle-fixed:.4f})')
r35 = table(*load_35()); f35 = pooled(r35, [0.5] * 35)
def gate(key):
    ch, pf = [], []
    for f in range(5):
        tr = [r for g, rows in enumerate(folds) if g != f for r in rows]
        m = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05).fit([r[key] for r in tr], [r['best'] for r in tr])
        c = snap(m.predict([r[key] for r in folds[f]])); ch += list(c); pf.append(pooled(folds[f], c) - pooled(folds[f], [0.5] * len(folds[f])))
    m = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05).fit([r[key] for r in allrows], [r['best'] for r in allrows])
    g35 = pooled(r35, snap(m.predict([r[key] for r in r35]))) - f35
    return pooled(allrows, ch) - fixed, pf, g35
res = {}
for key, name in (('x8', 'SPEC-53: 8 признаков'), ('x1', 'Stage 1: +разногласие/MI, +новизна, +качество пары'), ('x2', 'Stage 1+2: +плотность границы')):
    g, pf, g35 = gate(key); res[key] = (g, g35); print(f'{name:52s} фолды {g:+.4f} (по фолдам {np.round(pf,4).tolist()}) | 35 чипов {g35:+.4f}')
# Stage 4: бутстрэп по чипам
B = 2000; n = len(allrows); dor, dsev = [], []
for _ in range(B):
    idx = rng.integers(0, n, n); rs = [allrows[i] for i in idx]
    base = sum(r['confs'][0.5] for r in rs); dor.append(metric(sum(r['confs'][r['best']] for r in rs)) - metric(base)); dsev.append(metric(sum(r['ideal'] for r in rs)) - metric(base))
print(f'бутстрэп по чипам ({B}): оракул гейта +{np.mean(dor):.4f} [{np.percentile(dor,2.5):.4f}, {np.percentile(dor,97.5):.4f}]; потолок степени внутри гари +{np.mean(dsev):.4f} [{np.percentile(dsev,2.5):.4f}, {np.percentile(dsev,97.5):.4f}]')
ok1 = res['x1'][0] >= 0.005 and res['x1'][1] >= 0; ok2 = res['x2'][0] - res['x1'][0] >= 0.003
print('КРИТЕРИЙ SPEC-56 Stage 1 (фолды ≥ +0.005 и 35 чипов ≥ 0):', 'ПРОЙДЕН' if ok1 else 'НЕ ПРОЙДЕН', '| Stage 2 (≥ +0.003 сверх Stage 1):', 'ПРОЙДЕН' if ok2 else 'НЕ ПРОЙДЕН')
