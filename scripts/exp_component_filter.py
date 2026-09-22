"""SPEC-58: фильтр компонент бледной гари. Компоненты, которые видит только бустинг (нет пересечения
с выходом рецепта), принимаются или отбрасываются логистической регрессией на ≤ 3–5 признаках
компоненты; leave-one-fold-out по 5 групповым фолдам, затем обучение на всех → 35 чипов.
Рецепт — v22-аналог (оптика соседа, сиам с радаром, бустинг SWIR)."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import label, binary_dilation, binary_erosion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from src.comp.chips import BsDataset
from src.comp.features import stack
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
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
def _r(a, b):
    t = a + b; return np.divide(a - b, t, out=np.zeros_like(t), where=t != 0)
def components(chip, pb, po, ps, out):
    """компоненты бустинга без пересечения с выходом: признаки, маска, доля истины"""
    ok, zero = chip.valid(), chip.label_zero(); bb = (pb.argmax(2) > 0) & ~zero
    lab, n = label(bb); res = []
    dm = stack(chip, ('dmirbi', 'dndvi')); pre, post = chip.pre.astype(np.float32), chip.post.astype(np.float32)
    dndre = _r(pre[6], pre[3]) - _r(post[6], post[3])   # NDRE = (B8A − B5)/(B8A + B5), «до» минус «после»
    pburn_b, pburn_s = 1 - pb[..., 0], 1 - ps[..., 0]; T = chip.mask > 0
    for i in range(1, n + 1):
        m = lab == i; a = int(m.sum())
        if a < 20 or (m & (out > 0)).any(): continue
        ring = binary_dilation(m, iterations=5) & ~m & ok
        if ring.sum() < 20: continue
        per = (m & ~binary_erosion(m)).sum()
        x = [np.log(a), 4 * np.pi * a / per ** 2, dm[0][m].mean() - dm[0][ring].mean(), dm[1][m].mean() - dm[1][ring].mean(),
             dndre[m].mean() - dndre[ring].mean(), pburn_b[m].mean(), pburn_s[m].mean()]
        res.append(dict(x=x, mask=m, pos=T[m].mean() > 0.5, truth_frac=T[m].mean(), sev=np.where(m, pb[..., 1:].argmax(2) + 1, 0).astype(np.uint8)))
    return res
NOBAL = False
FEATS = {'A: dMIRBI-контраст, dNDVI-контраст, log площадь': [2, 3, 0], 'B: A + dNDRE-контраст': [2, 3, 4, 0], 'C: B + p бустинга, p сиама': [2, 3, 4, 0, 5, 6]}
def load_fold(f):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32)
    PS = np.load(OWN / f'bs-confirm-siam-f{f}-sar-v1/probabilities.npy').astype(np.float32)
    return ids, PB, PO, PS
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    return tune, np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32), PO, np.load('research/bs-siam-sar-screen-20260918/probabilities.npy').astype(np.float32)
def table(ids, PB, PO, PS):
    rows = []
    for i, c in enumerate(ids):
        chip = d.load(c); out = recipe(PB[i], PO[i], PS[i], chip.valid(), chip.label_zero())
        rows.append(dict(T=chip.mask, out=out, comps=components(chip, PB[i], PO[i], PS[i], out)))
    return rows
def apply(rows, decide):
    C = np.zeros((4, 4), int); lost = 0
    for r in rows:
        o = r['out'].copy()
        for k in r['comps']:
            if decide(r, k): o = np.where(k['mask'], k['sev'], o)
        C += conf(r['T'], o); t = r['T'] > 0; p = o > 0; lost += (((t & p).sum() / max((t | p).sum(), 1)) < 0.3)
    return metric(C), int(lost)
folds = [table(*load_fold(f)) for f in range(5)]; allc = [k for rows in folds for r in rows for k in r['comps']]
print(f'компонент «только бустинг» на 144 чипах: {len(allc)}, из них истинных (доля истины > 0.5): {sum(k["pos"] for k in allc)}; медиана площади {np.median([np.exp(k["x"][0]) for k in allc]):.0f} пикс.')
base, lost_b = apply(sum(folds, []), lambda r, k: False); orc, lost_o = apply(sum(folds, []), lambda r, k: k['pos']); alla, lost_a = apply(sum(folds, []), lambda r, k: True)
print(f'фолды: рецепт (v22-аналог) {base:.4f}, потеряно {lost_b} | оракул (все истинные компоненты) {orc:.4f} (+{orc-base:.4f}), потеряно {lost_o} | принять все {alla:.4f} ({alla-base:+.4f}), потеряно {lost_a}')
r35 = table(*load_35()); b35, l35 = apply(r35, lambda r, k: False); o35, _ = apply(r35, lambda r, k: k['pos'])
print(f'35 чипов: рецепт {b35:.4f}, потеряно {l35} | оракул {o35:.4f} (+{o35-b35:.4f}) | компонент {sum(len(r["comps"]) for r in r35)}, истинных {sum(k["pos"] for r in r35 for k in r["comps"])}')
def fit(rows_list, idx):
    X = [k['x'] for rows in rows_list for r in rows for k in r['comps']]; y = [k['pos'] for rows in rows_list for r in rows for k in r['comps']]
    if sum(y) < 3: return None
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight=(None if NOBAL else 'balanced'), max_iter=1000)).fit(np.array(X)[:, idx], y)
for name, idx in FEATS.items():
    pf = []; C = np.zeros((4, 4), int); lost = 0
    for f in range(5):
        m = fit([folds[g] for g in range(5) if g != f], idx)
        dec = (lambda r, k, m=m: bool(m.predict(np.array(k['x'])[idx][None])[0])) if m else (lambda r, k: False)
        mf, lf = apply(folds[f], dec); bf, _ = apply(folds[f], lambda r, k: False); pf.append(mf - bf)
        for r in folds[f]:
            o = r['out'].copy()
            for k in r['comps']:
                if dec(r, k): o = np.where(k['mask'], k['sev'], o)
            C += conf(r['T'], o); t = r['T'] > 0; p = o > 0; lost += (((t & p).sum() / max((t | p).sum(), 1)) < 0.3)
    m = fit(folds, idx); g35, l35g = apply(r35, lambda r, k, m=m: bool(m.predict(np.array(k['x'])[idx][None])[0]))
    print(f'{name:48s} фолды {metric(C)-base:+.4f} (по фолдам {np.round(pf,4).tolist()}), потеряно {lost} | 35 чипов {g35-b35:+.4f}, потеряно {l35g}')

# --- Ступень 2 (предзаявлена до запуска, после провала ступени 1): без balanced-весов; порог принятия τ
# выбирается по пулу фолдов 0–2, проверяется на фолдах 3–4 и на 35 чипах. Печатается AUC признаков (LOFO).
NOBAL = True
print('\nСтупень 2: порог по фолдам 0–2, проверка 3–4 и 35 чипов')
TAUS = np.arange(0.5, 0.96, 0.05)
for name, idx in FEATS.items():
    # AUC по LOFO-предсказаниям
    ys, ps = [], []
    for f in range(5):
        m = fit([folds[g] for g in range(5) if g != f], idx)
        for r in folds[f]:
            for k in r['comps']: ys.append(k['pos']); ps.append(m.predict_proba(np.array(k['x'])[idx][None])[0, 1])
    auc = roc_auc_score(ys, ps)
    def run(fold_ids, tau):
        C = np.zeros((4, 4), int); lost = 0; Cb = np.zeros((4, 4), int)
        for f in fold_ids:
            m = fit([folds[g] for g in range(5) if g != f], idx)
            for r in folds[f]:
                o = r['out'].copy()
                for k in r['comps']:
                    if m.predict_proba(np.array(k['x'])[idx][None])[0, 1] > tau: o = np.where(k['mask'], k['sev'], o)
                C += conf(r['T'], o); Cb += conf(r['T'], r['out']); t = r['T'] > 0; p = o > 0; lost += (((t & p).sum() / max((t | p).sum(), 1)) < 0.3)
        return metric(C) - metric(Cb), lost
    sel = {tau: run([0, 1, 2], tau)[0] for tau in TAUS}; tau = max(sel, key=sel.get)
    chk, lost_chk = run([3, 4], tau); allf, lost_all = run([0, 1, 2, 3, 4], tau)
    m = fit(folds, idx); g35, l35g = apply(r35, lambda r, k, m=m, tau=tau: m.predict_proba(np.array(k['x'])[idx][None])[0, 1] > tau)
    print(f'{name:48s} AUC {auc:.3f} | τ={tau:.2f}: выбор ф0–2 {sel[tau]:+.4f}, проверка ф3–4 {chk:+.4f}, пул 5 ф {allf:+.4f}, потеряно {lost_all} | 35 чипов {g35-b35:+.4f}, потеряно {l35g}')
