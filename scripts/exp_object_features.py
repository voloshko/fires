"""SPEC-60: объектные признаки пятна не из спектра для компонент «только бустинг»: геометрия
(заполнение bbox, совпадение границы пятна с границами полей по NDVI «до»), текстура (разброс и
локальная неоднородность dNBR внутри пятна), радар по пятну (ΔVV, ΔVH, Δ(VH/VV) минус кольцо),
дата (день года «после», разнесение), плюс p сиама. Логистика и HistGB глубины 2; LOFO; порог по
фолдам 0–2, проверка 3–4 и 35 чипов; AUC каждого признака."""
import sys, json, hashlib, numpy as np, pandas as pd; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import label, binary_dilation, binary_erosion, sobel, uniform_filter, find_objects
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src.comp.chips import BsDataset
from src.comp.features import stack
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
def _r(a, b):
    t = a + b; return np.divide(a - b, t, out=np.zeros_like(t), where=t != 0)
NAMES = ['log площадь', 'заполнение bbox', 'граница на границе поля', 'std dNBR внутри', 'лок. неоднородность dNBR', 'ΔVV пятно−кольцо', 'ΔVH пятно−кольцо', 'Δ(VH/VV) пятно−кольцо', 'VH/VV после', 'день года', 'разнесение дней', 'p сиама', 'p бустинга', 'dMIRBI-контраст']
def components(c, chip, pb, ps, out):
    ok, zero = chip.valid(), chip.label_zero(); bb = (pb.argmax(2) > 0) & ~zero; lab, n = label(bb); T = chip.mask > 0
    pre, post = chip.pre.astype(np.float32), chip.post.astype(np.float32); ndvi_pre = _r(pre[6], pre[2]); dn = chip.dnbr(); dmirbi = stack(chip, ('dmirbi',))[0]
    edge = np.hypot(sobel(ndvi_pre, 0), sobel(ndvi_pre, 1)); edge = edge > np.percentile(edge[ok], 85) if ok.any() else edge > np.percentile(edge, 85); edge = binary_dilation(edge, iterations=1)
    lstd = np.sqrt(np.maximum(uniform_filter(dn * dn, 3) - uniform_filter(dn, 3) ** 2, 0))
    vv0, vh0 = chip.sar[0].astype(np.float32) / 1000, chip.sar[1].astype(np.float32) / 1000
    if chip.sar_post is not None: vv1, vh1 = chip.sar_post[0].astype(np.float32) / 1000, chip.sar_post[1].astype(np.float32) / 1000
    else: vv1, vh1 = vv0, vh0
    rat0, rat1 = np.divide(vh0, vv0, out=np.zeros_like(vv0), where=vv0 != 0), np.divide(vh1, vv1, out=np.zeros_like(vv1), where=vv1 != 0)
    m = META.loc[c]; doy = pd.Timestamp(m.date_post).dayofyear if isinstance(m.date_post, str) else 0; gap = (pd.Timestamp(m.date_post) - pd.Timestamp(m.date_pre)).days if isinstance(m.date_post, str) else 0
    res = []
    for i, sl in enumerate(find_objects(lab), 1):
        if sl is None: continue
        mm = lab == i; a = int(mm.sum())
        if a < 20 or (mm & (out > 0)).any(): continue
        ring = binary_dilation(mm, iterations=5) & ~mm & ok
        if ring.sum() < 20: continue
        bnd = mm & ~binary_erosion(mm); bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        x = [np.log(a), a / bbox, (bnd & edge).sum() / max(bnd.sum(), 1), float(dn[mm].std()), float(lstd[mm].mean()),
             float((vv1 - vv0)[mm].mean() - (vv1 - vv0)[ring].mean()), float((vh1 - vh0)[mm].mean() - (vh1 - vh0)[ring].mean()), float((rat1 - rat0)[mm].mean() - (rat1 - rat0)[ring].mean()), float(rat1[mm].mean()),
             float(doy), float(gap), float((1 - ps[..., 0])[mm].mean()), float((1 - pb[..., 0])[mm].mean()), float(dmirbi[mm].mean() - dmirbi[ring].mean())]
        res.append(dict(x=x, mask=mm, pos=T[mm].mean() > 0.5, sev=np.where(mm, pb[..., 1:].argmax(2) + 1, 0).astype(np.uint8)))
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
        chip = d.load(c); out = recipe(PB[i], PO[i], PS[i], chip.valid(), chip.label_zero()); rows.append(dict(T=chip.mask, out=out, comps=components(c, chip, PB[i], PS[i], out)))
    return rows
def apply(rows, rule):
    C = np.zeros((4, 4), int); Cb = np.zeros((4, 4), int); lost = 0
    for r in rows:
        o = r['out'].copy()
        for k in r['comps']:
            if rule(k): o = np.where(k['mask'], k['sev'], o)
        C += conf(r['T'], o); Cb += conf(r['T'], r['out']); t = r['T'] > 0; p = o > 0; lost += (((t & p).sum() / max((t | p).sum(), 1)) < 0.3)
    return metric(C) - metric(Cb), int(lost)
folds = [table(*load_fold(f)) for f in range(5)]; allr = sum(folds, []); r35 = table(*load_35())
allc = [k for r in allr for k in r['comps']]; X = np.array([k['x'] for k in allc]); y = np.array([k['pos'] for k in allc])
print(f'компонент {len(allc)}, истинных {y.sum()}; оракул {apply(allr, lambda k: k["pos"])[0]:+.4f}, потеряно {apply(allr, lambda k: False)[1]} → {apply(allr, lambda k: k["pos"])[1]}')
print('AUC признаков (все компоненты), медианы истинные | ложные:')
for j, n in enumerate(NAMES): print(f'  {n:26s} AUC {max(roc_auc_score(y, X[:, j]), 1-roc_auc_score(y, X[:, j])):.3f}  {np.median(X[y, j]):+.4f} | {np.median(X[~y, j]):+.4f}')
SETS = {'D: не из спектра (геометрия, текстура, радар, дата)': list(range(0, 11)), 'E: D + p сиама': list(range(0, 12)), 'F: всё (E + p бустинга + dMIRBI)': list(range(0, 14))}
MODELS = {'логистика': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)), 'HistGB d2': lambda: HistGradientBoostingClassifier(max_depth=2, max_iter=150, learning_rate=0.05)}
TAUS = np.arange(0.3, 0.96, 0.05)
def fit(rows_list, idx, mk):
    Xs = np.array([k['x'] for rows in rows_list for r in rows for k in r['comps']])[:, idx]; ys = [k['pos'] for rows in rows_list for r in rows for k in r['comps']]
    return mk().fit(Xs, ys)
for sname, idx in SETS.items():
    for mname, mk in MODELS.items():
        models = [fit([folds[g] for g in range(5) if g != f], idx, mk) for f in range(5)]
        ys, ps = [], []
        for f in range(5):
            for r in folds[f]:
                for k in r['comps']: ys.append(k['pos']); ps.append(models[f].predict_proba(np.array(k['x'])[idx][None])[0, 1])
        auc = roc_auc_score(ys, ps)
        def run(fids, tau):
            C = np.zeros((4, 4), int); Cb = np.zeros((4, 4), int); lost = 0
            for f in fids:
                for r in folds[f]:
                    o = r['out'].copy()
                    for k in r['comps']:
                        if models[f].predict_proba(np.array(k['x'])[idx][None])[0, 1] > tau: o = np.where(k['mask'], k['sev'], o)
                    C += conf(r['T'], o); Cb += conf(r['T'], r['out']); t = r['T'] > 0; p = o > 0; lost += (((t & p).sum() / max((t | p).sum(), 1)) < 0.3)
            return metric(C) - metric(Cb), lost
        sel = {tau: run([0, 1, 2], tau)[0] for tau in TAUS}; tau = max(sel, key=sel.get); chk, _ = run([3, 4], tau); pool, lost = run(range(5), tau)
        mall = fit(folds, idx, mk); g35, l35 = apply(r35, lambda k, m=mall, tau=tau: m.predict_proba(np.array(k['x'])[idx][None])[0, 1] > tau)
        ok = pool >= 0.004 and chk >= 0 and g35 >= 0
        print(f'{sname:52s} {mname:10s} AUC {auc:.3f} | τ={tau:.2f}: выбор {sel[tau]:+.4f}, проверка {chk:+.4f}, пул {pool:+.4f}, потеряно {lost} | 35 чипов {g35:+.4f}, потеряно {l35} | {"ПРОЙДЕН" if ok else "нет"}')
