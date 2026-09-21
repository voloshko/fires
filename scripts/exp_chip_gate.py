"""SPEC-53: почиповый гейт доли сиама в смеси по признакам чипа без меток.

Оракул (лучшая доля на каждом чипе по истине) даёт потолок любого гейта; гейт —
HistGradientBoostingRegressor глубины 2 на 8 признаках, leave-one-fold-out по 5
групповым фолдам; затем гейт на всех 144 чипах применяется к 35 скрининговым."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
GRID = np.array([0.3, 0.4, 0.5, 0.6, 0.7]); WB = 0.4
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def predict(pb, po, ps, ok, zero, s):
    pn = (1 - s) * po + s * ps; P = WB * pb + (1 - WB) * pn; burn = P.argmax(2) > 0; burn[~ok] = (pn.argmax(2) > 0)[~ok]
    out = np.where(burn, P[..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[zero] = 0
    return drop_far(out, anchor=(po.argmax(2) > 0) & (ps.argmax(2) > 0))
def features(chip, pb, po, ps):
    ok = chip.valid(); bo, bs_ = po.argmax(2) > 0, ps.argmax(2) > 0; u = bo | bs_; dn = chip.dnbr()
    return [ok.mean(), bo.mean(), bs_.mean(), (bo & bs_).sum() / max(u.sum(), 1), pb[..., 1:].sum(2)[pb.argmax(2) > 0].mean() if (pb.argmax(2) > 0).any() else 0,
            float(np.median(dn[u])) if u.any() else 0, po.max(2).mean(), ps.max(2).mean()]
def load_fold(f):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(base['optical'] / 'probabilities.npy').astype(np.float32)
    PS = np.mean([np.load(OWN / f'bs-confirm-siam-f{f}-{t}-v1/probabilities.npy').astype(np.float32) for t in ('over', 'over2')], 0)
    return [d.load(c) for c in ids], PB, PO, PS
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    return [d.load(c) for c in tune], np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32), PO, np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
def table(chips, PB, PO, PS):
    """per chip: features, predictions for each s in GRID, truth."""
    rows = []
    for i, c in enumerate(chips):
        ok, zero = c.valid(), c.label_zero(); preds = {s: predict(PB[i], PO[i], PS[i], ok, zero, s) for s in GRID}
        per = np.array([w(score_bs(c.mask.reshape(-1), preds[s].reshape(-1))) for s in GRID]); per = np.nan_to_num(per)
        best = 0.5 if np.ptp(per) < 1e-9 else float(GRID[per.argmax()])
        rows.append(dict(x=features(c, PB[i], PO[i], PS[i]), preds=preds, truth=c.mask, best=best, per=per))
    return rows
def pooled(rows, choice): return w(score_bs(np.concatenate([r['truth'].reshape(-1) for r in rows]), np.concatenate([r['preds'][s].reshape(-1) for r, s in zip(rows, choice)])))
def snap(v): return GRID[np.abs(GRID[None] - np.asarray(v)[:, None]).argmin(1)]
folds = [table(*load_fold(f)) for f in range(5)]; allrows = sum(folds, [])
print('распределение лучшей доли сиама по 144 чипам:', {float(s): int(sum(r['best'] == s for r in allrows)) for s in GRID})
fixed = pooled(allrows, [0.5] * len(allrows)); oracle = pooled(allrows, [r['best'] for r in allrows))
print(f'фолды: фикс. 0.5 (v21) {fixed:.4f} | оракул по чипу {oracle:.4f} (+{oracle-fixed:.4f}) — потолок любого гейта')
gate_choice, per_fold = [], []
for f in range(5):
    tr = [r for g, rows in enumerate(folds) if g != f for r in rows]
    m = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05).fit([r['x'] for r in tr], [r['best'] for r in tr])
    ch = snap(m.predict([r['x'] for r in folds[f]])); gate_choice += list(ch)
    per_fold.append(pooled(folds[f], ch) - pooled(folds[f], [0.5] * len(folds[f])))
gate = pooled(allrows, gate_choice)
print(f'гейт LOFO: {gate:.4f} ({gate-fixed:+.4f}); по фолдам Δ {np.round(per_fold,4).tolist()}; выбранные доли {dict(zip(*np.unique(gate_choice, return_counts=True)))}')
m = HistGradientBoostingRegressor(max_depth=2, max_iter=100, learning_rate=0.05).fit([r['x'] for r in allrows], [r['best'] for r in allrows])
r35 = table(*load_35()); ch35 = snap(m.predict([r['x'] for r in r35]))
f35, g35, o35 = pooled(r35, [0.5] * 35), pooled(r35, ch35), pooled(r35, [r['best'] for r in r35])
print(f'35 чипов: фикс. 0.5 {f35:.4f} | гейт {g35:.4f} ({g35-f35:+.4f}) | оракул {o35:.4f}')
ok = (gate - fixed > 0.004) and (g35 - f35 >= 0)
print('КРИТЕРИЙ SPEC-53:', 'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН')
