"""Три гипотезы из кэша (SPEC-19):
(а) nodata вспомогательных слоёв как обрез разметки — чип 000016 держит 24 %
    ложной гари, а истина там кончается прямой линией;
(б) калибровка весов классов степени на честном сплите половин настроечных чипов;
(в) стекинг: бустинг второго уровня поверх вероятностей сети и бустинга, их
    окон и признаков, обучение на одной половине чипов, замер на другой, и наоборот."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import uniform_filter
from src.comp.chips import BsDataset
from src.comp.features import stack, NAMES_BASE
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
PN = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO = np.stack([c.label_zero() for c in chips])
P = 0.4*PB + 0.6*PN; P[~OK] = PN[~OK]
base = P.argmax(3).astype(np.uint8); base[ZERO] = 0
t = T > 0; fp = (base > 0) & ~t
def sc(pred, idx=None):
    idx = range(len(T)) if idx is None else idx
    r = score_bs_micro([T[i] for i in idx], [pred[i] for i in idx]); return r['iou_burn'], r['miou_sev'], (0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65
def fmt(r): return f'{r[0]:.4f}/{r[1]:.4f} взв {r[2]:.4f}'
print('база (9 сетей):', fmt(sc(base)))
# (а) nodata aux
aux = np.stack([c.aux for c in chips])
for k, name in ((0, 'dem'), (1, 'slope'), (2, 'landcover')):
    for nod_val in (0, -9999, 255):
        m = aux[:, k] == nod_val
        if m.sum(): print(f'  aux {name}=={nod_val}: пикс {int(m.sum()):8d} гарь {int((t&m).sum()):7d} fp {int((fp&m).sum()):6d}')
i16 = tune.index('BS_tr_000016'); a = aux[i16]
print('  чип 16: уникальных landcover', np.unique(a[2])[:12], ' dem min/max', float(a[0].min()), float(a[0].max()), ' slope min', float(a[1].min()))
print('  чип 16: пикселей с dem<=0:', int((a[0] <= 0).sum()), ' fp там:', int((fp[i16] & (a[0] <= 0)).sum()), ' гарь там:', int((t[i16] & (a[0] <= 0)).sum()))
# (б) калибровка весов классов: pred = argmax(P * w), w=(1,w1,w2,w3); честно на половинах
rng = np.random.default_rng(0); perm = rng.permutation(35); halves = (perm[:17], perm[17:])
grid = [(w1, w2, w3) for w1 in (0.8, 1.0, 1.2, 1.5) for w2 in (0.8, 1.0, 1.2) for w3 in (0.8, 1.0, 1.2)]
def calib(w):
    q = (P * np.array([1.0, *w], np.float32)).argmax(3).astype(np.uint8); q[ZERO] = 0; return q
tot = []
for fit_h, ev_h in (halves, halves[::-1]):
    best = max(grid, key=lambda w: sc(calib(w), fit_h)[2])
    r_ev = sc(calib(best), ev_h); r_base = sc(base, ev_h); tot.append((r_ev[2] - r_base[2], best))
print(f'(б) калибровка классов, честно на половинах: прибавка {tot[0][0]:+.4f} (веса {tot[0][1]}) и {tot[1][0]:+.4f} (веса {tot[1][1]})')
# (в) стекинг
from sklearn.ensemble import HistGradientBoostingClassifier
feats_all = []
for i, c in enumerate(chips):
    f = stack(c, NAMES_BASE)
    pb_burn = 1 - PN[i, ..., 0]
    layers = [PN[i, ..., k] for k in range(4)] + [PB[i, ..., k] for k in range(4)] + [
        uniform_filter(pb_burn, 5, mode='nearest'), uniform_filter(pb_burn, 15, mode='nearest'), uniform_filter(pb_burn, 31, mode='nearest'),
        OK[i].astype(np.float32), f[NAMES_BASE.index('dnbr')], f[NAMES_BASE.index('landcover')], f[NAMES_BASE.index('dnbr_win15')]]
    feats_all.append(np.stack(layers).reshape(len(layers), -1).T)
def sample(idx, n=40000):
    xs, ys = [], []
    for i in idx:
        y = T[i].reshape(-1); keep = ~ZERO[i].reshape(-1)
        pick = []
        for cls in range(4):
            cand = np.flatnonzero((y == cls) & keep)
            if cand.size: pick.append(rng.choice(cand, min(cand.size, n // 4), replace=False))
        pick = np.concatenate(pick); xs.append(feats_all[i][pick]); ys.append(y[pick])
    return np.concatenate(xs), np.concatenate(ys)
gains = []
for fit_h, ev_h in (halves, halves[::-1]):
    x, y = sample(fit_h)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, categorical_features=[13], random_state=0).fit(x, y)
    pred = base.copy()
    for i in ev_h:
        q = m.predict(feats_all[i]).astype(np.uint8).reshape(T[i].shape); q[ZERO[i]] = 0; pred[i] = q
    r_ev = sc(pred, ev_h); r_base = sc(base, ev_h); gains.append(r_ev[2] - r_base[2])
    print(f'(в) стекинг: половина замера {fmt(r_ev)} против базы {fmt(r_base)}')
print(f'(в) стекинг, честно: прибавки {gains[0]:+.4f} и {gains[1]:+.4f}')
