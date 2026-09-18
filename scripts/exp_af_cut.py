"""AF: решающая граница по out-of-fold (SPEC-17/19).

Граница 0.95 выбиралась на val по сетке 0.05..0.95 — и попала в ПОСЛЕДНЮЮ точку
сетки. Значит, оптимум может лежать выше и не был виден. Здесь пятикратная
перекрёстная проверка по 336 чипам train+val, сетка продолжена до 0.999.
Отложенные 84 не трогаются.
"""
import sys, json, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.af import AfDataset, build_training_set, train, features, f1, NAMES, THRESHOLDS

d = AfDataset('data/comp/train/af'); sp = json.load(open('data/comp/split_af.json'))
chips = sorted(sp['train'] + sp['val']); K = 5
folds = [chips[i::K] for i in range(K)]
GRID = np.concatenate([np.arange(0.5, 0.95, 0.05), np.arange(0.95, 0.999, 0.005), [0.999]])
proba_all, truth_all, lc_all, empty_chip = [], [], [], []
t0 = time.time()
for k in range(K):
    fit = [c for j, f in enumerate(folds) if j != k for c in f]
    x, y = build_training_set(d, fit); m = train(x, y)
    for c in folds[k]:
        ch = d.load(c); ok = ch.valid().reshape(-1)
        p = m.predict_proba(features(ch).reshape(len(NAMES), -1).T)[:, 1]
        p[~ok] = 0; p[np.isin(ch.aux[0].reshape(-1), THRESHOLDS['exclude_landcover'])] = 0
        t = ch.mask.reshape(-1) > 0
        proba_all.append(p); truth_all.append(t); lc_all.append(ch.aux[0].reshape(-1))
        empty_chip.append(np.repeat(t.sum() == 0, t.size))
    print(f'фолд {k+1}/{K} за {time.time()-t0:.0f}с', flush=True)
P, T, LC, E = map(np.concatenate, (proba_all, truth_all, lc_all, empty_chip))
print(f'\nчипов {len(chips)}, пустых по истине {int(sum(e[0] for e in empty_chip))}, горящих пикселей {int(T.sum())}')
print('граница   F1      prec    rec     fp     fn   fp на пустых чипах')
best = (0, 0)
for c in GRID:
    pr = P >= c; r = f1(T, pr)
    print(f'{c:7.3f} {r["f1"]:7.4f} {r["precision"]:7.4f} {r["recall"]:7.4f} {r["fp"]:6d} {r["fn"]:6d} {int((pr & E).sum()):6d}')
    if r['f1'] > best[0]: best = (r['f1'], float(c))
print(f'\nЛУЧШАЯ out-of-fold граница {best[1]:.3f}: F1 {best[0]:.4f}  (текущая 0.95)')
pr = P >= 0.95; fp = pr & ~T
print('ложные при 0.95 по покрову: ' + ', '.join(f'{int(k)}: {n/fp.sum():.1%}' for k, n in zip(*np.unique(LC[fp], return_counts=True))))
json.dump({'oof_cutoff': best[1], 'oof_f1': best[0], 'grid_f1': {f'{c:.3f}': f1(T, P >= c)['f1'] for c in GRID}},
          open('data/comp/af_oof_cutoff.json', 'w'), indent=2)
