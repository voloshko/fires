"""AF: признаки алгоритма-генератора меток (SPEC-17/19).

Разметка активного горения почти наверняка происходит из контекстного алгоритма
VNP14: он сравнивает пиксель не со средним окна, а с МЕДИАНОЙ и MAD фона, из
которого исключены сами кандидаты. Мы даём модели среднее по uniform_filter.
Гипотеза: дать те же статистики, что видит генератор истины, — модель выучит
генератор. Замер — out-of-fold на тех же 5 фолдах, что exp_af_cut.py.
"""
import sys, json, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import median_filter, maximum_filter, uniform_filter
from src.comp.af import AfDataset, features, train, f1, NAMES, THRESHOLDS, PIXELS_BACKGROUND, SEED

def robust_bg(band, cand, size):
    """Медиана и MAD окна по фону без кандидатов: кандидаты заменяются локальной
    медианой, чтобы не тянуть статистику вверх, — приближение отбраковки VNP14."""
    filled = np.where(cand, median_filter(band, size, mode='nearest'), band)
    med = median_filter(filled, size, mode='nearest')
    mad = median_filter(np.abs(filled - med), size, mode='nearest')
    return med, np.maximum(mad, 0.5)

def features2(chip):
    base = features(chip)
    i4, i5 = chip.viirs[3], chip.viirs[4]; dt = i4 - i5
    cand = (i4 - uniform_filter(i4, 15, mode='nearest') > 10) | (dt > 20)   # грубые кандидаты
    extra = []
    for size in (7, 15):
        m4, s4 = robust_bg(i4, cand, size); md, sd = robust_bg(dt, cand, size)
        extra += [(i4 - m4) / s4, (dt - md) / sd, s4, sd]
    extra.append(maximum_filter(i4, 3, mode='nearest') - i4)      # сосед горячее? (кластеры)
    extra.append(uniform_filter(cand.astype(np.float32), 5, mode='nearest'))   # плотность кандидатов
    extra.append((chip.viirs[5] > 90).astype(np.float32))          # ночь
    return np.concatenate([base, np.stack(extra).astype(np.float32)])

NAMES2 = NAMES + tuple(f'z4_{s}' for s in (7,15)) + ('zdt_7','zdt_15','mad4_7','maddt_7','mad4_15','maddt_15','nb_max','cand_dens','night')
# порядок в extra: (z4_7, zdt_7, mad4_7, maddt_7, z4_15, zdt_15, mad4_15, maddt_15, nb_max, cand_dens, night)
NAMES2 = NAMES + ('z4_7','zdt_7','mad4_7','maddt_7','z4_15','zdt_15','mad4_15','maddt_15','nb_max','cand_dens','night')

def build(d, chip_ids, feat_fn, n_names, seed=SEED):
    rng = np.random.default_rng(seed); xs, ys = [], []
    for c in chip_ids:
        ch = d.load(c); f = feat_fn(ch).reshape(n_names, -1).T
        lab = ch.mask.reshape(-1) > 0; ok = ch.valid().reshape(-1)
        fire = np.flatnonzero(lab & ok); back = np.flatnonzero(~lab & ok)
        if back.size > PIXELS_BACKGROUND: back = rng.choice(back, PIXELS_BACKGROUND, replace=False)
        take = np.concatenate([fire, back]); xs.append(f[take]); ys.append(lab[take].astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys)

def oof(d, folds, feat_fn, names):
    from sklearn.ensemble import HistGradientBoostingClassifier
    P, T = [], []
    for k in range(len(folds)):
        fit = [c for j, f in enumerate(folds) if j != k for c in f]
        x, y = build(d, fit, feat_fn, len(names))
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                           categorical_features=[names.index('landcover')], random_state=SEED).fit(x, y)
        for c in folds[k]:
            ch = d.load(c); ok = ch.valid().reshape(-1)
            p = m.predict_proba(feat_fn(ch).reshape(len(names), -1).T)[:, 1]
            p[~ok] = 0; p[np.isin(ch.aux[0].reshape(-1), THRESHOLDS['exclude_landcover'])] = 0
            P.append(p); T.append(ch.mask.reshape(-1) > 0)
    P, T = np.concatenate(P), np.concatenate(T)
    grid = np.concatenate([np.arange(0.5, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)])
    return max((f1(T, P >= c)['f1'], float(c)) for c in grid), f1(T, P >= 0.985)

d = AfDataset('data/comp/train/af'); sp = json.load(open('data/comp/split_af.json'))
chips = sorted(sp['train'] + sp['val']); folds = [chips[i::5] for i in range(5)]
for name, fn, nm in (('v1 среднее окна', features, NAMES), ('v2 медиана/MAD', features2, NAMES2)):
    t0 = time.time(); (best, cut), at985 = oof(d, folds, fn, nm)
    print(f'{name:18s} лучшая граница {cut:.3f}: F1 {best:.4f};  при 0.985: F1 {at985["f1"]:.4f} '
          f'prec {at985["precision"]:.4f} rec {at985["recall"]:.4f}  ({time.time()-t0:.0f}с)', flush=True)
