"""Расширение набора признаков (SPEC-19).

Контекст дал самую крупную прибавку (+0.030), значит стоит копать туда же.
Слабый класс — самое больное место: IoU 0.28 против 0.65 у сильного. Слабая
гарь отличается от фона малым сдвигом dNBR, поэтому проверяются признаки,
усиливающие именно малый сдвиг: RdNBR из литературы (Miller & Thode), окно
шире прежнего и разброс отражения в коротковолновом канале.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/Users/mc/projects/fires')
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs
from src.comp.model import sample_chip, SEED
from src.comp.postproc import drop_small

I = {n: i for i, n in enumerate(NAMES)}

def extra(base):
    """Дополнительные слои поверх текущих 19."""
    dnbr, nbr_pre = base[I['dnbr']], base[I['nbr_pre']]
    b12 = base[I['b12_post']]
    # RdNBR: делит сдвиг на корень предпожарного состояния, усиливая слабую гарь
    rdnbr = dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 0.001))
    win31 = uniform_filter(dnbr, 31, mode='nearest')
    b12_std = np.sqrt(np.maximum(uniform_filter(b12*b12, 5, mode='nearest')
                                 - uniform_filter(b12, 5, mode='nearest')**2, 0))
    return {'rdnbr': rdnbr, 'dnbr_win31': win31, 'b12_std5': b12_std}

SETS = {
    'как сейчас (19)': [],
    '+ rdnbr': ['rdnbr'],
    '+ окно 31': ['dnbr_win31'],
    '+ разброс b12': ['b12_std5'],
    'все три (22)': ['rdnbr', 'dnbr_win31', 'b12_std5'],
}

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])

def build(chip_ids, rng=None, n=18000):
    xs, ys = [], []
    for c in chip_ids:
        ch = d.load(c)
        base = stack(ch)
        ex = extra(base)
        full = np.concatenate([base, np.stack([ex[k] for k in ('rdnbr','dnbr_win31','b12_std5')])])
        # повторяем отбор пикселей sample_chip, но на расширенном наборе
        labels, valid = ch.mask, ch.valid()
        picked = []
        for cls in (0,1,2,3):
            idx = np.flatnonzero((labels==cls).ravel() & valid.ravel())
            if idx.size: picked.append(rng.choice(idx, min(idx.size, n//4), replace=False))
        if not picked:      # чип целиком под облаком — пропускаем, а не падаем
            continue
        idx = np.concatenate(picked)
        xs.append(full.reshape(full.shape[0],-1)[:, idx].T); ys.append(labels.ravel()[idx])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)

rng = np.random.default_rng(SEED)
X, Y = build(fit, rng)
print('пикселей', X.shape, flush=True)
cache = []
for c in tune:
    ch = d.load(c); base = stack(ch); ex = extra(base)
    full = np.concatenate([base, np.stack([ex[k] for k in ('rdnbr','dnbr_win31','b12_std5')])])
    cache.append((ch.mask, full.reshape(full.shape[0],-1).T.astype(np.float32), ch.valid(), ch.shape))
EXTRA_ORDER = ['rdnbr','dnbr_win31','b12_std5']
print('набор                 IoU_burn  mIoU_sev  класс1  секунд', flush=True)
for name, add in SETS.items():
    cols = list(range(len(NAMES))) + [len(NAMES)+EXTRA_ORDER.index(a) for a in add]
    t0=time.time()
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, categorical_features=[cols.index(I['landcover'])],
        random_state=SEED).fit(X[:, cols], Y)
    out=[]
    for truth, flat, ok, shp in cache:
        pred = m.predict(flat[:, cols]).astype(np.uint8).reshape(shp); pred[~ok]=0
        out.append(score_bs(truth, drop_small(pred, 100)))
    print('%-21s %8.4f %9.4f %7.4f %7.0f' % (name,
          np.nanmean([r['iou_burn'] for r in out]), np.nanmean([r['miou_sev'] for r in out]),
          np.nanmean([r['per_class'][1] for r in out]), time.time()-t0), flush=True)
