"""Состав выборки по классам (SPEC-19).

Перевес фона проверялся на 6000 пикселей и без фильтра мелких пятен — и
проигрывал. Теперь, когда установлено, что потеря IoU идёт именно от ложной
гари на фоне, проверка повторяется на 18000 пикселей и с фильтром.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/Users/mc/projects/fires')
from scipy.ndimage import label
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs
from src.comp.model import train, predict, SEED

def drop_small(pred, min_px=100):
    marks, count = label(pred > 0)
    if not count: return pred
    sizes = np.bincount(marks.reshape(-1))
    out = pred.copy(); out[np.isin(marks, np.flatnonzero(sizes < min_px))] = 0
    return out

def sample(chip, rng, quota):
    feats = stack(chip); valid = chip.valid().ravel(); labels = chip.mask
    picked = []
    for cls in (0, 1, 2, 3):
        idx = np.flatnonzero((labels == cls).ravel() & valid)
        if idx.size: picked.append(rng.choice(idx, min(idx.size, quota[cls]), replace=False))
    if not picked: return np.empty((0, len(NAMES)), np.float32), np.empty(0, np.uint8)
    idx = np.concatenate(picked)
    return feats.reshape(len(NAMES), -1)[:, idx].T, labels.ravel()[idx]

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
cache_tune = [d.load(c) for c in tune]
N = 18000
mixes = {"поровну 25/25/25/25": [.25,.25,.25,.25],
         "фон вдвое 40/20/20/20": [.40,.20,.20,.20],
         "фон втрое 55/15/15/15": [.55,.15,.15,.15],
         "фон как в данных 70/10/10/10": [.70,.10,.10,.10]}
print('состав                         IoU_burn  mIoU_sev  секунд', flush=True)
for name, share in mixes.items():
    t0 = time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
    quota = {c: int(N*share[c]) for c in range(4)}
    for c in fit:
        a, b = sample(d.load(c), rng, quota); xs.append(a); ys.append(b)
    m = train(np.concatenate(xs), np.concatenate(ys))
    out = [score_bs(ch.mask, drop_small(predict(m, ch))) for ch in cache_tune]
    print('%-30s %8.4f %9.4f %7.0f' % (name,
          np.nanmean([r['iou_burn'] for r in out]),
          np.nanmean([r['miou_sev'] for r in out]), time.time()-t0), flush=True)
