"""Где насыщается объём выборки с чипа (SPEC-19).

6000 → 18000 дало +0.019 IoU. Вопрос — есть ли ещё запас, или кривая вышла на
полку. Обучение на 144 чипах, замер на 35 настроечных, всегда с фильтром
мелких пятен: без него сравнение неинформативно.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/home/mc/fires')
from scipy.ndimage import label
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.model import train, predict, sample_chip, SEED

def drop_small(pred, min_px=100):
    marks, count = label(pred > 0)
    if not count: return pred
    sizes = np.bincount(marks.reshape(-1))
    out = pred.copy(); out[np.isin(marks, np.flatnonzero(sizes < min_px))] = 0
    return out

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
cache_tune = [d.load(c) for c in tune]
print('пикселей  всего      IoU_burn  mIoU_sev  секунд', flush=True)
for pix in (18000, 36000, 72000, 150000):
    t0 = time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
    for c in fit:
        a, b = sample_chip(d.load(c), rng, pix); xs.append(a); ys.append(b)
    x, y = np.concatenate(xs), np.concatenate(ys)
    m = train(x, y)
    out = [score_bs(ch.mask, drop_small(predict(m, ch))) for ch in cache_tune]
    print('%-9d %-10d %8.4f %9.4f %7.0f' % (pix, x.shape[0],
          np.nanmean([r['iou_burn'] for r in out]),
          np.nanmean([r['miou_sev'] for r in out]), time.time()-t0), flush=True)
