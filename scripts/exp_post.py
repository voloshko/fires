"""Постобработка и объём выборки на бустинге (SPEC-19).

Фильтр мелких пятен уже дал +0.027. Здесь проверяется, добавляет ли к нему
что-нибудь залатывание дыр внутри гари, и стоит ли брать с чипа больше
пикселей. Обучение на 144 чипах, замер на 35 настроечных.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/Users/mc/projects/fires')
from scipy.ndimage import label, binary_fill_holes, binary_closing
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.model import build_training_set, train, predict

def drop_small(pred, min_px):
    marks, count = label(pred > 0)
    if not count: return pred
    sizes = np.bincount(marks.reshape(-1))
    out = pred.copy(); out[np.isin(marks, np.flatnonzero(sizes < min_px))] = 0
    return out

def fill(pred):
    """Дыра внутри гари — чаще артефакт, чем уцелевший островок леса."""
    burn = pred > 0
    holes = binary_fill_holes(burn) & ~burn
    out = pred.copy(); out[holes] = 1     # залатанное считаем слабым поражением
    return out

def close(pred):
    burn = binary_closing(pred > 0, np.ones((5, 5)))
    out = pred.copy(); out[burn & (pred == 0)] = 1
    return out

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])

for pix in (6000, 18000):
    t0 = time.time(); x, y = build_training_set(d, fit, n=pix) if 'n' in train.__code__.co_varnames else (None, None)
    from src.comp.model import sample_chip, SEED
    rng = np.random.default_rng(SEED); xs, ys = [], []
    for c in fit:
        a, b = sample_chip(d.load(c), rng, pix); xs.append(a); ys.append(b)
    x, y = np.concatenate(xs), np.concatenate(ys)
    m = train(x, y)
    print(f'\n=== {pix} пикселей с чипа: {x.shape[0]} всего, обучено за {time.time()-t0:.0f}с', flush=True)
    preds = [(ch.mask, predict(m, ch)) for ch in (d.load(c) for c in tune)]
    for name, fn in [("как есть", lambda p: p),
                     ("фильтр 100", lambda p: drop_small(p, 100)),
                     ("фильтр 100 + дыры", lambda p: fill(drop_small(p, 100))),
                     ("фильтр 100 + закрытие", lambda p: close(drop_small(p, 100))),
                     ("дыры без фильтра", fill)]:
        out = [score_bs(t, fn(p)) for t, p in preds]
        print('  %-24s IoU_burn %.4f  mIoU_sev %.4f' % (name,
              np.nanmean([r['iou_burn'] for r in out]), np.nanmean([r['miou_sev'] for r in out])), flush=True)
