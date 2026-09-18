"""Подбор решающей границы «гарь / не гарь» (пункт 2 плана улучшений).

Модель обучается ТОЛЬКО на 144 чипах и меряется на 35 настроечных, которых не
видела. Прогон на финальной модели, обученной на всех 224, дал бы 0.65 вместо
0.47 — это внутривыборочное число, и принимать по нему решения нельзя.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/Users/mc/projects/fires')
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs
from src.comp.model import build_training_set, train

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
t0 = time.time()
x, y = build_training_set(d, fit)
m = train(x, y)
print(f'обучено на {len(fit)} чипах за {time.time()-t0:.0f}с, меряю на {len(tune)}', flush=True)

cache = []
for n, c in enumerate(tune, 1):
    ch = d.load(c)
    cache.append((ch.mask, m.predict_proba(stack(ch).reshape(len(NAMES), -1).T), ch.valid()))
    if n % 10 == 0: print(f'  {n}/{len(tune)}', flush=True)

print('порог   IoU_burn  mIoU_sev')
for cut in [None, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    out = []
    for truth, p, ok in cache:
        sev = 1 + p[:, 1:].argmax(1)
        burn = (1 - p[:, 0]) > cut if cut is not None else p.argmax(1) > 0
        pred = np.where(burn, sev, 0).reshape(truth.shape).astype(np.uint8); pred[~ok] = 0
        out.append(score_bs(truth, pred))
    print('%-7s %8.4f %9.4f' % ('argmax' if cut is None else f'{cut:.2f}',
          np.nanmean([r['iou_burn'] for r in out]), np.nanmean([r['miou_sev'] for r in out])), flush=True)
