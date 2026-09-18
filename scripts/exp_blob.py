"""Фильтр связных областей на бустинге (SPEC-19).

Проверка гипотезы «ложная гарь на фоне — это мелкие пятна» независимо от сети:
если фильтр помогает и бустингу, и сети, гипотеза о природе ошибки верна.
Обучение на 144 чипах, замер на 35 настроечных.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import label
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.model import build_training_set, train, predict

def drop_small(pred, min_px):
    marks, count = label(pred > 0)
    if not count: return pred
    sizes = np.bincount(marks.reshape(-1))
    out = pred.copy(); out[np.isin(marks, np.flatnonzero(sizes < min_px))] = 0
    return out

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
t0=time.time(); x, y = build_training_set(d, fit); m = train(x, y)
print(f'обучено на {len(fit)} чипах за {time.time()-t0:.0f}с', flush=True)
preds = []
for n, c in enumerate(tune, 1):
    ch = d.load(c); preds.append((ch.mask, predict(m, ch)))
    if n % 10 == 0: print(f'  {n}/{len(tune)}', flush=True)
print('порог пятна  IoU_burn  mIoU_sev')
for mn in (0, 50, 100, 200, 400, 800, 1600):
    out = [score_bs(t, drop_small(p, mn) if mn else p) for t, p in preds]
    print('%-12s %8.4f %9.4f' % (mn or 'без фильтра',
          np.nanmean([r['iou_burn'] for r in out]), np.nanmean([r['miou_sev'] for r in out])), flush=True)
