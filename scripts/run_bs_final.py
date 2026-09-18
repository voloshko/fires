"""Финальная модель гари под сабмит: бустинг на ВСЕХ 224 чипах, 19 признаков.

Отложенные 45 чипов входят в обучение намеренно: выбор конфигурации сделан на
настроечной части, оценка больше не нужна, а 25 % данных простаивать не должны.
Числа для receipt'а берутся из прежнего замера, сделанного до этого шага.
"""
import sys, time, numpy as np; sys.path.insert(0,'/Users/mc/projects/fires')
from src.comp.chips import BsDataset
from src.comp.model import build_training_set, train, save
import rasterio.errors

d = BsDataset('data/comp/train/bs')
ids = [c for c in d.chip_ids() if d.has_post(c)]
ok = []
for c in ids:
    try: d.load(c); ok.append(c)
    except rasterio.errors.RasterioIOError: print('пропущен нечитаемый чип:', c)
print(f'чипов с post: {len(ids)}, читаемых: {len(ok)}')
t0=time.time(); x, y = build_training_set(d, ok)
print('пикселей', x.shape, 'классы', np.bincount(y).tolist(), flush=True)
m = train(x, y); save(m, 'models/bs_hgb.pkl')
print(f'обучено за {time.time()-t0:.0f}с → models/bs_hgb.pkl')
