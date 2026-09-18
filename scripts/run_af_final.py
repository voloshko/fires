"""Финальная AF-модель под сабмит: учится на ВСЕХ 420 чипах.

Решающая граница и пороги выбраны раньше, на валидационной части
(SPEC-17-LIVE-001), и здесь не пересматриваются — иначе замер станет
самоподтверждающимся. Отложенная часть больше не нужна: выбор сделан,
а 40 % данных простаивать не должны.
"""
import sys, json, time; sys.path.insert(0,'/Users/mc/projects/fires')
from src.comp.af import AfDataset, build_training_set, train, save

d = AfDataset('data/comp/train/af')
m = json.load(open('data/comp/model_af.json'))
cutoff = m['cutoff']
ids = d.chip_ids()
t0 = time.time()
x, y = build_training_set(d, ids)
print(f'все чипы: {len(ids)}, пикселей {x.shape}, горящих {int(y.sum())}')
model = train(x, y)
save(model, 'models/af_hgb.pkl', cutoff=cutoff)
print(f'обучено за {time.time()-t0:.0f}с, граница {cutoff} перенесена без пересмотра')
print(f'для справки, замер прежней модели на отложенной части: F1={m["f1"]:.4f}')
