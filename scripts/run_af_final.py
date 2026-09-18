"""Финальная AF-модель под сабмит: учится на ВСЕХ 420 чипах.

Решающая граница и пороги выбраны раньше, на валидационной части
(SPEC-17-LIVE-001), и здесь не пересматриваются — иначе замер станет
самоподтверждающимся. Отложенная часть больше не нужна: выбор сделан,
а 40 % данных простаивать не должны.
"""
import sys, json, time; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.af import AfDataset, build_training_set, train, save

d = AfDataset('data/comp/train/af')
m = json.load(open('data/comp/model_af.json'))
# Граница 0.95 с val попала в ПОСЛЕДНЮЮ точку сетки 0.05..0.95 — оптимум был не
# виден. Пересчитана out-of-fold на 336 чипах train+val (scripts/exp_af_cut.py):
# 0.985, F1 0.9052 против 0.8973 при 0.95. Отложенные 84 не трогались.
cutoff = json.load(open('data/comp/af_oof_cutoff.json'))['oof_cutoff']
ids = d.chip_ids()
t0 = time.time()
x, y = build_training_set(d, ids)
print(f'все чипы: {len(ids)}, пикселей {x.shape}, горящих {int(y.sum())}')
model = train(x, y)
save(model, 'models/af_hgb.pkl', cutoff=cutoff)
print(f'обучено за {time.time()-t0:.0f}с, граница {cutoff} перенесена без пересмотра')
print(f'для справки, замер прежней модели на отложенной части: F1={m["f1"]:.4f}')
