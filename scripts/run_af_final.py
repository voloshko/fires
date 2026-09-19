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
cutoff_oof_random = json.load(open('data/comp/af_oof_cutoff.json'))['oof_cutoff']   # 0.985 — для модели на случайном фоне; см. ниже
ids = d.chip_ids()
t0 = time.time()
# Жёсткие отрицательные (SPEC-33/38, receipt SPEC-38-REPLAY-001): сначала
# базовая модель на случайном фоне, затем фон пересэмплируется по её
# вероятностям при том же бюджете — nested OOF F1 0.9066 → 0.928–0.930 на
# двух сидах. Граница при таком отборе — 0.5, не 0.985: калибровка сдвигается.
from pathlib import Path
from scripts.hypothesis_lab import af_cache, af_set
cache, _ = af_cache('data/comp/train/af', Path('models/af_cache'), ids)
x, y = af_set(cache, ids, 20260918); base = train(x, y, 20260918)
x, y = af_set(cache, ids, 20260918, base); model = train(x, y, 20260918)
cutoff = 0.5
print(f'все чипы: {len(ids)}, пикселей {x.shape}, горящих {int(y.sum())}; жёсткие отрицательные по базовой модели')
save(model, 'models/af_hgb.pkl', cutoff=cutoff)
print(f'обучено за {time.time()-t0:.0f}с, граница {cutoff} перенесена без пересмотра')
print(f'для справки, замер прежней модели на отложенной части: F1={m["f1"]:.4f}')
