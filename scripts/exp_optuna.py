"""Поиск гиперпараметров бустинга по Optuna (SPEC-19).

Ручной перебор из пяти точек — грубая сетка, а не поиск: он проверил только то,
что я успел придумать. Здесь TPE ищет по семи осям сразу, включая те, что
руками не трогались (регуляризация, доля признаков, минимум в листе).

Цель — метрика соревнования по гари: 0.35·IoU_burn + 0.30·mIoU_sev,
нормированная на сумму весов. Обучение на 144 чипах, замер на 35 настроечных;
отложенные 45 не участвуют.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/home/mc/fires')
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from src.comp.chips import BsDataset
from src.comp.features import NAMES
from src.comp.metric import score_bs
from src.comp.model import sample_chip, SEED
from src.comp.postproc import drop_small

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
optuna.logging.set_verbosity(optuna.logging.WARNING)

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
X, Y = np.concatenate(xs), np.concatenate(ys)
cache = [(ch.mask, np.ascontiguousarray(__import__('src.comp.features', fromlist=['stack']).stack(ch)
          .reshape(len(NAMES), -1).T), ch.valid(), ch.shape) for ch in (d.load(c) for c in tune)]
print(f'обучающих пикселей {X.shape}, замер на {len(cache)} чипах, проб {TRIALS}', flush=True)

def objective(t):
    m = HistGradientBoostingClassifier(
        max_iter=t.suggest_int('max_iter', 150, 900, step=50),
        learning_rate=t.suggest_float('learning_rate', 0.02, 0.30, log=True),
        max_leaf_nodes=t.suggest_int('max_leaf_nodes', 15, 127, log=True),
        min_samples_leaf=t.suggest_int('min_samples_leaf', 5, 200, log=True),
        l2_regularization=t.suggest_float('l2', 1e-3, 30.0, log=True),
        max_features=t.suggest_float('max_features', 0.4, 1.0),
        max_bins=t.suggest_categorical('max_bins', [64, 128, 255]),
        categorical_features=[NAMES.index('landcover')],
        early_stopping=False, random_state=SEED).fit(X, Y)
    out = []
    for truth, flat, ok, shp in cache:
        pred = m.predict(flat).astype(np.uint8).reshape(shp); pred[~ok] = 0
        out.append(score_bs(truth, drop_small(pred, 100)))
    burn = float(np.nanmean([r['iou_burn'] for r in out]))
    miou = float(np.nanmean([r['miou_sev'] for r in out]))
    t.set_user_attr('iou_burn', burn); t.set_user_attr('miou_sev', miou)
    return (0.35*burn + 0.30*miou) / 0.65

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
def log(st, tr):
    print(f'проба {tr.number:3d}  цель {tr.value:.4f}  IoU {tr.user_attrs["iou_burn"]:.4f}  '
          f'mIoU {tr.user_attrs["miou_sev"]:.4f}  лучшее {st.best_value:.4f}  '
          f'{time.time()-t0:.0f}с', flush=True)
study.optimize(objective, n_trials=TRIALS, callbacks=[log])
print('\nлучшее:', json.dumps(study.best_params, ensure_ascii=False))
print(f'цель {study.best_value:.4f}, IoU {study.best_trial.user_attrs["iou_burn"]:.4f}, '
      f'mIoU {study.best_trial.user_attrs["miou_sev"]:.4f}')
json.dump({'best_params': study.best_params, 'best_value': study.best_value,
           **study.best_trial.user_attrs}, open('optuna_bs.json','w'), ensure_ascii=False, indent=2)
