"""A/B по признакам BS на настроечной части. Лёгкая версия.

Прошлый вариант перебирал 11 порогов на две модели — 28 полных проходов по
35 чипам, 20 минут при 800 % CPU. Здесь один проход на модель: сравнивается
только то, ради чего эксперимент затевался, — контекстные признаки.
"""
import sys, json, time, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from sklearn.ensemble import HistGradientBoostingClassifier
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs
from src.comp.model import SEED, sample_chip

CTX = [NAMES.index(n) for n in ("dnbr_win5", "dnbr_win15", "dnbr_std5", "dnbr_chip")]
PIX = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

d = BsDataset('data/comp/train/bs'); split = json.load(open('data/comp/split_bs.json'))
ids = [c for c in split['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
print(f'обучение {len(fit)}, настроечная {len(tune)}, пикселей с чипа {PIX}', flush=True)

rng = np.random.default_rng(SEED); xs, ys = [], []
for cid in fit:
    a, b = sample_chip(d.load(cid), rng, PIX); xs.append(a); ys.append(b)
X, Y = np.concatenate(xs), np.concatenate(ys)
print('обучающих пикселей', X.shape, flush=True)

cache = []
for cid in tune:
    chip = d.load(cid)
    cache.append((chip.mask, stack(chip).reshape(len(NAMES), -1).T, chip.valid()))
print('настроечная часть загружена', flush=True)

for label, cols in [("19 признаков (с контекстом)", list(range(len(NAMES)))),
                    ("15 признаков (контроль)",    [i for i in range(len(NAMES)) if i not in CTX])]:
    t0 = time.time()
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, categorical_features=[cols.index(NAMES.index("landcover"))],
        random_state=SEED).fit(X[:, cols], Y)
    out = []
    for truth, feats, ok in cache:
        pred = m.predict(feats[:, cols]).reshape(truth.shape).astype(np.uint8)
        pred[~ok] = 0
        out.append(score_bs(truth, pred))
    print('%-30s IoU_burn=%.4f mIoU_sev=%.4f  (%.0fс)' % (label,
          np.nanmean([r['iou_burn'] for r in out]),
          np.nanmean([r['miou_sev'] for r in out]), time.time()-t0), flush=True)
