"""Гиперпараметры бустинга (SPEC-19). Обучение на 144, замер на 35."""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/Users/mc/projects/fires')
from sklearn.ensemble import HistGradientBoostingClassifier
from src.comp.chips import BsDataset
from src.comp.features import NAMES
from src.comp.metric import score_bs
from src.comp.model import predict, sample_chip, SEED

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
X, Y = np.concatenate(xs), np.concatenate(ys)
cache = [d.load(c) for c in tune]
print('пикселей', X.shape, flush=True)
print('вариант                  IoU_burn  mIoU_sev  секунд', flush=True)
grid = [("как сейчас 300/31", dict(max_iter=300, max_leaf_nodes=31)),
        ("600 деревьев",      dict(max_iter=600, max_leaf_nodes=31)),
        ("63 листа",          dict(max_iter=300, max_leaf_nodes=63)),
        ("600/63",            dict(max_iter=600, max_leaf_nodes=63)),
        ("медленнее 0.05",    dict(max_iter=600, max_leaf_nodes=31, learning_rate=0.05))]
for name, kw in grid:
    t0=time.time()
    m = HistGradientBoostingClassifier(l2_regularization=1.0, random_state=SEED,
        categorical_features=[NAMES.index("landcover")], **{'learning_rate':0.1, **kw}).fit(X, Y)
    out = [score_bs(ch.mask, predict(m, ch)) for ch in cache]
    print('%-24s %8.4f %9.4f %7.0f' % (name,
          np.nanmean([r['iou_burn'] for r in out]),
          np.nanmean([r['miou_sev'] for r in out]), time.time()-t0), flush=True)
