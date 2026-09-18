"""Относительный фильтр связных областей (SPEC-19). Истина — периметр ОДНОГО
пожара; предсказанные пятна вдали от главного (убранные поля, соседние старые
гари) — ложные. Фиксированный порог по размеру (100/200 пикс.) отвергнут;
здесь порог задаётся долей от крупнейшего пятна чипа. Из кэша, секунды."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import label
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
pn = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
p = (0.4*PB + 0.6*pn); base = p.argmax(3).astype(np.uint8); base[~OK] = pn.argmax(3)[~OK]; base[ZERO] = 0
def rel_filter(pred, frac, min_abs=0):
    out = pred.copy(); marks, n = label(pred > 0)
    if n < 2: return out
    sizes = np.bincount(marks.ravel()); sizes[0] = 0
    small = np.flatnonzero((sizes < frac * sizes.max()) | (sizes < min_abs)); small = small[small > 0]
    out[np.isin(marks, small)] = 0; return out
def sc(pred):
    r = score_bs_micro(list(T), list(pred)); return f"{r['iou_burn']:.4f}/{r['miou_sev']:.4f} взв {(0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65:.4f}"
print('без фильтра:', sc(base))
for frac in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    print(f'доля < {frac:<5}:', sc(np.stack([rel_filter(x, frac) for x in base])))
# оставить только крупнейшее пятно
print('только крупнейшее:', sc(np.stack([rel_filter(x, 1.0) for x in base])))
