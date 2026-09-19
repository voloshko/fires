"""Кромка (SPEC-19). Профиль ошибок по расстоянию до истинной кромки: 83 %
пропусков лежат в 8 пикселях от неё, а 65 % ложной гари — дальше 16 пикселей
(отдельные объекты). Значит, у кромки модель систематически недобирает.
Проверка: расширить гарь на пиксель (класс — ближайший), и порог P(гарь)
у кромки. Из кэша, секунды."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import binary_dilation, distance_transform_edt as edt
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
pn = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
p = 0.4*PB + 0.6*pn; p[~OK] = pn[~OK]
base = p.argmax(3).astype(np.uint8); base[ZERO] = 0
def sc(pred):
    r = score_bs_micro(list(T), list(pred)); return f"{r['iou_burn']:.4f}/{r['miou_sev']:.4f} взв {(0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65:.4f}"
def grow(pred, it):
    out = pred.copy()
    for i in range(len(out)):
        burn = out[i] > 0
        if not burn.any(): continue
        ring = binary_dilation(burn, iterations=it) & ~burn
        idx = edt(~burn, return_distances=False, return_indices=True)
        out[i][ring] = out[i][tuple(idx)][ring]
    out[ZERO] = 0; return out
print('база:', sc(base))
for it in (1, 2): print(f'расширение на {it} пикс:', sc(grow(base, it)))
sev = p[..., 1:].argmax(3) + 1
for thr in (0.5, 0.45, 0.4, 0.35, 0.3):
    q = np.where(p[..., 0] < 1 - thr, sev, 0).astype(np.uint8); q[ZERO] = 0
    print(f'гарь при P(гарь) ≥ {thr}:', sc(q))
# порог только у кромки предсказания (в 2 пикс.)
for thr in (0.35, 0.25):
    q = base.copy()
    for i in range(len(q)):
        burn = q[i] > 0; ring = binary_dilation(burn, iterations=2) & ~burn
        add = ring & (p[i, ..., 0] < 1 - thr); q[i][add] = sev[i][add]
    q[ZERO] = 0; print(f'у кромки (2 пикс) гарь при P ≥ {thr}:', sc(q))
