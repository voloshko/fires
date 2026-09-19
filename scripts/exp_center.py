"""За пределы семантики «один пожар — один периметр» (SPEC-19). Чип нарезан
вокруг события: размеченный пожар должен тяготеть к центру, чужие — к краям.
Проверка распределений и фильтр компонент по расстоянию до главного пятна."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import label, distance_transform_edt as edt
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
PN = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
P = 0.4*PB + 0.6*PN; burn = P.argmax(3) > 0; burn[~OK] = (PN.argmax(3) > 0)[~OK]
base = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); base[ZERO] = 0
t = T > 0; fp = (base > 0) & ~t
H, W = T.shape[1:]; yy, xx = np.mgrid[:H, :W]; r = np.hypot(yy - H/2, xx - W/2)
print('расстояние от центра: доля истины / доля fp / доля площади')
for lo, hi in ((0, 64), (64, 128), (128, 192), (192, 256), (256, 400)):
    m = (r >= lo) & (r < hi); print(f'  {lo:3d}-{hi:3d}: {t[:, m].sum()/t.sum():.3f}  {fp[:, m].sum()/fp.sum():.3f}  {m.mean():.3f}')
# компоненты предсказания: касание границы, расстояние до главного пятна, доля истины внутри
rows = []
for i in range(len(base)):
    marks, n = label(base[i] > 0)
    if not n: continue
    sizes = np.bincount(marks.ravel()); sizes[0] = 0; main = sizes.argmax()
    dist_main = edt(marks != main)
    for k in range(1, n + 1):
        m = marks == k; border = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        rows.append((i, k, int(m.sum()), float(t[i][m].mean()), bool(border), float(dist_main[m].min()) if k != main else 0.0, float(r[m].min())))
rows = np.array(rows, dtype=object)
size = rows[:, 2].astype(int); purity = rows[:, 3].astype(float); border = rows[:, 4].astype(bool); dmain = rows[:, 5].astype(float)
print(f'\nкомпонент предсказания: {len(rows)}; главных: {(dmain == 0).sum()}')
for name, m in (('касаются границы', border), ('не касаются', ~border)):
    print(f'  {name:18s}: {m.sum():4d} комп., пикселей {size[m].sum():7d}, доля истины в них {np.average(purity[m], weights=size[m]):.3f}')
print('  по расстоянию до главного пятна (не главные):')
for lo, hi in ((0, 1), (1, 20), (20, 50), (50, 100), (100, 200), (200, 1e9)):
    m = (dmain >= lo) & (dmain < hi) & (dmain > 0) if lo else (dmain == 0)
    if m.sum(): print(f'    {lo:>4}-{hi if hi < 1e9 else "∞":>4}: {m.sum():4d} комп., пикселей {size[m].sum():7d}, доля истины {np.average(purity[m], weights=size[m]):.3f}')
def sc(pred):
    q = score_bs_micro(list(T), list(pred)); return f"{q['iou_burn']:.4f}/{q['miou_sev']:.4f} взв {(0.35*q['iou_burn']+0.30*q['miou_sev'])/0.65:.4f}"
print('\nбаза:', sc(base))
for D in (50, 100, 150, 200):
    out = base.copy()
    for i in range(len(out)):
        marks, n = label(out[i] > 0)
        if n < 2: continue
        sizes = np.bincount(marks.ravel()); sizes[0] = 0; main = sizes.argmax(); dist_main = edt(marks != main)
        for k in range(1, n + 1):
            if k != main and dist_main[marks == k].min() > D: out[i][marks == k] = 0
    print(f'убрать компоненты дальше {D} пикс. от главного пятна:', sc(out))
