"""Бриф №3, блок A: где теряется степень. Рецепт v21 на 144 групповых чипах:
матрица ошибок 4×4, IoU по классам, доля ошибок слабого класса у кромки гари,
и дешёвая альтернатива — степень по порогам dNBR внутри найденной гари
(пороги подбираются на четырёх фолдах, применяются к пятому)."""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
folds = []
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(base['optical'] / 'probabilities.npy').astype(np.float32)
    PS = np.mean([np.load(OWN / f'bs-confirm-siam-f{f}-{t}-v1/probabilities.npy').astype(np.float32) for t in ('over', 'over2')], 0)
    OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips]); T = np.stack([c.mask for c in chips]); DN = np.stack([c.dnbr() for c in chips])
    pn = 0.5 * PO + 0.5 * PS; P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    out = np.stack([drop_far(o, anchor=a) for o, a in zip(out, (PO.argmax(3) > 0) & (PS.argmax(3) > 0))])
    folds.append(dict(T=T, out=out, DN=DN, OK=OK, P=P))
T = np.concatenate([f['T'] for f in folds]); out = np.concatenate([f['out'] for f in folds]); OK = np.concatenate([f['OK'] for f in folds])
cm = np.zeros((4, 4), int)
for a in range(4):
    for b in range(4): cm[a, b] = ((T == a) & (out == b)).sum()
print('матрица ошибок (строки — истина 0..3, столбцы — предсказание), тыс. пикс.:'); print(np.round(cm / 1000).astype(int))
for k in range(1, 4):
    i = ((T == k) & (out == k)).sum(); u = ((T == k) | (out == k)).sum(); print(f'IoU класса {k}: {i/u:.3f}  (истины {(T==k).sum()/1000:.0f} тыс., предсказано {(out==k).sum()/1000:.0f} тыс.)')
r = score_bs(T.reshape(-1), out.reshape(-1)); print(f'v21: IoU гари {r["iou_burn"]:.4f}, mIoU степеней {r["miou_sev"]:.4f}, взв {w(r):.4f}')
# кромка: ошибки слабого класса в 2 пикс. от границы истинной гари
tb = T > 0; edge = np.stack([binary_dilation(b, iterations=2) & ~binary_erosion(b, iterations=2) for b in tb])
err1 = (T == 1) & (out != 1)
print(f'слабый класс: ошибок {err1.sum()/1000:.0f} тыс., из них у кромки гари (±2 пикс.) {(err1&edge).sum()/err1.sum():.1%}; сам класс у кромки на {((T==1)&edge).sum()/(T==1).sum():.1%}')
print(f'слабый класс уходит: в фон {((T==1)&(out==0)).sum()/(T==1).sum():.1%}, в средний {((T==1)&(out==2)).sum()/(T==1).sum():.1%}, в сильный {((T==1)&(out==3)).sum()/(T==1).sum():.1%}')
# идеальная степень при найденной гари: потолок
ideal = np.where(out > 0, np.where(T > 0, T, out), 0); ri = score_bs(T.reshape(-1), ideal.reshape(-1))
print(f'потолок: если внутри найденной гари степень верна — mIoU {ri["miou_sev"]:.4f}, взв {w(ri):.4f} (+{w(ri)-w(r):.4f})')
# степень по порогам dNBR внутри найденной гари, пороги с других фолдов
def fit_thr(TT, DD, mask):
    best, bt = -1, None
    for t1 in np.arange(0.05, 0.45, 0.02):
        for t2 in np.arange(t1 + 0.04, 0.8, 0.02):
            sev = np.where(DD < t1, 1, np.where(DD < t2, 2, 3)); m = mask & (TT > 0)
            iou = np.mean([((TT == k) & (sev == k) & m).sum() / max((((TT == k) | (sev == k)) & m).sum(), 1) for k in (1, 2, 3)])
            if iou > best: best, bt = iou, (t1, t2)
    return bt
alt = []
for f in range(5):
    others = [g for g in range(5) if g != f]
    t1, t2 = fit_thr(np.concatenate([folds[g]['T'] for g in others]), np.concatenate([folds[g]['DN'] for g in others]), np.concatenate([folds[g]['OK'] for g in others]))
    o = folds[f]['out']; dn = folds[f]['DN']; sev = np.where(dn < t1, 1, np.where(dn < t2, 2, 3)); alt.append(np.where(o > 0, sev, 0).astype(np.uint8)); print(f'фолд {f}: пороги dNBR {t1:.2f}/{t2:.2f}')
alt = np.concatenate(alt); ra = score_bs(T.reshape(-1), alt.reshape(-1))
print(f'степень по порогам dNBR внутри найденной гари: mIoU {ra["miou_sev"]:.4f}, взв {w(ra):.4f} ({w(ra)-w(r):+.4f})')
mix = np.where(out > 0, np.where(out == 1, alt, out), 0); rm = score_bs(T.reshape(-1), mix.reshape(-1))
print(f'смесь: сеть решает 2/3, пороги dNBR — только там, где сеть сказала 1: mIoU {rm["miou_sev"]:.4f}, взв {w(rm):.4f} ({w(rm)-w(r):+.4f})')
