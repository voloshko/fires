"""SPEC-86: C1-S (одна дата, 4 класса) против порогов dNBR (пара дат) на свежем наборе 2 против тяжести MTBS.
mIoU по классам 1–3 пулом, бутстреп по окнам; общие пиксели: метка ≥ 0, MTBS ≠ 6, валидны снимки «до» и «после»."""
import json, sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import c1_softmax
D = Path('external/hls_fresh2'); R = Path('research')
sev_ids = {w['name'] for w in json.load(open(D / 'manifest_sev.json'))['windows']}
man = [w for w in json.load(open(D / 'manifest_pre.json'))['windows'] if w['name'] in sev_ids]
rd = lambda w, k: rasterio.open(D / f"{w['name']}{k}").read()
X = np.stack([rd(w, '_merged.tif').astype(np.float32) for w in man]); XB = np.stack([rd(w, '.pre.tif').astype(np.float32) for w in man])
M = np.stack([rd(w, '.mask.tif')[0] for w in man]); SV = np.stack([rd(w, '.sev.tif')[0] for w in man])
T = np.select([SV == 2, SV == 3, SV == 4], [1, 2, 3], 0); V = (M >= 0) & (SV != 6) & (X != -9999).all(1) & (XB != -9999).all(1)
X[np.broadcast_to(~V[:, None], X.shape)] = 0
P = c1_softmax(X, [R / f'c1s-final-s{s}' for s in (1, 2, 3)], R / 'fresh2-sev-v1' / 'c1s.npy').argmax(1)
nbr = lambda A: (A[:, 3] - A[:, 5]) / (A[:, 3] + A[:, 5] + 1e-6); dn = nbr(XB) - nbr(X)
B = np.digitize(dn, [0.1, 0.27, 0.66])
def conf(Pr):
    C = np.zeros((len(T), 4, 4), np.int64)
    for i in range(len(T)): C[i] = np.bincount(T[i][V[i]] * 4 + Pr[i][V[i]], minlength=16).reshape(4, 4)
    return C
def miou(C):
    S = C.sum(0); iou = [S[k, k] / max(S[k, :].sum() + S[:, k].sum() - S[k, k], 1) for k in range(4)]; return float(np.mean(iou[1:])), iou
Cm, Cb = conf(P), conf(B); rng = np.random.default_rng(0)
d = [miou(Cm[i])[0] - miou(Cb[i])[0] for i in (rng.integers(0, len(T), len(T)) for _ in range(1000))]; lo, hi = np.percentile(d, [2.5, 97.5])
mm, im = miou(Cm); mb, ib = miou(Cb); pt = mm - mb
bin_iou = lambda C: C.sum(0)[1:, 1:].sum() / (C.sum(0).sum() - C.sum(0)[0, 0])
print(f'окон {len(T)} | пикселей тяжести 1/2/3: {[int(((T == k) & V).sum()) for k in (1, 2, 3)]}')
print(f'C1-S (одна дата): mIoU_sev {mm:.4f} | IoU по классам {np.round(im, 4).tolist()} | бинарный IoU гари {bin_iou(Cm):.4f}')
print(f'dNBR (пара дат):   mIoU_sev {mb:.4f} | IoU по классам {np.round(ib, 4).tolist()} | бинарный IoU гари {bin_iou(Cb):.4f}')
v = 'ЛУЧШЕ' if pt > 0 and lo > 0 else 'ХУЖЕ' if hi < 0 else 'ВРОВЕНЬ'
print(f'C1-S − dNBR: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
print('матрица C1-S (строки — MTBS 0..3, столбцы — модель):', Cm.sum(0).tolist())
