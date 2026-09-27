"""SPEC-88: «C1-MM ≥ 0.5 или (dNBR > 0.27 и чистый пиксель)» против одной C1-MM на свежем MTBS-2. Чистый: NIR+SWIR2 ≥ 0.05 на
обоих снимках, синий «до» < 0.25, NDWI «после» ≤ 0. C1-MM — кэш research/hls-fresh2-v1/c1mm_ensemble.npy (порядок manifest.json)."""
import json, sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, Scorer
D = Path('external/hls_fresh2'); man, X, Y, V, _ = load_dir(D, 'manifest.json', layers=())
P = np.load('research/hls-fresh2-v1/c1mm_ensemble.npy').astype(np.float32); assert len(P) == len(man)
nbr = lambda a: (a[3] - a[5]) / (a[3] + a[5] + 1e-6); m1 = P >= 0.5; m2 = m1.copy(); npre = 0
for i, w in enumerate(man):
    p = D / f"{w['name']}.pre.tif"
    if not p.exists(): continue
    B = rasterio.open(p).read().astype(np.float32); A = X[i]; npre += 1
    clean = V[i] & (B != -9999).all(0) & (A[3] + A[5] >= 0.05) & (B[3] + B[5] >= 0.05) & (B[0] < 0.25) & ((A[1] - A[3]) / (A[1] + A[3] + 1e-6) <= 0)
    m2[i] |= (nbr(B) - nbr(A) > 0.27) & clean
sc = Scorer(Y, V); a, b = sc.stats(m1.astype(np.float32), 0.5), sc.stats(m2.astype(np.float32), 0.5); pt, lo, hi, v = sc.compare(b, a)
pr = lambda M: np.round([(M & Y & V).sum() / (Y & V).sum(), (M & Y & V).sum() / max((M & V).sum(), 1)], 3).tolist()
print(f'MTBS-2 ({len(Y)} окон, со снимком «до» {npre}): модель {sc.iou(a):.4f} (полнота/точность {pr(m1)}) | правило {sc.iou(b):.4f} ({pr(m2)}) | {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
typ = np.array([w['incid_type'] for w in man])
for t in sorted(set(typ)):
    i = np.where(typ == t)[0]
    if len(i) > 1: print(f'  {t:16s} окон {len(i):3d} | модель {sc.iou(a, i):.4f} | правило {sc.iou(b, i):.4f}')
