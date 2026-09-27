"""SPEC-87: «C1-MM ≥ 0.5 или dNBR > 0.27» против одной C1-MM на проверке EMS и FLOGA; маска воды «Fmask и NDWI»."""
import json, sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, Scorer
from src.comp.terrain import water_ndwi
R = Path('research'); MM = [R / f'hls-c1mm-final-s{s}' for s in (1, 2, 3, 4, 5)]; Ys, Vs, A, B = [], [], [], []
for title, d, cache in (('EMS', 'external/ems_hls', R / 'ems-test-v1'), ('FLOGA', 'external/floga_hls', R / 'floga-test-v1')):
    D = Path(d); man, X, Y, V, L = load_dir(d, 'manifest_test.json', layers=('fmask',))
    W = np.stack([water_ndwi(f, x[1], x[3]) for f, x in zip(L['fmask'], X)]); P = c1_probs(X, MM, cache / 'c1mm.npy')
    m1 = (P >= 0.5) & ~W; m2 = m1.copy(); npre = 0
    nbr = lambda a: (a[3] - a[5]) / (a[3] + a[5] + 1e-6)
    for i, w in enumerate(man):
        p = D / f"{w['name']}.pre.tif"
        if not p.exists(): continue
        Xb = rasterio.open(p).read().astype(np.float32); vb = (Xb != -9999).all(0) & V[i]; npre += 1
        m2[i] |= (nbr(Xb) - nbr(X[i]) > 0.27) & vb & ~W[i]
    sc = Scorer(Y, V); a, b = sc.stats(m1.astype(np.float32), 0.5), sc.stats(m2.astype(np.float32), 0.5); pt, lo, hi, v = sc.compare(b, a)
    pr = lambda M: ((M & Y & V).sum() / (Y & V).sum(), (M & Y & V).sum() / max((M & V).sum(), 1))
    print(f'{title} ({len(Y)} окон, со снимком «до» {npre}): модель {sc.iou(a):.4f} (полнота/точность {np.round(pr(m1), 3).tolist()}) | '
          f'модель или dNBR {sc.iou(b):.4f} ({np.round(pr(m2), 3).tolist()}) | разница {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}', flush=True)
    Ys.append(Y); Vs.append(V); A.append(m1); B.append(m2)
Y, V = np.concatenate(Ys), np.concatenate(Vs); sc = Scorer(Y, V); a, b = sc.stats(np.concatenate(A).astype(np.float32), 0.5), sc.stats(np.concatenate(B).astype(np.float32), 0.5); pt, lo, hi, v = sc.compare(b, a)
print(f'H1 (оба набора, {len(Y)} окон): модель {sc.iou(a):.4f} | модель или dNBR {sc.iou(b):.4f} | {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
