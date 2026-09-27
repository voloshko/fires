"""SPEC-84 ч. 1: EFFIS и наша C1-MM против экспертной разметки (проверка FLOGA 87 окон, проверка EMS 70 окон).
Контур EFFIS его года в сетке окна кэшируется как <окно>.effis.tif; C1-MM — кэш SPEC-82 с маской «Fmask и NDWI», порог 0.5."""
import sys, time, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, Scorer
from src.comp.terrain import water_ndwi
from src.comp.effis import effis_mask_retry
R = Path('research'); MM = [R / f'hls-c1mm-final-s{s}' for s in (1, 2, 3, 4, 5)]
for title, d, cache in (('FLOGA', 'external/floga_hls', R / 'floga-test-v1'), ('EMS', 'external/ems_hls', R / 'ems-test-v1')):
    man, X, Y, V, L = load_dir(d, 'manifest_test.json', layers=('fmask',)); D = Path(d); E = []
    for w in man:
        p = D / f"{w['name']}.effis.tif"
        if not p.exists():
            with rasterio.open(D / f"{w['name']}_merged.tif") as r: crs, tr, sh, prof = r.crs, r.transform, r.shape, r.profile
            m = effis_mask_retry(crs, tr, sh, int(w['item_date'][:4]))
            with rasterio.open(p, 'w', driver='GTiff', height=sh[0], width=sh[1], count=1, dtype='uint8', crs=crs, transform=tr, compress='deflate') as o: o.write(m.astype(np.uint8)[None])
        E.append(rasterio.open(p).read(1).astype(bool))
    E = np.stack(E); W = np.stack([water_ndwi(f, x[1], x[3]) for f, x in zip(L['fmask'], X)])
    P = np.where(W, 0, c1_probs(X, MM, cache / 'c1mm.npy'))
    sc = Scorer(Y, V); so, se = sc.stats(P, 0.5), sc.stats(E.astype(np.float32), 0.5); pt, lo, hi, v = sc.compare(so, se)
    def pr(B): B = B & V; tp = (B & Y).sum(); return tp / max((Y & V).sum(), 1), tp / max(B.sum(), 1)
    (rc, pc), (re_, pe) = pr(P >= 0.5), pr(E)
    print(f'{title} ({len(Y)} окон, окон без EFFIS {int((~E.any((1, 2))).sum())}): C1-MM {sc.iou(so):.4f} (полнота {rc:.3f}, точность {pc:.3f}) | '
          f'EFFIS {sc.iou(se):.4f} (полнота {re_:.3f}, точность {pe:.3f}) | мы − EFFIS {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}', flush=True)
