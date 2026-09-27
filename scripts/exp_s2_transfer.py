"""SPEC-83 ч. А: C1-MM на Sentinel-2 L2A (двойники) против HLS на общих валидных пикселях; проверки FLOGA и EMS.
Маски: HLS — «Fmask и NDWI»; L2A — плохие SCL {0,1,3,8,9,10}, вода SCL 6 и NDWI > 0. Порог 0.5. Критерий спеки."""
import sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, Scorer
from src.comp.terrain import water_ndwi
R = Path('research'); MM = [R / f'hls-c1mm-final-s{s}' for s in (1, 2, 3, 4, 5)]; ok_all = True
for title, d, cache in (('FLOGA', 'external/floga_hls', R / 'floga-s2-v1'), ('EMS', 'external/ems_hls', R / 'ems-s2-v1')):
    man, X, Y, V, L = load_dir(d, 'manifest_s2.json', layers=('fmask', 's2scl')); D = Path(d)
    XS = np.stack([rasterio.open(D / f"{w['name']}.s2.tif").read().astype(np.float32) for w in man])
    S = L['s2scl']; badS = np.isin(S, [0, 1, 3, 8, 9, 10]) | (XS == -9999).any(1); XS[np.broadcast_to(badS[:, None], XS.shape)] = 0
    nd = (XS[:, 1] - XS[:, 3]) / (XS[:, 1] + XS[:, 3] + 1e-6); WS = (S == 6) & (nd > 0)
    WH = np.stack([water_ndwi(f, x[1], x[3]) for f, x in zip(L['fmask'], X)])
    VC = V & ~badS
    PH = np.where(WH, 0, c1_probs(X, MM, cache / 'hls.npy')); PS = np.where(WS, 0, c1_probs(XS, MM, cache / 's2.npy'))
    sc = Scorer(Y, VC); h, s = sc.stats(PH, 0.5), sc.stats(PS, 0.5); pt, lo, hi, v = sc.compare(s, h)
    good = pt >= -0.01 and lo >= -0.03; ok_all &= good
    print(f'{title} ({len(Y)} окон, общих валидных {VC.sum() / V.sum():.3f} от HLS): HLS {sc.iou(h):.4f} | L2A {sc.iou(s):.4f} | L2A − HLS {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {"годен" if good else "НЕ годен"}', flush=True)
print('ВЕРДИКТ SPEC-83 ч. А:', 'перенос почти бесплатный' if ok_all else 'перенос не бесплатный')
