"""SPEC-85: сценарий «первый пролёт». Оптика — C1-MM на первом снимке Sentinel-2 после начала пожара (плохие SCL → «не гарь»,
вода SCL 6 и NDWI > 0); оптика + радар — на плохих пикселях решает S1-M (3 сида). H1 — IoU гари по всем пикселям с меткой ≥ 0."""
import json, sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import c1_probs, Scorer
R = Path('research'); MM = [R / f'hls-c1mm-final-s{s}' for s in (1, 2, 3, 4, 5)]; SM = [R / f's1m-final-s{s}' for s in (1, 2, 3)]
Ys, Vs, Po, Pf, Bad, Ps1, hrs = [], [], [], [], [], [], []
for title, d in (('EMS', 'external/ems_hls'), ('FLOGA', 'external/floga_hls')):
    D = Path(d); man = [w for w in json.load(open(D / 'manifest_test_fp.json'))['windows'] if (D / f"{w['name']}.fp.s1.tif").exists()]
    rd = lambda w, k: rasterio.open(D / f"{w['name']}.{k}.tif").read()
    Y = np.stack([rd(w, 'mask')[0] == 1 for w in man]); V = np.stack([rd(w, 'mask')[0] >= 0 for w in man])
    XS = np.stack([rd(w, 'fp.s2').astype(np.float32) for w in man]); S = np.stack([rd(w, 'fp.s2scl')[0] for w in man])
    X1 = np.stack([rd(w, 'fp.s1').astype(np.float32) for w in man]); ok1 = (X1 != -9999).all(1); X1[np.broadcast_to(~ok1[:, None], X1.shape)] = 0
    bad = np.isin(S, [0, 1, 3, 8, 9, 10]) | (XS == -9999).any(1); XS[np.broadcast_to(bad[:, None], XS.shape)] = 0
    nd = (XS[:, 1] - XS[:, 3]) / (XS[:, 1] + XS[:, 3] + 1e-6); wat = (S == 6) & (nd > 0)
    po = np.where(bad | wat, 0, c1_probs(XS, MM, R / f'fp-{title.lower()}-v1' / 'c1mm.npy'))
    ps = np.where(ok1 & ~wat, c1_probs(X1, SM, R / f'fp-{title.lower()}-v1' / 's1m.npy'), 0)
    pf = np.where(bad, ps, po)
    sc = Scorer(Y, V); a, b, c = sc.stats(po, 0.5), sc.stats(pf, 0.5), sc.stats(ps, 0.5); pt, lo, hi, v = sc.compare(b, a)
    h = [json.load(open(D / f"{w['name']}.fp.json"))['hours_after_start'] for w in man]
    cloud = (bad & V).sum() / V.sum(); su = Scorer(Y, V & bad[:, :, :], n_boot=1)
    print(f'{title} ({len(Y)} окон, первый пролёт через {np.median(h):.0f} ч (медиана), плохих пикселей {cloud:.3f}): оптика {sc.iou(a):.4f} | '
          f'оптика + радар {sc.iou(b):.4f} | радар один {sc.iou(c):.4f} | +радар − оптика {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v} | '
          f'под облаками: оптика {su.iou(su.stats(po, 0.5)):.4f}, радар {su.iou(su.stats(pf, 0.5)):.4f}', flush=True)
    Ys.append(Y); Vs.append(V); Po.append(po); Pf.append(pf)
Y, V = np.concatenate(Ys), np.concatenate(Vs); sc = Scorer(Y, V); a, b = sc.stats(np.concatenate(Po), 0.5), sc.stats(np.concatenate(Pf), 0.5); pt, lo, hi, v = sc.compare(b, a)
print(f'H1 (оба набора, {len(Y)} окон): оптика {sc.iou(a):.4f} | оптика + радар {sc.iou(b):.4f} | разница {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
