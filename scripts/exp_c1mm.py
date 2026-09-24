"""SPEC-82: C1-MM (HLS + свежий MTBS + EMS и половина FLOGA ×2) против C1-F. H1 — проверка FLOGA, H2 — проверка EMS;
рецепт для гор (маска «Fmask и NDWI», порог 0.5). Здоровье, не решает: validation HLS и второй свежий MTBS."""
import json, sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, Scorer
from src.comp.terrain import water_ndwi
R = Path('research'); S = (1, 2, 3, 4, 5); M = {'C1-MM': 'hls-c1mm-final-s{}', 'C1-F': 'hls-c1f-final-s{}', 'C1-M': 'hls-c1m-final-s{}'}
VER = {}
for h, (d, mf, cache) in {'H1 проверка FLOGA': ('external/floga_hls', 'manifest_test.json', R / 'floga-test-v1'), 'H2 проверка EMS': ('external/ems_hls', 'manifest_test.json', R / 'ems-test-v1')}.items():
    man, X, Y, V, L = load_dir(d, mf, layers=('fmask',)); WN = np.stack([water_ndwi(f, x[1], x[3]) for f, x in zip(L['fmask'], X)])
    P = {k: np.where(WN, 0, c1_probs(X, [R / p.format(s) for s in S], cache / f"{k.lower().replace('-', '')}.npy")) for k, p in M.items()}
    sc = Scorer(Y, V); st = {k: sc.stats(p, 0.5) for k, p in P.items()}; pt, lo, hi, v = sc.compare(st['C1-MM'], st['C1-F']); p2, l2, h2_, v2 = sc.compare(st['C1-MM'], st['C1-M'])
    print(f"{h} ({len(Y)} окон): C1-MM {sc.iou(st['C1-MM']):.4f} | C1-F {sc.iou(st['C1-F']):.4f} | C1-M {sc.iou(st['C1-M']):.4f} | C1-MM − C1-F {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v} | C1-MM − C1-M {p2:+.4f}, [{l2:+.4f}, {h2_:+.4f}] → {v2}", flush=True); VER[h] = v
m2, X2, Y2, V2, _ = load_dir('external/hls_fresh2'); s2 = Scorer(Y2, V2, n_boot=1)
F2 = {k: c1_probs(X2, [R / p.format(s) for s in S], R / 'hls-fresh2-v1' / f"{k.lower().replace('-', '')}_ensemble.npy") for k, p in M.items()}
print('здоровье, свежий MTBS-2 (0.5, без маски):', ' | '.join(f'{k} {s2.iou(s2.stats(p, 0.5)):.4f}' for k, p in F2.items()))
names = json.load(open(R / 'biome-hls-f0/data_manifest.json'))['hls_val']; YV, VV = [], []
for n in names:
    f = Path('external/hls_burn_scars/validation') / n; im = rasterio.open(f).read(); mk = rasterio.open(str(f).replace('_merged.tif', '.mask.tif')).read()[0]
    VV.append((im != -9999).all(0) & (mk >= 0)); YV.append(mk == 1)
sv = Scorer(np.stack(YV), np.stack(VV), n_boot=1)
print('здоровье, validation HLS (0.5):', ' | '.join(f"{k} {sv.iou(sv.stats(np.mean([np.load(R / p.format(s) / 'probabilities.npy').astype(np.float32) for s in S], 0), 0.5)):.4f}" for k, p in M.items()))
print('ВЕРДИКТ SPEC-82:', 'смешанный резерв работает' if 'ХУЖЕ' not in VER.values() and 'ЛУЧШЕ' in VER.values() else 'не работает', VER)
