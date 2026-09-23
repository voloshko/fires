"""SPEC-74: веса смеси и пороги подбираются на первом свежем наборе, проверяются на втором.
SPEC-75: C1-F (HLS + первый свежий набор) против C1 и Prithvi на втором свежем наборе. Выбор пишется в research/fresh-mix-select.json."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, prithvi_probs, Scorer
R = Path('research'); S = (1, 2, 3, 4, 5); T = [round(t, 2) for t in np.arange(0.30, 0.701, 0.05)]; W = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
m1, X1, Y1, V1, _ = load_dir('external/hls_fresh'); m2, X2, Y2, V2, _ = load_dir('external/hls_fresh2')
assert not {w['event_id'] for w in m1} & {w['event_id'] for w in m2}, 'наборы пересекаются'
C1a = c1_probs(X1, [R / f'hls-c1-final-s{s}' for s in S], R / 'hls-fresh-v1/c1_ensemble.npy'); P1 = prithvi_probs(X1, R / 'hls-fresh-v1/prithvi.npy')
C1b = c1_probs(X2, [R / f'hls-c1-final-s{s}' for s in S], R / 'hls-fresh2-v1/c1_ensemble.npy'); P2 = prithvi_probs(X2, R / 'hls-fresh2-v1/prithvi.npy')
s1, s2 = Scorer(Y1, V1), Scorer(Y2, V2)
# SPEC-74, выбор на первом
grid = {(w, t): s1.iou(s1.stats(w * C1a + (1 - w) * P1, t)) for w in W for t in T}; (w_, t_) = max(grid, key=grid.get)
tc = max(T, key=lambda t: s1.iou(s1.stats(C1a, t))); tp = max(T, key=lambda t: s1.iou(s1.stats(P1, t)))
ic, ip = s1.iou(s1.stats(C1a, tc)), s1.iou(s1.stats(P1, tp)); single = ('C1', tc) if ic >= ip else ('Prithvi', tp)
print(f'первый набор: смесь w={w_} (доля C1), t={t_}: {grid[(w_, t_)]:.4f} | C1 t={tc}: {ic:.4f} | Prithvi t={tp}: {ip:.4f} → лучшая одиночная {single}')
print('  профиль по w при лучшем t для каждого w:', {w: round(max(grid[(w, t)] for t in T), 4) for w in W})
json.dump(dict(w=w_, t=t_, c1_t=tc, prithvi_t=tp, single=single, fresh1_mix=grid[(w_, t_)]), open(R / 'fresh-mix-select.json', 'w'), indent=1)
# SPEC-74, проверка на втором
mix2 = s2.stats(w_ * C1b + (1 - w_) * P2, t_); sing2 = s2.stats(C1b if single[0] == 'C1' else P2, single[1])
print(f'\nвторой набор ({len(Y2)} окон): смесь {s2.iou(mix2):.4f} | {single[0]} ({single[1]}) {s2.iou(sing2):.4f}')
pt, lo, hi, v = s2.compare(mix2, sing2); print(f'SPEC-74 смесь − лучшая одиночная: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
a, b, c = s2.stats(C1b, 0.5), s2.stats(P2, 0.5), s2.stats((C1b + P2) / 2, 0.5)
print(f'повтор SPEC-72 на втором (0.5): C1 {s2.iou(a):.4f} | Prithvi {s2.iou(b):.4f} | смесь пополам {s2.iou(c):.4f}; C1 − Prithvi {s2.compare(a, b)}')
# SPEC-75
if all((R / f'hls-c1f-final-s{s}/model.pt').exists() for s in S):
    F2 = c1_probs(X2, [R / f'hls-c1f-final-s{s}' for s in S], R / 'hls-fresh2-v1/c1f_ensemble.npy'); f = s2.stats(F2, 0.5)
    print(f'\nSPEC-75 на втором (0.5): C1-F {s2.iou(f):.4f} | C1 {s2.iou(a):.4f} | Prithvi {s2.iou(b):.4f}')
    pt, lo, hi, v = s2.compare(f, a); print(f'H1 C1-F − C1: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
    pt, lo, hi, v = s2.compare(f, b); print(f'H2 C1-F − Prithvi: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
    typ = np.array([w['incid_type'] for w in m2])
    for t in sorted(set(typ)):
        i = np.where(typ == t)[0]; print(f'  {t:16s} окон {len(i):3d} | C1-F {s2.iou(f, i):.4f} | C1 {s2.iou(a, i):.4f} | Prithvi {s2.iou(b, i):.4f}')
    import rasterio
    names = json.load(open(R / 'biome-hls-f0/data_manifest.json'))['hls_val']; YV, VV = [], []
    for n in names:
        p = Path('external/hls_burn_scars/validation') / n; im = rasterio.open(p).read(); mk = rasterio.open(str(p).replace('_merged.tif', '.mask.tif')).read()[0]
        VV.append((im != -9999).all(0) & (mk >= 0)); YV.append(mk == 1)
    sv = Scorer(np.stack(YV), np.stack(VV), n_boot=1); EF = np.mean([np.load(R / f'hls-c1f-final-s{s}/probabilities.npy').astype(np.float32) for s in S], 0)
    print(f'здоровье на validation HLS (не решает): C1-F при 0.5 {sv.iou(sv.stats(EF, 0.5)):.4f} (C1 0.8704), при 0.7 {sv.iou(sv.stats(EF, 0.7)):.4f} (C1 0.8666)')
else:
    print('\nSPEC-75: веса C1-F ещё не готовы')
