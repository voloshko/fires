"""SPEC-69: select — выбор рецепта и порога на отложенных тайлах training; final — ансамбль 5 сидов на validation
против Prithvi-EO-2.0 BurnScars (research/prithvi-hls-val-v1), бутстреп по сценам, вердикт по критерию спеки."""
import glob, json, sys, numpy as np, rasterio
from pathlib import Path
R = Path('research'); PRITHVI = 0.8622; SEEDS = (1, 2, 3, 4, 5)
def labels(names, split):
    Y, V = [], []
    for n in names:
        f = Path('external/hls_burn_scars') / split / n; img = rasterio.open(f).read(); m = rasterio.open(str(f).replace('_merged.tif', '.mask.tif')).read()[0]
        V.append((img != -9999).all(0) & (m >= 0)); Y.append(m == 1)
    return np.stack(Y), np.stack(V)
def iou(P, Y, V, thr): B = (P >= thr) & V; T = Y & V; return (B & T).sum() / (B | T).sum()
if sys.argv[1] == 'select':
    names = json.load(open(R / 'hls-beat-C0-inner/data_manifest.json'))['evaluation']; Y, V = labels(names, 'training')
    P = {c: np.load(R / f'hls-beat-{c}-inner/probabilities.npy').astype(np.float32) for c in ('C0', 'C1', 'C2')}
    s = {c: float(iou(p, Y, V, 0.5)) for c, p in P.items()}; best = max(s, key=s.get)
    win = best if s[best] - s['C0'] >= 0.003 else 'C0'
    grid = {round(t, 2): float(iou(P[win], Y, V, t)) for t in np.arange(0.30, 0.701, 0.05)}; thr = max(grid, key=grid.get)
    for c in s: print(f'{c}: IoU гари на отложенных {s[c]:.4f}')
    print('порог:', {k: round(v, 4) for k, v in grid.items()}); print(f'победитель {win}, порог {thr}')
    json.dump(dict(scores=s, winner=win, threshold=thr, grid=grid, holdout_scenes=len(names)), open(R / 'hls-beat-select.json', 'w'), indent=1)
else:
    sel = json.load(open(R / 'hls-beat-select.json')); thr = sel['threshold']
    names = json.load(open(R / 'biome-hls-f0/data_manifest.json'))['hls_val']; Y, V = labels(names, 'validation')
    for s in SEEDS: assert json.load(open(R / f'hls-beat-final-s{s}/data_manifest.json'))['evaluation'] == names
    Ps = [np.load(R / f'hls-beat-final-s{s}/probabilities.npy').astype(np.float32) for s in SEEDS]
    E = np.mean(Ps, 0); pr = np.load(R / 'prithvi-hls-val-v1/probabilities.npy').astype(np.float32)
    def row(name, P, t):
        B = (P >= t) & V; T = Y & V; tp = (B & T).sum(); fp = (B & ~T).sum(); fn = (~B & T).sum(); tn = (~B & ~Y & V).sum()
        ib = tp / (tp + fp + fn); ino = tn / (tn + fp + fn)
        per = np.mean([((b & y) & v).sum() / ((b | y) & v).sum() for b, y, v in zip(B, Y, V) if ((b | y) & v).sum()])
        print(f'{name:28s} IoU гари {ib:.4f} | mIoU {(ib + ino) / 2:.4f} | по сценам {per:.4f}'); return P >= t
    row('Prithvi-EO-2.0 (0.5)', pr, 0.5); row('наш SPEC-66 (0.5)', np.load(R / 'biome-hls-f0/probabilities_hls_val.npy').astype(np.float32), 0.5)
    for s, P in zip(SEEDS, Ps): row(f'{sel["winner"]} сид {s} ({thr})', P, thr)
    bo = row(f'{sel["winner"]} ансамбль 5 ({thr})', E, thr); bp = pr >= 0.5
    rng = np.random.default_rng(0); n = len(Y)
    def ib(B, i): return (B[i] & Y[i] & V[i]).sum() / ((B[i] | Y[i]) & V[i]).sum()
    d = np.array([ib(bo, i) - ib(bp, i) for i in (rng.integers(0, n, n) for _ in range(1000))]); lo, hi = np.percentile(d, [2.5, 97.5])
    ours = ib(bo, np.arange(n)); print(f'наш − Prithvi: {ours - ib(bp, np.arange(n)):+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}]')
    v = 'ОБОГНАЛИ (implemented)' if ours > PRITHVI and lo > 0 else 'ДОГНАЛИ (partial)' if ours >= PRITHVI else 'НЕ ОБОГНАЛИ (rejected)'
    print('ВЕРДИКТ SPEC-69:', v)
