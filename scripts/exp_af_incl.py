"""AF: застройка и вода в обучении и в предсказании (SPEC-33 кэш).
Разметка ставит огонь на покровах 50 и 80 (318 пикселей на 336 чипах), а мы
исключали их и из выборки (eligible), и из выхода — модель там слепа
(p = 0.000). Пять фолдов, рецепт жёстких отрицательных соседа, два варианта
выборки: eligible (как было) и valid (с 50/80). Замер — пул OOF, лучшая граница
по сетке для обоих одинаково; предсказание — с исключением и без."""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'research/af-hard-v2')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.af import f1, NAMES, train
sys.path.insert(0, str(Path.home() / 'fires'))
from scripts.hypothesis_lab import choose_background
chips = sorted(p.stem for p in (ROOT / 'hard_probabilities').glob('*.npz'))
folds = [chips[i::5] for i in range(5)]
Z = {c: np.load(ROOT / 'cache' / f'{c}.npz') for c in chips}
def build(ids, seed, mask_key, miner=None):
    rng = np.random.default_rng(seed); xs, ys = [], []
    for c in ids:
        z = Z[c]; x, y, m = z['x'], z['y'].astype(bool), z[mask_key].astype(bool)
        fire = np.flatnonzero(y & m); back = np.flatnonzero(~y & m)
        scores = None if miner is None else miner.predict_proba(x)[:, 1]
        take = np.r_[fire, choose_background(back, rng, scores)]
        xs.append(x[take]); ys.append(y[take].astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys)
GRID = np.r_[np.arange(0.2, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)]
LC = np.concatenate([Z[c]['x'][:, NAMES.index('landcover')] for c in chips]); EXCL = np.isin(LC, (50, 80))
Y = np.concatenate([Z[c]['y'].astype(bool) for c in chips]); OK = np.concatenate([Z[c]['valid'].astype(bool) for c in chips])
for mask_key in ('eligible', 'valid'):
    t0 = time.time(); P = np.zeros(Y.size, np.float32); pos = 0
    for k in range(5):
        fit = [c for j, f in enumerate(folds) if j != k for c in f]
        x, y = build(fit, 20260918, mask_key); base = train(x, y, 20260918)
        x, y = build(fit, 20260918, mask_key, base); m = train(x, y, 20260918)
        for c in folds[k]:
            i = chips.index(c) * 65536; P[i:i + 65536] = m.predict_proba(Z[c]['x'])[:, 1]
    for name, mask in (('без 50/80 на выходе', OK & ~EXCL), ('с 50/80 на выходе', OK)):
        fbest, cbest = max((f1(Y, (P >= c) & mask)['f1'], float(c)) for c in GRID)
        r = f1(Y, (P >= cbest) & mask)
        print(f'выборка {mask_key:8s} | {name:20s} | F1 {fbest:.4f} @ {cbest:.3f}  tp {r["tp"]} fp {r["fp"]} fn {r["fn"]}  | огонь на 50/80 пойман: {int((Y & EXCL & (P >= cbest) & mask).sum())}/{int((Y & EXCL).sum())}  ({time.time()-t0:.0f}с)', flush=True)
    np.save(f'/tmp/af_oof_{mask_key}.npy', P)
