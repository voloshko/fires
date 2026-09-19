"""AF: отдельная модель для ночных пикселей (solar_zenith > 90). Ночью пропущено
30 % огня против 4 % днём; у ночи другая физика (нет солнечного нагрева фона).
OOF на 5 фолдах из кэша SPEC-33: общая модель против «день + ночь раздельно»
и против общей модели с признаком ночи. Один протокол выбора границы."""
import sys, time, numpy as np
from pathlib import Path
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'research/af-hard-v2')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.af import f1, NAMES, train
from scripts.hypothesis_lab import choose_background
chips = sorted(p.stem for p in (ROOT / 'hard_probabilities').glob('*.npz')); folds = [chips[i::5] for i in range(5)]
Z = {c: np.load(ROOT / 'cache' / f'{c}.npz') for c in chips}; NI = NAMES.index('solar_zenith')
def build(ids, seed, sel=None, miner=None, extra=False):
    rng = np.random.default_rng(seed); xs, ys = [], []
    for c in ids:
        z = Z[c]; x, y, m = z['x'], z['y'].astype(bool), z['valid'].astype(bool)
        if sel is not None: m = m & sel(x)
        if extra: x = np.c_[x, (x[:, NI] > 90).astype(np.float32)]
        fire = np.flatnonzero(y & m); back = np.flatnonzero(~y & m)
        if not back.size: continue
        scores = None if miner is None else miner.predict_proba(x)[:, 1]
        take = np.r_[fire, choose_background(back, rng, scores)]; xs.append(x[take]); ys.append(y[take].astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys)
class Drop:
    """Обёртка: выкидывает константные столбцы (ночью I1–I3 = 0), иначе биннинг падает."""
    def __init__(self, m, keep): self.m, self.keep = m, keep
    def predict_proba(self, x): return self.m.predict_proba(x[:, self.keep])
def fit_drop(x, y):
    keep = np.flatnonzero(x.std(0) > 1e-6)
    from sklearn.ensemble import HistGradientBoostingClassifier
    lc = list(keep).index(NAMES.index('landcover')) if NAMES.index('landcover') in keep else None
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                       categorical_features=[lc] if lc is not None else None, random_state=20260918).fit(x[:, keep], y)
    return Drop(m, keep)
def hard(ids, sel=None, extra=False):
    x, y = build(ids, 20260918, sel, None, extra); b = fit_drop(x, y)
    x, y = build(ids, 20260918, sel, b, extra); return fit_drop(x, y)
Y = np.concatenate([Z[c]['y'].astype(bool) for c in chips]); OK = np.concatenate([Z[c]['valid'].astype(bool) for c in chips])
NIGHT = np.concatenate([Z[c]['x'][:, NI] > 90 for c in chips])
GRID = np.r_[np.arange(0.2, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)]
def report(name, P):
    fb, cb = max((f1(Y, (P >= c) & OK)['f1'], float(c)) for c in GRID); pr = (P >= cb) & OK
    rn = f1(Y[NIGHT], pr[NIGHT]); rd = f1(Y[~NIGHT], pr[~NIGHT])
    print(f'{name:34s} F1 {fb:.4f} @ {cb:.3f} | ночь F1 {rn["f1"]:.3f} (fn {rn["fn"]}) | день F1 {rd["f1"]:.3f}', flush=True)
t0 = time.time()
for name, mode in (('день и ночь раздельно', 'split'),):
    P = np.zeros(Y.size, np.float32)
    for k in range(5):
        fit = [c for j, f in enumerate(folds) if j != k for c in f]
        if mode == 'one': m = hard(fit); models = [(m, None)]
        elif mode == 'flag': m = hard(fit, extra=True); models = [(m, 'flag')]
        else: models = [(hard(fit, lambda x: x[:, NI] <= 90), 'day'), (hard(fit, lambda x: x[:, NI] > 90), 'night')]
        for c in folds[k]:
            i = chips.index(c) * 65536; x = Z[c]['x']; night = x[:, NI] > 90
            for m, kind in models:
                if kind == 'flag': p = m.predict_proba(np.c_[x, night.astype(np.float32)])[:, 1]; P[i:i+65536] = p
                elif kind is None: P[i:i+65536] = m.predict_proba(x)[:, 1]
                else:
                    sel = night if kind == 'night' else ~night
                    if sel.any(): P[i:i+65536][sel] = m.predict_proba(x[sel])[:, 1]
    report(name, P); print(f'  ({time.time()-t0:.0f}с)')
