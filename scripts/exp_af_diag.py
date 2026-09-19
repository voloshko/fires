"""AF: диагностика ошибок и смесь двух моделей соседа (SPEC-33 кэш).
Из кэша OOF-вероятностей research/af-hard-v2 (случайный фон / жёсткие
отрицательные, 336 чипов) без переобучения: концентрация ошибок по чипам,
покрову, дню/ночи и расстоянию до очага; смесь вероятностей двух моделей."""
import sys, glob, json, numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt as edt
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'research/af-hard-v2')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.af import f1, NAMES
chips = sorted(p.stem for p in (ROOT / 'random_probabilities').glob('*.npz'))
PR, PH, Y, OK, LC, NIGHT, DIST = [], [], [], [], [], [], []
for c in chips:
    z = np.load(ROOT / 'cache' / f'{c}.npz'); x, y, ok = z['x'], z['y'].astype(bool), z['valid'].astype(bool)
    PR.append(np.load(ROOT / 'random_probabilities' / f'{c}.npz')['p']); PH.append(np.load(ROOT / 'hard_probabilities' / f'{c}.npz')['p'])
    Y.append(y); OK.append(ok); LC.append(x[:, NAMES.index('landcover')]); NIGHT.append(x[:, NAMES.index('solar_zenith')] > 90)
    ym = y.reshape(256, 256); DIST.append((edt(~ym) if ym.any() else np.full((256, 256), 999.0)).reshape(-1))
PR, PH, Y, OK, LC, NIGHT, DIST = map(np.concatenate, (PR, PH, Y, OK, LC, NIGHT, DIST))
n = len(chips); per = Y.size // n
def best(p):
    grid = np.r_[np.arange(0.3, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)]
    return max((f1(Y, (p >= c) & OK)['f1'], float(c)) for c in grid)
fr, cr = best(PR); fh, ch = best(PH); print(f'случайный фон: F1 {fr:.4f} @ {cr:.3f}   жёсткие: F1 {fh:.4f} @ {ch:.3f}')
for w in (0.2, 0.3, 0.5, 0.7):
    fm, cm = best((1 - w) * PH + w * PR); print(f'смесь {1-w:.1f}·жёсткие + {w:.1f}·случайный: F1 {fm:.4f} @ {cm:.3f}')
# честность смеси: половины чипов
GRID = np.r_[np.arange(0.3, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)]
rng = np.random.default_rng(0); perm = rng.permutation(n); halves = (perm[:n // 2], perm[n // 2:])
def idx(h): return np.concatenate([np.arange(i * per, (i + 1) * per) for i in h])
def fit_mix(i):
    return max(((f1(Y[i], ((((1 - w) * PH[i] + w * PR[i]) >= c) & OK[i]))['f1'], c, w) for w in (0.0, 0.2, 0.3, 0.5) for c in GRID))
for fit_h, ev_h in (halves, halves[::-1]):
    fi, ei = idx(fit_h), idx(ev_h)
    _, cbest, wbest = fit_mix(fi)
    ev_mix = f1(Y[ei], ((((1 - wbest) * PH[ei] + wbest * PR[ei]) >= cbest) & OK[ei]))['f1']
    ch_fit = max((f1(Y[fi], (PH[fi] >= c) & OK[fi])['f1'], c) for c in GRID)[1]
    ev_hard = f1(Y[ei], (PH[ei] >= ch_fit) & OK[ei])['f1']
    print(f'честно: w={wbest}, граница {cbest:.3f} → смесь {ev_mix:.4f} против жёстких {ev_hard:.4f} ({ev_mix-ev_hard:+.4f})')
# диагностика жёсткой модели при её границе
p = (PH >= ch) & OK; fp = p & ~Y; fn = Y & ~p; tp = Y & p
print(f'\nжёсткие @ {ch:.3f}: tp {tp.sum()} fp {fp.sum()} fn {fn.sum()}')
chip_fp = fp.reshape(n, per).sum(1); chip_fn = fn.reshape(n, per).sum(1); chip_y = Y.reshape(n, per).sum(1)
order = np.argsort(-(chip_fp + chip_fn))
print('худшие чипы: чип, огня, fp, fn')
for i in order[:10]: print(f'  {chips[i]} {chip_y[i]:5d} {chip_fp[i]:5d} {chip_fn[i]:5d}')
print(f'пять худших держат fp {chip_fp[order[:5]].sum()/max(1,fp.sum()):.1%} и fn {chip_fn[order[:5]].sum()/max(1,fn.sum()):.1%}')
print('по покрову: класс, огня, fp, fn'); 
for k in np.unique(LC[Y | fp]): m = LC == k; print(f'  {int(k):3d} {int((Y&m).sum()):6d} {int((fp&m).sum()):5d} {int((fn&m).sum()):5d}')
print(f'ночь: огня {int((Y&NIGHT).sum())}, fp {int((fp&NIGHT).sum())}, fn {int((fn&NIGHT).sum())} | день: огня {int((Y&~NIGHT).sum())}, fp {int((fp&~NIGHT).sum())}, fn {int((fn&~NIGHT).sum())}')
print('fp по расстоянию до ближайшего истинного очага:'); 
for lo, hi in ((0, 1.5), (1.5, 3), (3, 6), (6, 12), (12, 998), (998, 1e9)):
    m = (DIST >= lo) & (DIST < hi); print(f'  {lo:>4}-{hi if hi < 1e9 else "нет огня на чипе":>5}: fp {int((fp&m).sum()):5d}')
print(f'fn по уверенности: p<0.1 {int((fn&(PH<0.1)).sum())}, 0.1–0.3 {int((fn&(PH>=0.1)&(PH<0.3)).sum())}, ≥0.3 {int((fn&(PH>=0.3)).sum())}')
