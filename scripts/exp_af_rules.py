"""AF: два априора против разметки (из кэша SPEC-33). (1) Исключение застройки и
воды из предсказаний — разметка там огонь ставит (289 + 29 пикселей, все
пропущены). (2) Ночь: 30 % пропусков против 4 % днём — отдельная граница."""
import sys, numpy as np
from pathlib import Path
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'research/af-hard-v2')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.af import f1, NAMES
chips = sorted(p.stem for p in (ROOT / 'hard_probabilities').glob('*.npz'))
PH, Y, OK, EL, LC, NIGHT = [], [], [], [], [], []
for c in chips:
    z = np.load(ROOT / 'cache' / f'{c}.npz'); x = z['x']
    PH.append(np.load(ROOT / 'hard_probabilities' / f'{c}.npz')['p']); Y.append(z['y'].astype(bool)); OK.append(z['valid'].astype(bool)); EL.append(z['eligible'].astype(bool))
    LC.append(x[:, NAMES.index('landcover')]); NIGHT.append(x[:, NAMES.index('solar_zenith')] > 90)
PH, Y, OK, EL, LC, NIGHT = map(np.concatenate, (PH, Y, OK, EL, LC, NIGHT))
n = len(chips); per = Y.size // n
EXCL = np.isin(LC, (50, 80))
print(f'valid включает исключение покрова? valid&excl = {int((OK & EXCL).sum())}; eligible&excl = {int((EL & EXCL).sum())}; огня на 50/80: {int((Y & EXCL).sum())}')
GRID = np.r_[np.arange(0.2, 0.95, 0.05), np.arange(0.95, 0.999, 0.005)]
def best(p, mask, i=slice(None)):
    return max((f1(Y[i], ((p[i] >= c) & mask[i]))['f1'], float(c)) for c in GRID)
print('как в продакшене (valid, без 50/80):', '%.4f @ %.3f' % best(PH, OK & ~EXCL))
print('без исключения покрова (valid):     ', '%.4f @ %.3f' % best(PH, OK))
m50 = OK & ~np.isin(LC, (80,)); print('вернуть только застройку (50):        ', '%.4f @ %.3f' % best(PH, m50))
print(f'p жёсткой модели на огне в застройке: медиана {np.median(PH[Y & (LC == 50)]):.3f}, доля ≥0.35: {(PH[Y & (LC == 50)] >= 0.35).mean():.2f}')
# ночь/день раздельные границы, честно на половинах
rng = np.random.default_rng(0); perm = rng.permutation(n); halves = (perm[:n // 2], perm[n // 2:])
def idx(h): return np.concatenate([np.arange(i * per, (i + 1) * per) for i in h])
def pred_split(cd, cn, mask, i): return ((PH[i] >= np.where(NIGHT[i], cn, cd)) & mask[i])
for name, mask in (('без 50/80', OK & ~EXCL), ('с 50/80', OK)):
    print(f'--- {name}')
    fd, cd = best(PH, mask & ~NIGHT); fn_, cn = best(PH, mask & NIGHT)
    print(f'  оптимум границ: день {cd:.3f} (F1 день {fd:.4f}), ночь {cn:.3f} (F1 ночь {fn_:.4f})')
    for fit_h, ev_h in (halves, halves[::-1]):
        fi, ei = idx(fit_h), idx(ev_h)
        c1 = best(PH, mask, fi)[1]
        cd_, cn_ = max((f1(Y[fi], pred_split(cd0, cn0, mask, fi))['f1'], cd0, cn0) for cd0 in GRID[::2] for cn0 in GRID[::2])[1:]
        one = f1(Y[ei], (PH[ei] >= c1) & mask[ei])['f1']; two = f1(Y[ei], pred_split(cd_, cn_, mask, ei))['f1']
        print(f'  честно: одна граница {c1:.3f} → {one:.4f}; день {cd_:.2f}/ночь {cn_:.2f} → {two:.4f} ({two-one:+.4f})')
