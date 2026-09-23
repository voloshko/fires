"""SPEC-78: маска воды «бит 5 Fmask и NDWI > 0» у C1-F против отсутствия маски и одной Fmask; повтор SPEC-77 (C-коррекция поверх
новой маски). Проверка — горный набор-4 external/hls_fresh4/manifest_mount.json. Вторично — C1, Prithvi, по уклону."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, prithvi_probs, Scorer
from src.comp.terrain import water, water_ndwi, cos_incidence, c_correct, slope_deg
R = Path('research'); S = (1, 2, 3, 4, 5)
# SPEC-79 переиспользует замер: exp_water2.py <каталог> <манифест> <кэш>; по умолчанию — горный набор-4 SPEC-78
D, MF, cache = (sys.argv[1], sys.argv[2], Path(sys.argv[3])) if len(sys.argv) > 3 else ('external/hls_fresh4', 'manifest_mount.json', R / 'fresh4-mount-v1')
man, X, Y, V, L = load_dir(D, MF, layers=('fmask', 'sza', 'saa', 'dem'))
WF = np.stack([water(f) for f in L['fmask']]); WN = np.stack([water_ndwi(f, x[1], x[3]) for f, x in zip(L['fmask'], X)])
SL = np.stack([slope_deg(z.astype(np.float64)) for z in L['dem']]).astype(np.float32)
states = sorted({w.get('country') or w['event_id'][:2] for w in man})
print(f'окон {len(Y)} | штаты {states} | доля гари {(Y & V).sum() / V.sum():.4f} | вода Fmask {WF[V].mean():.3f}, Fmask и NDWI {WN[V].mean():.3f} | гарь под Fmask-водой {int((WF & Y).sum())}, под новой {int((WN & Y).sum())}')
XC = X.copy()
for i in range(len(X)):
    ci = cos_incidence(L['dem'][i].astype(np.float64), L['sza'][i] * 0.01, L['saa'][i] * 0.01)
    XC[i], _ = c_correct(X[i], ci, float(np.mean(L['sza'][i]) * 0.01), V[i] & ~WN[i], SL[i])
M = {'C1-F': (R / 'hls-c1f-final-s{}', 'c1f'), 'C1': (R / 'hls-c1-final-s{}', 'c1')}
base = {k: c1_probs(X, [Path(str(p).format(s)) for s in S], cache / f'{c}.npy') for k, (p, c) in M.items()}; base['Prithvi'] = prithvi_probs(X, cache / 'prithvi.npy')
cc = {k: c1_probs(XC, [Path(str(p).format(s)) for s in S], cache / f'{c}_cc.npy') for k, (p, c) in M.items()}; cc['Prithvi'] = prithvi_probs(XC, cache / 'prithvi_cc.npy')
sc = Scorer(Y, V); st = {}
for k in base:
    st[k, 'нет'] = sc.stats(base[k], 0.5); st[k, 'Fmask'] = sc.stats(np.where(WF, 0, base[k]), 0.5); st[k, 'Fmask+NDWI'] = sc.stats(np.where(WN, 0, base[k]), 0.5)
    st[k, 'Fmask+NDWI+C'] = sc.stats(np.where(WN, 0, cc[k]), 0.5)
    print(f'{k:8s} ' + ' | '.join(f'{v} {sc.iou(st[k, v]):.4f}' for v in ('нет', 'Fmask', 'Fmask+NDWI', 'Fmask+NDWI+C')))
V_ = {}
for k in base:
    for name, a, b in (('главное: Fmask+NDWI − нет', 'Fmask+NDWI', 'нет'), ('второе: Fmask+NDWI − Fmask', 'Fmask+NDWI', 'Fmask'), ('повтор SPEC-77: +C − Fmask+NDWI', 'Fmask+NDWI+C', 'Fmask+NDWI')):
        pt, lo, hi, v = sc.compare(st[k, a], st[k, b]); print(f'  {k:8s} {name}: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}'); V_[k, name] = v
print('  по уклону, C1-F (полнота / точность):')
for a_, b_ in ((0, 10), (10, 20), (20, 30), (30, 90)):
    pm = V & (SL >= a_) & (SL < b_); row = []
    for name, P in (('нет', base['C1-F']), ('Fmask', np.where(WF, 0, base['C1-F'])), ('Fmask+NDWI', np.where(WN, 0, base['C1-F'])), ('+C', np.where(WN, 0, cc['C1-F']))):
        B = (P >= 0.5) & pm; tp = (B & Y).sum(); row.append(f'{name} {tp / max((Y & pm).sum(), 1):.3f}/{tp / max(B.sum(), 1):.3f}')
    print(f'    {a_:2d}–{b_:2d}°  ' + ' | '.join(row))
print('ВЕРДИКТЫ C1-F:', ' | '.join(f'{n.split(":")[0]} {V_["C1-F", n]}' for n in ('главное: Fmask+NDWI − нет', 'второе: Fmask+NDWI − Fmask', 'повтор SPEC-77: +C − Fmask+NDWI')))
