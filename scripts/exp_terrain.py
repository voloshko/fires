"""SPEC-76/77: маска воды (бит 5 Fmask) и C-коррекция освещённости склона. Разработка — FLOGA 186 окон (печать),
проверка — горный свежий набор external/hls_fresh3/manifest_mount.json (решает). Основная модель — C1-F."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, prithvi_probs, Scorer
from src.comp.terrain import water, cos_incidence, c_correct, slope_deg
R = Path('research'); S = (1, 2, 3, 4, 5)
SETS = {'разработка: FLOGA (Греция)': ('external/floga_hls', 'manifest.json', R / 'floga-hls-v1'),
        'проверка: горы MTBS 2022+ (запад США)': ('external/hls_fresh3', 'manifest_mount.json', R / 'fresh3-mount-v1')}
for title, (d, mf, cache) in SETS.items():
    man, X, Y, V, L = load_dir(d, mf, layers=('fmask', 'sza', 'saa', 'dem')); W = np.stack([water(f) for f in L['fmask']])
    SL = np.stack([slope_deg(z.astype(np.float64)) for z in L['dem']]).astype(np.float32)
    XC, used = X.copy(), 0
    for i in range(len(X)):
        ci = cos_incidence(L['dem'][i].astype(np.float64), L['sza'][i] * 0.01, L['saa'][i] * 0.01)
        XC[i], u = c_correct(X[i], ci, float(np.mean(L['sza'][i]) * 0.01), V[i] & ~W[i], SL[i]); used += len(u) > 0
    print(f'\n=== {title}: окон {len(Y)}, доля гари {(Y & V).sum() / V.sum():.4f}, вода {W[V].mean():.3f}, C-коррекция применена в {used} окнах', flush=True)
    base = {'C1-F': c1_probs(X, [R / f'hls-c1f-final-s{s}' for s in S], cache / 'c1f.npy'), 'C1': c1_probs(X, [R / f'hls-c1-final-s{s}' for s in S], cache / 'c1.npy'),
            'Prithvi': prithvi_probs(X, cache / 'prithvi.npy')}
    cc = {'C1-F': c1_probs(XC, [R / f'hls-c1f-final-s{s}' for s in S], cache / 'c1f_cc.npy'), 'C1': c1_probs(XC, [R / f'hls-c1-final-s{s}' for s in S], cache / 'c1_cc.npy'),
          'Prithvi': prithvi_probs(XC, cache / 'prithvi_cc.npy')}
    sc = Scorer(Y, V); st = {}
    for k in base:
        st[(k, 'как есть')] = sc.stats(base[k], 0.5); st[(k, 'вода')] = sc.stats(np.where(W, 0, base[k]), 0.5); st[(k, 'вода + C')] = sc.stats(np.where(W, 0, cc[k]), 0.5)
        print(f'{k:8s} ' + ' | '.join(f'{v} {sc.iou(st[(k, v)]):.4f}' for v in ('как есть', 'вода', 'вода + C')))
    for k in base:
        pt, lo, hi, v = sc.compare(st[(k, 'вода')], st[(k, 'как есть')]); print(f'  SPEC-76 {k:8s} вода − как есть: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
        pt, lo, hi, v = sc.compare(st[(k, 'вода + C')], st[(k, 'вода')]); print(f'  SPEC-77 {k:8s} C-коррекция (на фоне маски воды): {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
    print('  по уклону, C1-F (полнота / точность):')
    for a, b in ((0, 10), (10, 20), (20, 30), (30, 90)):
        pm = V & (SL >= a) & (SL < b); row = []
        for name, P in (('как есть', base['C1-F']), ('вода', np.where(W, 0, base['C1-F'])), ('вода + C', np.where(W, 0, cc['C1-F']))):
            B = (P >= 0.5) & pm; tp = (B & Y).sum(); row.append(f'{name} {tp / max((Y & pm).sum(), 1):.3f}/{tp / max(B.sum(), 1):.3f}')
        print(f'    {a:2d}–{b:2d}°  ' + ' | '.join(row))
    if 'проверка' in title:
        print('ВЕРДИКТЫ (C1-F на проверке): SPEC-76', sc.compare(st[('C1-F', 'вода')], st[('C1-F', 'как есть')])[3], '| SPEC-77', sc.compare(st[('C1-F', 'вода + C')], st[('C1-F', 'вода')])[3])
