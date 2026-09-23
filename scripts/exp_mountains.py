"""SPEC-73: горы — окна FLOGA на HLS v2 (external/floga_hls). Нулевой выстрел C1, C1-F, Prithvi, смеси; IoU гари при 0.5
с 95 % бутстрепом по окнам; разбивка по уклону пикселя (< 10°, 10–20°, 20–30°, > 30°) и по году."""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import load_dir, c1_probs, prithvi_probs, Scorer
R = Path('research'); S = (1, 2, 3, 4, 5); C = R / 'floga-hls-v1'
man, X, Y, V, ex = load_dir('external/floga_hls'); sl = ex['slope']
print(f'окон {len(Y)} | доля гари {(Y & V).sum() / V.sum():.4f} | уклон гари, медиана {np.nanmedian(sl[Y & V]):.1f}°')
M = {'C1 (лес HLS)': c1_probs(X, [R / f'hls-c1-final-s{s}' for s in S], C / 'c1.npy'), 'Prithvi': prithvi_probs(X, C / 'prithvi.npy')}
if all((R / f'hls-c1f-final-s{s}/model.pt').exists() for s in S): M['C1-F (лес + свежий)'] = c1_probs(X, [R / f'hls-c1f-final-s{s}' for s in S], C / 'c1f.npy')
M['смесь пополам'] = (M['C1 (лес HLS)'] + M['Prithvi']) / 2
sel = R / 'fresh-mix-select.json'
if sel.exists():
    j = json.load(open(sel)); M[f"смесь SPEC-74 (w={j['w']})"] = j['w'] * M['C1 (лес HLS)'] + (1 - j['w']) * M['Prithvi']
sc = Scorer(Y, V); st = {k: sc.stats(p, 0.5) for k, p in M.items()}
for k, s in st.items():
    lo, hi = sc.ci(s); print(f'{k:28s} IoU гари {sc.iou(s):.4f}  95 % [{lo:.4f}, {hi:.4f}]')
pt, lo, hi, v = sc.compare(st['C1 (лес HLS)'], st['Prithvi']); print(f'C1 − Prithvi: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}')
print('\nпо уклону пикселя (IoU гари при 0.5):')
bins = [(0, 10), (10, 20), (20, 30), (30, 90)]
for a, b in bins:
    pm = (sl >= a) & (sl < b); s_ = Scorer(Y, V, n_boot=1, pixel_mask=pm)
    print(f'  {a:2d}–{b:2d}°  пикс. гари {int((Y & V & pm).sum()):8d} | ' + ' | '.join(f'{k.split(" (")[0]} {s_.iou(s_.stats(p, 0.5)):.4f}' for k, p in M.items()))
yr = np.array([w['year'] for w in man])
for y in sorted(set(yr)):
    i = np.where(yr == y)[0]; print(f'  {y}: окон {len(i)} | ' + ' | '.join(f'{k.split(" (")[0]} {sc.iou(s, i):.4f}' for k, s in st.items()))
