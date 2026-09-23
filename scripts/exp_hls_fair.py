"""SPEC-71: (а) Prithvi-EO-2.0 BurnScars с теми же 8 преобразованиями при инференсе, что у нас (CPU);
(б) где Prithvi нас бьёт: разница IoU гари по сценам против площади гари и доли нодаты, ошибки у кромки и
в глубине, находка мелких пятен. Аргумент — путь к нашим вероятностям на validation 264."""
import glob, json, sys, numpy as np, rasterio, torch, yaml
from pathlib import Path
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); from src.comp.hls import tta8
torch.set_num_threads(12)
HLS = Path('external/hls_burn_scars'); EXT = Path('external/prithvi-eo2-300m-burnscars'); R = Path('research')
names = json.load(open(R / 'biome-hls-f0/data_manifest.json'))['hls_val']; X, Y, V = [], [], []
for n in names:
    f = HLS / 'validation' / n; img = rasterio.open(f).read().astype(np.float32); m = rasterio.open(str(f).replace('_merged.tif', '.mask.tif')).read()[0]
    v = (img != -9999).all(0) & (m >= 0); img[:, ~v] = 0; X.append(img); Y.append(m == 1); V.append(v)
Y, V = np.stack(Y), np.stack(V)
pt = R / 'prithvi-hls-val-tta8-v1' / 'probabilities.npy'
if not pt.exists():
    pt.parent.mkdir(parents=True, exist_ok=True); cfg = yaml.safe_load(open(EXT / 'burn_scars_config.yaml'))
    MEAN = np.array(cfg['data']['init_args']['means'], np.float32); STD = np.array(cfg['data']['init_args']['stds'], np.float32)
    from terratorch.cli_tools import LightningInferenceModel
    m = LightningInferenceModel.from_config(str(EXT / 'burn_scars_config.yaml'), str(EXT / 'Prithvi_EO_V2_300M_BurnScars.pt')).model.eval()
    net = lambda x: (lambda o: getattr(o, 'output', o))(m(x)); P = np.zeros(Y.shape, np.float16)
    for i, x in enumerate(X):
        with torch.no_grad(): P[i] = tta8(net, torch.from_numpy(((x - MEAN[:, None, None]) / STD[:, None, None])[None])).softmax(1)[0, 1].numpy()
        if i % 40 == 0: print('prithvi tta8', i, flush=True)
    np.save(pt, P)
PR = np.load(R / 'prithvi-hls-val-v1/probabilities.npy').astype(np.float32); PT = np.load(pt).astype(np.float32)
ours_path = sys.argv[1] if len(sys.argv) > 1 else 'research/biome-hls-f0/probabilities_hls_val.npy'; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
OU = np.load(ours_path).astype(np.float32)
def pooled(P, t=0.5): B = (P >= t) & V; return (B & Y).sum() / ((B | Y) & V).sum()
print(f'Prithvi без преобразований {pooled(PR):.4f} | Prithvi 8 преобразований {pooled(PT):.4f} | наш ({ours_path}, порог {thr}) {pooled(OU, thr):.4f}')
def per_scene(P, t): B = (P >= t) & V; return np.array([((b & y) & v).sum() / max(((b | y) & v).sum(), 1) for b, y, v in zip(B, Y, V)])
io, ip = per_scene(OU, thr), per_scene(PR, 0.5); area = (Y & V).sum((1, 2)); nod = 1 - V.mean((1, 2)); has = area > 0
q = np.quantile(area[has], [0.25, 0.5, 0.75]); print(f'\nсцен с гарью {has.sum()} из {len(Y)}; квартили площади гари, пикс: {q.astype(int).tolist()}')
for lo, hi, name in ((0, q[0], 'мелкие'), (q[0], q[1], 'ниже медианы'), (q[1], q[2], 'выше медианы'), (q[2], 1e9, 'крупные')):
    s = has & (area >= lo) & (area < hi); print(f'{name:14s} сцен {s.sum():3d} | наш {io[s].mean():.4f} | Prithvi {ip[s].mean():.4f} | Δ {io[s].mean() - ip[s].mean():+.4f}')
s = nod > 0.01; print(f'нодата > 1 %: сцен {s.sum()} | Δ {(io[s] - ip[s]).mean() if s.any() else float("nan"):+.4f}; без нодаты Δ {(io[~s & has] - ip[~s & has]).mean():+.4f}')
edge = np.stack([ndimage.binary_dilation(y, iterations=2) & ~ndimage.binary_erosion(y, iterations=2) for y in Y])
for P, t, name in ((OU, thr, 'наш'), (PR, 0.5, 'Prithvi')):
    B = (P >= t) & V; err = (B ^ Y) & V; print(f'{name:8s} ошибки пикс.: у кромки ±2 {(err & edge).sum():8d} | в глубине {(err & ~edge).sum():8d} | FP {(B & ~Y).sum():8d} | FN {(~B & Y & V).sum():8d}')
comp = [(i, ndimage.find_objects(lab)[k], lab == k + 1) for i, y in enumerate(Y) for lab, n in [ndimage.label(y)] for k in range(n)]
for P, t, name in ((OU, thr, 'наш'), (PR, 0.5, 'Prithvi')):
    B = P >= t; hit = {'< 100': [0, 0], '100–1000': [0, 0], '> 1000': [0, 0]}
    for i, _, mask in comp:
        a = mask.sum(); key = '< 100' if a < 100 else '100–1000' if a < 1000 else '> 1000'; hit[key][1] += 1; hit[key][0] += (B[i][mask].mean() >= 0.5)
    print(f'{name:8s} найдено пятен (> половины площади):', {k: f'{h}/{n}' for k, (h, n) in hit.items()})
d = io - ip; worst = np.argsort(d)[:8]; print('\nхудшие сцены для нас:', [(names[i].split('.')[2] + '.' + names[i].split('.')[3], round(float(d[i]), 3), int(area[i])) for i in worst])
