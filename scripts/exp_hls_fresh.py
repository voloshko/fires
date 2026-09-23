"""SPEC-72: свежий тест (external/hls_fresh). Наш ансамбль C1 (5 сидов, research/hls-c1-final-s*/model.pt, 8 преобразований)
и Prithvi-EO-2.0 BurnScars (без и с 8 преобразованиями, CPU); H1 — смесь против лучшей одиночной, H2 — C1 против Prithvi,
оба при пороге 0.5, IoU гари по пулу, бутстреп по окнам. Кэш вероятностей — research/hls-fresh-v1/."""
import json, sys, numpy as np, rasterio, torch, yaml
from pathlib import Path
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_unet import UNet
from src.comp.hls import tta8
R = Path('research'); FR = Path('external/hls_fresh'); OUT = R / 'hls-fresh-v1'; OUT.mkdir(parents=True, exist_ok=True)
EXT = Path('external/prithvi-eo2-300m-burnscars'); SEEDS = (1, 2, 3, 4, 5)
man = json.load(open(FR / 'manifest.json'))['windows']; X, Y, V = [], [], []
for w in man:
    img = rasterio.open(FR / f"{w['name']}_merged.tif").read().astype(np.float32); m = rasterio.open(FR / f"{w['name']}.mask.tif").read()[0]
    v = (img != -9999).all(0) & (m >= 0); img[:, ~v] = 0; X.append(img); Y.append(m == 1); V.append(v)
Y, V = np.stack(Y), np.stack(V); print('окон', len(Y), '| доля гари', round(float((Y & V).sum() / V.sum()), 4), flush=True)

def ours():
    p = OUT / 'c1_ensemble.npy'
    if p.exists(): return np.load(p).astype(np.float32)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; acc = np.zeros(Y.shape, np.float32)
    for s in SEEDS:
        b = torch.load(R / f'hls-c1-final-s{s}/model.pt', map_location='cpu', weights_only=False)
        net = UNet(6, classes=2, w=b['w'], depth=7); net.load_state_dict(b['state']); net = net.to(dev).eval(); mean, std = b['mean'], b['std']
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
            for i, x in enumerate(X):
                xt = torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].to(dev)
                acc[i] += tta8(net, xt).softmax(1)[0, 1].float().cpu().numpy()
        print('C1 сид', s, 'готов', flush=True); del net; torch.cuda.empty_cache()
    acc /= len(SEEDS); np.save(p, acc.astype(np.float16)); return acc

def prithvi(tta):
    p = OUT / f'prithvi{"_tta8" if tta else ""}.npy'
    if p.exists(): return np.load(p).astype(np.float32)
    torch.set_num_threads(12); cfg = yaml.safe_load(open(EXT / 'burn_scars_config.yaml'))
    MEAN = np.array(cfg['data']['init_args']['means'], np.float32); STD = np.array(cfg['data']['init_args']['stds'], np.float32)
    from terratorch.cli_tools import LightningInferenceModel
    m = LightningInferenceModel.from_config(str(EXT / 'burn_scars_config.yaml'), str(EXT / 'Prithvi_EO_V2_300M_BurnScars.pt')).model.eval().cpu()
    net = lambda x: (lambda o: getattr(o, 'output', o))(m(x)); P = np.zeros(Y.shape, np.float16)
    for i, x in enumerate(X):
        xt = torch.from_numpy(((x - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32))[None]
        with torch.no_grad(): P[i] = (tta8(net, xt) if tta else net(xt).float()).softmax(1)[0, 1].numpy()
        if i % 50 == 0: print('Prithvi', 'tta8' if tta else '', i, flush=True)
    np.save(p, P); return P.astype(np.float32)

if sys.argv[1:] == ['prithvi']: prithvi(False); prithvi(True); sys.exit(0)   # CPU-часть заранее, пока учится C1
# Проверка годности переобученного ансамбля на validation (не критерий): ±0.005 от 0.8666.
val_names = json.load(open(R / 'biome-hls-f0/data_manifest.json'))['hls_val']; YV, VV = [], []
for n in val_names:
    f = Path('external/hls_burn_scars/validation') / n; i = rasterio.open(f).read(); m = rasterio.open(str(f).replace('_merged.tif', '.mask.tif')).read()[0]
    VV.append((i != -9999).all(0) & (m >= 0)); YV.append(m == 1)
YV, VV = np.stack(YV), np.stack(VV); EV = np.mean([np.load(R / f'hls-c1-final-s{s}/probabilities.npy').astype(np.float32) for s in SEEDS], 0)
bv = (EV >= 0.7) & VV; iv = (bv & YV).sum() / ((bv | YV) & VV).sum()
print(f'годность: ансамбль C1 на validation при 0.7 = {iv:.4f} (SPEC-69: 0.8666) —', 'ГОДЕН' if abs(iv - 0.8666) <= 0.005 else 'НЕ ГОДЕН', flush=True)

C, PR, PT = ours(), prithvi(False), prithvi(True); MIX = (C + PR) / 2
def binz(P, t): return (P >= t) & V
_st = {}
def iou(B, idx=None):
    k = id(B)
    if k not in _st: _st[k] = ((B & Y).sum((1, 2)).astype(np.float64), ((B | Y) & V).sum((1, 2)).astype(np.float64))
    a, u = _st[k]; idx = slice(None) if idx is None else idx; return a[idx].sum() / u[idx].sum()
rows = {'C1 ансамбль (0.5)': binz(C, 0.5), 'Prithvi (0.5)': binz(PR, 0.5), 'смесь C1 + Prithvi (0.5)': binz(MIX, 0.5),
        'C1 ансамбль (0.7, правило SPEC-69)': binz(C, 0.7), 'Prithvi 8 преобразований (0.5)': binz(PT, 0.5)}
for k, B in rows.items():
    fp = (B & ~Y).sum(); fn = (~B & Y & V).sum(); print(f'{k:36s} IoU гари {iou(B):.4f} | FP {fp:9d} | FN {fn:9d}')
rng = np.random.default_rng(0); n = len(Y); boots = [rng.integers(0, n, n) for _ in range(1000)]
def verdict(a, b, name):
    d = np.array([iou(a, i) - iou(b, i) for i in boots]); lo, hi = np.percentile(d, [2.5, 97.5]); pt = iou(a) - iou(b)
    v = 'ЛУЧШЕ' if pt > 0 and lo > 0 else 'ХУЖЕ' if hi < 0 else 'ВРОВЕНЬ'
    print(f'{name}: {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {v}'); return v
best = 'C1 ансамбль (0.5)' if iou(rows['C1 ансамбль (0.5)']) >= iou(rows['Prithvi (0.5)']) else 'Prithvi (0.5)'
print(f'\nH1 смесь против лучшей одиночной ({best}):'); verdict(rows['смесь C1 + Prithvi (0.5)'], rows[best], '  смесь − одиночная')
print('H2 C1 против Prithvi:'); verdict(rows['C1 ансамбль (0.5)'], rows['Prithvi (0.5)'], '  C1 − Prithvi')
print('\nвторичное: по типу пожара (IoU гари)')
typ = np.array([w['incid_type'] for w in man])
for t in sorted(set(typ)):
    idx = np.where(typ == t)[0]; print(f'  {t:16s} окон {len(idx):3d} | C1 {iou(rows["C1 ансамбль (0.5)"], idx):.4f} | Prithvi {iou(rows["Prithvi (0.5)"], idx):.4f} | смесь {iou(rows["смесь C1 + Prithvi (0.5)"], idx):.4f}')
comp = [(i, lab == k + 1) for i, y in enumerate(Y) for lab, nn in [ndimage.label(y)] for k in range(nn)]
for k in ('C1 ансамбль (0.5)', 'Prithvi (0.5)', 'смесь C1 + Prithvi (0.5)'):
    B = rows[k]; hit = {'< 100': [0, 0], '100–1000': [0, 0], '> 1000': [0, 0]}
    for i, mk in comp:
        a = mk.sum(); key = '< 100' if a < 100 else '100–1000' if a < 1000 else '> 1000'; hit[key][1] += 1; hit[key][0] += B[i][mk].mean() >= 0.5
    print(f'  найдено пятен {k:26s}', {kk: f'{h}/{nn}' for kk, (h, nn) in hit.items()})
