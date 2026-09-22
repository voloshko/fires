# SPEC-69 (исходный замер): наш U-Net (biome-hls-f0) против Prithvi-EO-2.0 BurnScars на HLS validation 264, одной меркой.
import glob, json, sys, time, numpy as np, rasterio, torch, yaml
from pathlib import Path
torch.set_num_threads(12)
HLS = Path('external/hls_burn_scars'); EXT = Path('external/prithvi-eo2-300m-burnscars'); OUT = Path('research/prithvi-hls-val-v1'); OUT.mkdir(parents=True, exist_ok=True)
files = sorted(glob.glob(str(HLS / 'validation' / '*_merged.tif')))
man = json.load(open('research/biome-hls-f0/data_manifest.json')); assert [Path(f).name for f in files] == man['hls_val'], 'порядок сцен не совпал'
X, Y, V = [], [], []
for f in files:
    img = rasterio.open(f).read().astype(np.float32); m = rasterio.open(f.replace('_merged.tif', '.mask.tif')).read()[0].astype(np.int64)
    v = (img != -9999).all(0) & (m >= 0); img[:, ~v] = 0; m[~v] = 0; X.append(img); Y.append(m); V.append(v)
Y, V = np.stack(Y).astype(bool), np.stack(V)
ours = np.load('research/biome-hls-f0/probabilities_hls_val.npy').astype(np.float32)
pp = OUT / 'probabilities.npy'
if pp.exists(): pr = np.load(pp)
else:
    cfg = yaml.safe_load(open(EXT / 'burn_scars_config.yaml')); MEAN = np.array(cfg['data']['init_args']['means'], np.float32); STD = np.array(cfg['data']['init_args']['stds'], np.float32)
    from terratorch.cli_tools import LightningInferenceModel
    model = LightningInferenceModel.from_config(str(EXT / 'burn_scars_config.yaml'), str(EXT / 'Prithvi_EO_V2_300M_BurnScars.pt')).model.eval()
    pr = np.zeros(ours.shape, np.float16); t = time.time()
    for i, x in enumerate(X):
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        with torch.no_grad(): o = model(torch.from_numpy(x[None])); lg = getattr(o, 'output', o).float()
        pr[i] = torch.softmax(lg, 1)[0, 1].numpy()
        if i % 20 == 0: print(i, f'{time.time()-t:.0f}s', flush=True)
    np.save(pp, pr)
def metrics(P, name):
    B = P >= 0.5; tp = (B & Y & V).sum(); fp = (B & ~Y & V).sum(); fn = (~B & Y & V).sum(); tn = (~B & ~Y & V).sum()
    ib = tp / (tp + fp + fn); ino = tn / (tn + fp + fn); acc = (tp + tn) / V.sum()
    per = [((b & y & v).sum() / ((b | y) & v).sum()) for b, y, v in zip(B, Y, V) if ((b | y) & v).sum()]
    print(f'{name:22s} IoU гари {ib:.4f} | IoU фона {ino:.4f} | mIoU {(ib+ino)/2:.4f} | точность {acc:.4f} | IoU гари средн. по сценам {np.mean(per):.4f}')
    return B
b1 = metrics(ours, 'наш U-Net (лес)'); b2 = metrics(pr, 'Prithvi-EO-2.0 300M')
metrics((ours + pr) / 2, 'смесь 50/50')
# бутстреп по сценам: разница IoU гари
rng = np.random.default_rng(0); n = len(Y)
def ib(B, idx): tp = (B[idx] & Y[idx] & V[idx]).sum(); return tp / (((B[idx] | Y[idx]) & V[idx]).sum())
d = [ib(b2, i) - ib(b1, i) for i in (rng.integers(0, n, n) for _ in range(500))]
print('Prithvi − наш, IoU гари: 95 % интервал', np.percentile(d, [2.5, 97.5]).round(4))
