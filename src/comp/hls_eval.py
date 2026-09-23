"""SPEC-73/74/75: общий замер на каталоге окон в формате HLS Burn Scars (*_merged.tif + .mask.tif + manifest.json).
Вероятности кэшируются: наш ансамбль C1 (8 преобразований) и Prithvi-EO-2.0 BurnScars (CPU)."""
import json
from pathlib import Path
import numpy as np


def load_dir(d, manifest='manifest.json', layers=('slope',)):
    import rasterio
    d = Path(d); man = json.load(open(d / manifest))['windows']; X, Y, V, extra = [], [], [], {}
    for w in man:
        img = rasterio.open(d / f"{w['name']}_merged.tif").read().astype(np.float32); m = rasterio.open(d / f"{w['name']}.mask.tif").read()[0]
        v = (img != -9999).all(0) & (m >= 0); img[:, ~v] = 0; X.append(img); Y.append(m == 1); V.append(v)
        for k in layers:
            s = d / f"{w['name']}.{k}.tif"
            if s.exists(): extra.setdefault(k, []).append(rasterio.open(s).read()[0])
    return man, np.stack(X), np.stack(Y), np.stack(V), {k: np.stack(v) for k, v in extra.items()}


def c1_probs(X, model_dirs, cache):
    cache = Path(cache)
    if cache.exists(): return np.load(cache).astype(np.float32)
    import torch
    from scripts.train_unet import UNet
    from src.comp.hls import tta8
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; acc = np.zeros((len(X),) + X.shape[2:], np.float32)
    for md in model_dirs:
        b = torch.load(Path(md) / 'model.pt', map_location='cpu', weights_only=False)
        net = UNet(6, classes=2, w=b['w'], depth=7); net.load_state_dict(b['state']); net = net.to(dev).eval(); mean, std = b['mean'], b['std']
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
            for i, x in enumerate(X):
                xt = torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].to(dev)
                acc[i] += tta8(net, xt).softmax(1)[0, 1].float().cpu().numpy()
        del net
    acc /= len(model_dirs); cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, acc.astype(np.float16)); return acc


def prithvi_probs(X, cache, tta=False):
    cache = Path(cache)
    if cache.exists(): return np.load(cache).astype(np.float32)
    import torch, yaml
    from src.comp.hls import tta8
    from terratorch.cli_tools import LightningInferenceModel
    ext = Path('external/prithvi-eo2-300m-burnscars'); torch.set_num_threads(12); cfg = yaml.safe_load(open(ext / 'burn_scars_config.yaml'))
    MEAN = np.array(cfg['data']['init_args']['means'], np.float32); STD = np.array(cfg['data']['init_args']['stds'], np.float32)
    m = LightningInferenceModel.from_config(str(ext / 'burn_scars_config.yaml'), str(ext / 'Prithvi_EO_V2_300M_BurnScars.pt')).model.eval().cpu()
    net = lambda x: (lambda o: getattr(o, 'output', o))(m(x)); P = np.zeros((len(X),) + X.shape[2:], np.float16)
    for i, x in enumerate(X):
        xt = torch.from_numpy(((x - MEAN[:, None, None]) / STD[:, None, None]).astype(np.float32))[None]
        with torch.no_grad(): P[i] = (tta8(net, xt) if tta else net(xt).float()).softmax(1)[0, 1].numpy()
    cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, P); return P.astype(np.float32)


class Scorer:
    """IoU гари по пулу и бутстреп по окнам: пересечение и объединение по окну считаются один раз."""
    def __init__(self, Y, V, n_boot=1000, seed=0, pixel_mask=None):
        self.Y, self.V = Y, V if pixel_mask is None else V & pixel_mask
        rng = np.random.default_rng(seed); self.boots = [rng.integers(0, len(Y), len(Y)) for _ in range(n_boot)]

    def stats(self, P, t):
        B = (P >= t) & self.V; return (B & self.Y).sum((1, 2)).astype(np.float64), ((B | self.Y) & self.V).sum((1, 2)).astype(np.float64)

    @staticmethod
    def iou(st, idx=slice(None)):
        u = st[1][idx].sum(); return st[0][idx].sum() / u if u else float('nan')

    def ci(self, st):
        return np.percentile([self.iou(st, i) for i in self.boots], [2.5, 97.5])

    def compare(self, sa, sb):
        d = np.array([self.iou(sa, i) - self.iou(sb, i) for i in self.boots]); lo, hi = np.percentile(d, [2.5, 97.5]); pt = self.iou(sa) - self.iou(sb)
        return pt, lo, hi, ('ЛУЧШЕ' if pt > 0 and lo > 0 else 'ХУЖЕ' if hi < 0 else 'ВРОВЕНЬ')
