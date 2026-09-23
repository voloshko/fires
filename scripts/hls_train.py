"""SPEC-69: наш U-Net на HLS Burn Scars. --fit inner — подгонка на 80 % тайлов training, замер на отложенных 20 %;
--fit all — все 540 сцен, замер на validation 264. Вход — 6 полос HLS, 2 класса, 8 преобразований при инференсе."""
import argparse, glob, json, sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rasterio
from scripts.train_unet import UNet
from src.comp.hls import tile_holdout, augment, tta8
CONFIGS = {'C0': dict(w=32, epochs=60, rot=False, gain=0.0), 'C1': dict(w=32, epochs=120, rot=True, gain=0.1), 'C2': dict(w=48, epochs=100, rot=True, gain=0.1)}
p = argparse.ArgumentParser(); p.add_argument('--fit', choices=['inner', 'all'], required=True); p.add_argument('--config', choices=list(CONFIGS), required=True)
p.add_argument('--seed', type=int, default=1); p.add_argument('--batch', type=int, default=8); p.add_argument('--out', required=True); p.add_argument('--smoke', action='store_true'); p.add_argument('--extra', action='append', default=[], help='SPEC-75: каталоги доп. сцен (*_merged.tif + .mask.tif)')
a = p.parse_args(); c = CONFIGS[a.config]; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
HLS = Path('external/hls_burn_scars')
def hls_list(split): return sorted(glob.glob(str(HLS / split / '*_merged.tif')))
def hls_load(files):
    X, Y, V = [], [], []
    for f in files:
        img = rasterio.open(f).read().astype(np.float32); m = rasterio.open(f.replace('_merged.tif', '.mask.tif')).read()[0].astype(np.int64)
        valid = (img != -9999).all(0) & (m >= 0); img[:, ~valid] = 0; m[~valid] = 0
        X.append(img); Y.append(m); V.append(valid)
    return np.stack(X), np.stack(Y), np.stack(V)
tr = hls_list('training')
if a.fit == 'inner':
    hold = tile_holdout([Path(f).name for f in tr]); fit = [f for f, h in zip(tr, hold) if not h]; ev = [f for f, h in zip(tr, hold) if h]
else: fit, ev = tr, hls_list('validation')
for e in a.extra: fit = fit + sorted(glob.glob(str(Path(e) / '*_merged.tif')))
epochs = c['epochs']
if a.smoke: fit, ev, epochs = fit[:8], ev[:2], 1
X, Y, V = hls_load(fit)
mean = X[:, :, ::4, ::4].mean((0, 2, 3)); std = X[:, :, ::4, ::4].std((0, 2, 3)) + 1e-6
Xt = torch.from_numpy(((X - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float16)).cuda(); Yt = torch.from_numpy(np.where(V, Y, -1)).cuda(); del X
net = UNet(6, classes=2, w=c['w'], depth=7).cuda().train()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); steps = max(epochs * (len(Xt) // a.batch), 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
t0 = time.time(); losses = []
for ep in range(epochs):
    order = rng.permutation(len(Xt)); el = []
    for k in range(0, len(order) - a.batch + 1, a.batch):
        idx = torch.as_tensor(order[k:k + a.batch]).cuda(); x, y = augment(Xt[idx].float(), Yt[idx], rng, rot=c['rot'], gain=c['gain'])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = net(x).float(); known = y >= 0; pb = lg.softmax(1)[:, 1] * known; t = (y == 1).float()
            loss = F.cross_entropy(lg, y.clamp(min=0), weight=torch.tensor([.25, 1.]).cuda(), reduction='none')[known].mean() + 1 - (2 * (pb * t).sum() + 1) / (pb.sum() + t.sum() + 1)
        loss.backward(); opt.step(); sched.step(); el.append(float(loss))
    losses.append(float(np.mean(el)))
    if (ep + 1) % 10 == 0 or a.smoke: print(a.config, a.fit, a.seed, 'epoch', ep + 1, 'loss', round(losses[-1], 4), 'seconds', int(time.time() - t0), flush=True)
torch.save(dict(state=net.state_dict(), mean=mean, std=std, config=a.config, seed=a.seed, fit=a.fit, **c), out / 'model.pt')  # SPEC-72: веса для свежего теста
del Xt, Yt; torch.cuda.empty_cache(); net.eval()
Xe, Ye, Ve = hls_load(ev); P = []
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for i in range(len(Xe)):
        x = torch.from_numpy(((Xe[i:i + 1] - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)).cuda()
        P.append(tta8(net, x).softmax(1)[0, 1].cpu().numpy().astype(np.float16))
P = np.stack(P); B = (P.astype(np.float32) >= 0.5) & Ve; T = (Ye == 1) & Ve
res = dict(config=a.config, fit=a.fit, seed=a.seed, scenes_fit=len(fit), scenes_eval=len(ev), seconds=time.time() - t0, iou_05=float((B & T).sum() / max((B | T).sum(), 1)), loss=losses, **c)
np.save(out / 'probabilities.npy', P)
json.dump(dict(fit=[Path(f).name for f in fit], evaluation=[Path(f).name for f in ev], mean=mean.tolist(), std=std.tolist()), open(out / 'data_manifest.json', 'w'), indent=1)
json.dump(res, open(out / 'summary.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'loss'}), flush=True)
