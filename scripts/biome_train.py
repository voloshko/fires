"""SPEC-66: наш U-Net (6 полос одной даты, 2 класса) на лесе (HLS Burn Scars), степи (наши чипы,
сцена «после») и на обоих; перенос лес ↔ степь. Метрика — IoU гари на валидных пикселях.
Степь — групповые фолды соседа; лес — их же сплит training/validation (264 сцены)."""
import argparse, json, time, sys, glob, numpy as np, torch, torch.nn.functional as F; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
import rasterio
from scipy.ndimage import zoom
from src.comp.chips import BsDataset
from scripts.train_unet import UNet
p = argparse.ArgumentParser(); p.add_argument('--train', choices=['hls', 'steppe', 'both'], required=True); p.add_argument('--fold', type=int, default=0); p.add_argument('--seed', type=int, default=1)
p.add_argument('--epochs', type=int); p.add_argument('--batch', type=int, default=8); p.add_argument('--out', required=True); p.add_argument('--smoke', action='store_true')
a = p.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
HYP = Path.home() / 'fires-hypotheses'; HLS = Path('external/hls_burn_scars'); d = BsDataset('data/comp/train/bs'); BANDS = [0, 1, 2, 6, 7, 8]
def hls_list(split): return sorted(glob.glob(str(HLS / split / '*_merged.tif')))
def hls_load(files):
    X, Y, V = [], [], []
    for f in files:
        img = rasterio.open(f).read().astype(np.float32); m = rasterio.open(f.replace('_merged.tif', '.mask.tif')).read()[0].astype(np.int64)
        valid = (img != -9999).all(0) & (m >= 0); img[:, ~valid] = 0; m[~valid] = 0
        X.append(img); Y.append(m); V.append(valid)
    return np.stack(X), np.stack(Y), np.stack(V)
def steppe_load(ids, scale=1.0):
    X, Y, V = [], [], []
    for c in ids:
        ch = d.load(c); post = (ch.post if ch.post.size else ch.pre).astype(np.float32)[BANDS] / 10000.0; m = (ch.mask > 0).astype(np.int64); v = ch.valid()
        if scale != 1.0:
            post = np.stack([zoom(b, scale, order=1) for b in post]); m = zoom(m.astype(np.float32), scale, order=0).astype(np.int64); v = zoom(v.astype(np.float32), scale, order=0) > 0.5
        X.append(post); Y.append(m); V.append(v)
    return np.stack(X), np.stack(Y), np.stack(V)
man = json.load(open(HYP / f'research/bs-confirm-siam-f{a.fold}-v1/data_manifest.json')); s_fit, s_ev = man['fit'], man['evaluation']
h_tr, h_va = hls_list('training'), hls_list('validation')
if a.smoke: s_fit, s_ev, h_tr, h_va = s_fit[:4], s_ev[:2], h_tr[:4], h_va[:2]
parts = []
if a.train in ('hls', 'both'): parts.append(hls_load(h_tr))
if a.train in ('steppe', 'both'): parts.append(steppe_load(s_fit))
X = np.concatenate([q[0] for q in parts]); Y = np.concatenate([q[1] for q in parts]); V = np.concatenate([q[2] for q in parts])
mean = X[:, :, ::4, ::4].mean((0, 2, 3)); std = X[:, :, ::4, ::4].std((0, 2, 3)) + 1e-6
Xt = torch.from_numpy(((X - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float16)).cuda(); Yt = torch.from_numpy(np.where(V, Y, -1)).cuda(); del X
net = UNet(6, classes=2, w=32, depth=7).cuda().train()
epochs = a.epochs or {'hls': 60, 'steppe': 200, 'both': 50}[a.train]
if a.smoke: epochs = 1
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); steps = max(epochs * (len(Xt) // a.batch), 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
t0 = time.time(); losses = []
for ep in range(epochs):
    order = rng.permutation(len(Xt)); el = []
    for k in range(0, len(order) - a.batch + 1, a.batch):
        idx = torch.as_tensor(order[k:k + a.batch]).cuda(); x, y = Xt[idx].float(), Yt[idx]
        if rng.random() < 0.5: x, y = x.flip(3), y.flip(2)
        if rng.random() < 0.5: x, y = x.flip(2), y.flip(1)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = net(x).float(); known = y >= 0; pb = lg.softmax(1)[:, 1] * known; t = (y == 1).float()
            loss = F.cross_entropy(lg, y.clamp(min=0), weight=torch.tensor([.25, 1.]).cuda(), reduction='none')[known].mean() + 1 - (2 * (pb * t).sum() + 1) / (pb.sum() + t.sum() + 1)
        loss.backward(); opt.step(); sched.step(); el.append(float(loss))
    losses.append(float(np.mean(el)))
    if (ep + 1) % 10 == 0 or a.smoke: print(a.train, a.fold, 'epoch', ep + 1, 'loss', losses[-1], 'seconds', int(time.time() - t0), flush=True)
del Xt, Yt; torch.cuda.empty_cache(); net.eval()
def predict(Xn):
    P = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(len(Xn)):
            x = torch.from_numpy(((Xn[i:i + 1] - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)).cuda(); lg = net(x).float()
            for dims in ([3], [2], [2, 3]): lg = lg + torch.flip(net(torch.flip(x, dims)).float(), dims)
            P.append((lg / 4).softmax(1)[0, 1].cpu().numpy().astype(np.float16))
    return np.stack(P)
def iou(P, Y, V, thr=0.5):
    pr = (P.astype(np.float32) > thr) & V; t = (Y == 1) & V; return float((pr & t).sum() / max((pr | t).sum(), 1))
def lost(P, Y, V):
    n = 0
    for pp, yy, vv in zip(P, Y, V):
        t = (yy == 1) & vv; pr = (pp.astype(np.float32) > 0.5) & vv
        if t.any(): n += ((pr & t).sum() / max((pr | t).sum(), 1)) < 0.3
    return int(n)
res = dict(train=a.train, fold=a.fold, seed=a.seed, epochs=epochs, chips_train=int(len(V)), seconds=time.time() - t0, loss=losses)
Xh, Yh, Vh = hls_load(h_va); Ph = predict(Xh); res['hls_val_iou'] = iou(Ph, Yh, Vh); res['hls_val_n'] = len(h_va); np.save(out / 'probabilities_hls_val.npy', Ph)
ev = s_ev if a.train != 'hls' else [c for f in range(5) for c in json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']]
if a.smoke: ev = ev[:2]
Xs, Ys, Vs = steppe_load(ev); Ps = predict(Xs); res['steppe_iou'] = iou(Ps, Ys, Vs); res['steppe_lost'] = lost(Ps, Ys, Vs); res['steppe_n'] = len(ev); np.save(out / 'probabilities_steppe.npy', Ps)
Xs3, Ys3, Vs3 = steppe_load(ev, 2 / 3); Ps3 = predict(np.pad(Xs3, ((0, 0), (0, 0), (0, 512 - Xs3.shape[-2]), (0, 512 - Xs3.shape[-1])), mode='reflect'))[:, :Xs3.shape[-2], :Xs3.shape[-1]]
res['steppe_iou_30m'] = iou(Ps3, Ys3, Vs3)
json.dump(dict(steppe_eval=ev, hls_val=[Path(f).name for f in h_va], bands=BANDS, mean=mean.tolist(), std=std.tolist()), open(out / 'data_manifest.json', 'w'), indent=1)
json.dump(res, open(out / 'summary.json', 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k != 'loss'}), flush=True)
