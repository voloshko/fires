"""SPEC-64: дообучение Prithvi-EO-2.0-300M (кодировщик + UNet-декодер из BurnScars) на наших чипах,
4 класса степени, сцена «после», 6 полос. Групповой фолд соседа (fit/evaluation из его манифеста),
потеря как у сиама (CE с весом фона 0.25 + dice гарь/фон), отражения, BF16. На выходе — кэш
вероятностей в формате bs-confirm-* для замеров смеси."""
import argparse, json, time, sys, numpy as np, torch, torch.nn.functional as F; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
p = argparse.ArgumentParser(); p.add_argument('--fold', type=int); p.add_argument('--seed', type=int, default=20261001); p.add_argument('--epochs', type=int, default=60); p.add_argument('--batch', type=int, default=8)
p.add_argument('--lr-encoder', type=float, default=5e-5); p.add_argument('--lr-head', type=float, default=5e-4); p.add_argument('--out', required=True); p.add_argument('--smoke', action='store_true'); p.add_argument('--final', action='store_true'); p.add_argument('--save-model', action='store_true')
args = p.parse_args(); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
HYP = Path.home() / 'fires-hypotheses'; EXT = Path('external/prithvi-eo2-300m-burnscars'); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
d = BsDataset('data/comp/train/bs')
if args.final: fit, ev = [c for c in d.chip_ids() if d.has_post(c)], []
else:
    m = json.load(open(HYP / f'research/bs-confirm-siam-f{args.fold}-v1/data_manifest.json')); fit, ev = m['fit'], m['evaluation']
if args.smoke: fit, ev = fit[:4], ev[:2]
import yaml; cfg = yaml.safe_load(open(EXT / 'burn_scars_config.yaml')); MEAN = torch.tensor(cfg['data']['init_args']['means']).view(1, 6, 1, 1); STD = torch.tensor(cfg['data']['init_args']['stds']).view(1, 6, 1, 1)
BANDS = [0, 1, 2, 6, 7, 8]
def load(ids):
    X, Y = [], []
    for c in ids:
        ch = d.load(c); post = (ch.post if ch.post.size else ch.pre).astype(np.float32)[BANDS] / 10000.0
        X.append(torch.from_numpy(post).half()); Y.append(torch.from_numpy(ch.mask.astype(np.int64)))
    return torch.stack(X), torch.stack(Y)
X, Y = load(fit); X = ((X.float() - MEAN) / STD).half().cuda(); Y = Y.cuda()
from terratorch.cli_tools import LightningInferenceModel
lm = LightningInferenceModel.from_config(str(EXT / 'burn_scars_config.yaml'), str(EXT / 'Prithvi_EO_V2_300M_BurnScars.pt')); net = lm.model.model
net.head.head[2] = torch.nn.Conv2d(64, 4, 1)   # голова на 4 класса вместо гарь/фон
net = net.cuda().train()
enc = [q for n, q in net.named_parameters() if n.startswith('encoder')]; rest = [q for n, q in net.named_parameters() if not n.startswith('encoder')]
opt = torch.optim.AdamW([{'params': enc, 'lr': args.lr_encoder}, {'params': rest, 'lr': args.lr_head}], weight_decay=0.05)
steps = max(args.epochs * (len(fit) // args.batch), 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_encoder, args.lr_head], total_steps=steps, pct_start=0.1)
weights = torch.tensor([.25, 1., 1., 1.]).cuda(); t0 = time.time(); losses = []
def fwd(x):
    o = net(x); return getattr(o, 'output', o)
for epoch in range(args.epochs):
    order = rng.permutation(len(fit)); el = []
    for k in range(0, len(order) - args.batch + 1, args.batch):
        idx = torch.as_tensor(order[k:k + args.batch]).cuda(); x, y = X[idx].float(), Y[idx]
        if rng.random() < 0.5: x, y = x.flip(3), y.flip(2)
        if rng.random() < 0.5: x, y = x.flip(2), y.flip(1)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = fwd(x).float(); pb = 1 - logits.softmax(1)[:, 0]; t = (y > 0).float()
            loss = F.cross_entropy(logits, y, weight=weights) + 1 - (2 * (pb * t).sum() + 1) / (pb.sum() + t.sum() + 1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); el.append(float(loss))
    losses.append(float(np.mean(el)))
    if (epoch + 1) % 10 == 0 or args.smoke: print('prithvi', args.seed, 'epoch', epoch + 1, 'loss', losses[-1], 'seconds', int(time.time() - t0), flush=True)
del X, Y; torch.cuda.empty_cache(); net.eval()
if args.save_model or args.final: torch.save(dict(state=net.state_dict(), bands=BANDS, mean=MEAN, std=STD, classes=4, seed=args.seed, epochs=args.epochs, fit=fit), out / 'model.pt')
json.dump(dict(fit=fit, evaluation=ev, source='neighbour fold manifest' if not args.final else 'all chips with post'), open(out / 'data_manifest.json', 'w'), indent=1)
if ev:
    Xe, _ = load(ev); Xe = ((Xe.float() - MEAN) / STD).cuda(); probs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(len(ev)):
            x = Xe[i:i + 1]; lg = fwd(x).float()
            for dims in ([3], [2], [2, 3]): lg = lg + torch.flip(fwd(torch.flip(x, dims)).float(), dims)
            probs.append((lg / 4).softmax(1)[0].permute(1, 2, 0).half().cpu().numpy())
    np.save(out / 'probabilities.npy', np.stack(probs))
json.dump(dict(seconds=time.time() - t0, loss=losses, epochs=args.epochs, batch=args.batch, lr=[args.lr_encoder, args.lr_head], gpu_peak_bytes=torch.cuda.max_memory_allocated(), smoke=args.smoke, chips=len(fit)), open(out / 'summary.json', 'w'), indent=1)
print('done', len(fit), 'fit', len(ev), 'eval', int(time.time() - t0), 's', flush=True)
