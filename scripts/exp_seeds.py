"""Ансамбль по сидам (SPEC-19). Несколько сетей одной конфигурации, разные сиды —
усреднение вероятностей. Бустинг берётся из кэша exp_mask.py, правило под
маской — новое (сеть одна). Меряется лестница: 1, 2, 3, … сетей."""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs_micro
from src.comp.unet import load as load_net

NETS = sys.argv[1:]
z = np.load(f'models/tune_proba_{len(NAMES)}.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
chips = [d.load(c) for c in tune]
feats = [np.nan_to_num(stack(c), posinf=0, neginf=0).astype(np.float32) for c in chips]
ZERO = np.stack([c.label_zero() for c in chips])   # правило по классам SCL, см. exp_scl.py

def probs(path):
    cached = path.replace('.pt', '.tune.npy')
    try: return np.load(cached)
    except FileNotFoundError: pass
    net, mean_np, std_np, DEV = load_net(path)
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    maskch = bundle.get('maskch', 0)
    mean = torch.tensor(mean_np, device=DEV).view(1,-1,1,1); std = torch.tensor(std_np, device=DEV).view(1,-1,1,1)
    out = []
    with torch.no_grad():
        for f, ok in zip(feats, OK):
            x = np.concatenate([f, ok[None].astype(np.float32)]) if maskch else f
            x = (torch.from_numpy(x).unsqueeze(0).to(DEV) - mean) / std
            lg = net(x).float()
            for dims in ([2], [3], [2, 3]): lg = lg + torch.flip(net(torch.flip(x, dims)).float(), dims)
            out.append((lg/4).softmax(1)[0].permute(1,2,0).cpu().numpy().astype(np.float16))
    out = np.stack(out); np.save(cached, out); return out

def measure(pn, w=0.6):
    p = (1-w)*PB + w*pn; out = p.argmax(3).astype(np.uint8); out[~OK] = pn.argmax(3)[~OK]; out[ZERO] = 0
    r = score_bs_micro(list(T), list(out))
    return r['iou_burn'], r['miou_sev'], r['per_class'][1], (0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65

PN = []
for n in NETS:
    t0 = time.time(); PN.append(probs(n).astype(np.float32))
    one = measure(PN[-1]); acc = measure(np.mean(PN, 0))
    print(f'{n:28s} одна: {one[0]:.4f}/{one[1]:.4f} кл1 {one[2]:.3f} взв {one[3]:.4f}   '
          f'накоплено {len(PN)} сетей: {acc[0]:.4f}/{acc[1]:.4f} кл1 {acc[2]:.3f} взв {acc[3]:.4f}  ({time.time()-t0:.0f}с)', flush=True)
if len(PN) > 1:
    print('сети без бустинга (w=1):', '%.4f/%.4f' % measure(np.mean(PN, 0), 1.0)[:2])
    for w in (0.5, 0.7): print(f'вес сети {w}:', '%.4f/%.4f взв %.4f' % (lambda r: (r[0], r[1], r[3]))(measure(np.mean(PN, 0), w)))
