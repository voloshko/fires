"""Ансамбль бустинга и сети (SPEC-19).

Три кандидата лежат в пределах 0.004 друг от друга, но ошибаются по-разному:
бустинг видит пиксель и его окно, сеть — форму пятна. Усреднение вероятностей
двух разных по устройству моделей обычно даёт больше, чем любая из них.

Обучение на 144 чипах, замер на 35 настроечных. Сеть берётся из перебора —
она обучена на тех же 144.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0,'/home/mc/fires')
import torch
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs
from src.comp.model import train, sample_chip, SEED
from src.comp.postproc import drop_small
from scripts.train_unet import UNet

NET = sys.argv[1] if len(sys.argv) > 1 else 'models/exp_bg050.pt'
PIX = 18000
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])

t0=time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng, PIX); xs.append(a); ys.append(b)
boost = train(np.concatenate(xs), np.concatenate(ys))
print(f'бустинг обучен за {time.time()-t0:.0f}с', flush=True)

bundle = torch.load(NET, map_location='cpu', weights_only=False)
net = UNet(len(NAMES), w=bundle['state']['d1.0.weight'].shape[0])
net.load_state_dict(bundle['state']); net.to(DEV).eval()
mean = torch.tensor(bundle['mean'], device=DEV).view(1,-1,1,1)
std = torch.tensor(bundle['std'], device=DEV).view(1,-1,1,1)
print(f'сеть {NET} загружена', flush=True)

pairs = []
with torch.no_grad():
    for c in tune:
        ch = d.load(c)
        flat = stack(ch).reshape(len(NAMES), -1).T
        pb = boost.predict_proba(flat).reshape(*ch.shape, 4)
        x = np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32)
        t = (torch.from_numpy(x).unsqueeze(0).to(DEV) - mean) / std
        pn = net(t).softmax(1)[0].permute(1,2,0).cpu().numpy()
        pairs.append((ch.mask, pb, pn, ch.valid()))
print('вероятности посчитаны', flush=True)

print('вес сети  IoU_burn  mIoU_sev')
for w in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
    out = []
    for truth, pb, pn, ok in pairs:
        p = (1-w)*pb + w*pn
        pred = p.argmax(2).astype(np.uint8); pred[~ok] = 0
        out.append(score_bs(truth, drop_small(pred, 100)))
    print('%-9.2f %8.4f %9.4f' % (w,
          np.nanmean([r['iou_burn'] for r in out]),
          np.nanmean([r['miou_sev'] for r in out])), flush=True)
