"""Усреднение по 8 элементам диэдральной группы вместо 4 отражений (SPEC-19).
Меряется на 5 сидах d7 в ансамбле с бустингом под правилом SCL."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from src.comp.chips import BsDataset
from src.comp.features import stack
from src.comp.metric import score_bs_micro
from src.comp.unet import load as load_net
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO = np.stack([c.label_zero() for c in chips])
def probs8(model, ch):
    net, mean, std, dev = model
    x = (np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32) - mean[:, None, None]) / std[:, None, None]
    x = torch.from_numpy(x).unsqueeze(0).to(dev); acc = 0
    with torch.no_grad():
        for tr in (False, True):
            xt = x.transpose(2, 3) if tr else x
            for dims in ([], [2], [3], [2, 3]):
                xf = torch.flip(xt, dims) if dims else xt
                lg = net(xf).float(); lg = torch.flip(lg, dims) if dims else lg
                acc = acc + (lg.transpose(2, 3) if tr else lg)
    return (acc / 8).softmax(1)[0].permute(1, 2, 0).cpu().numpy()
tags = ('d7w32', 'd7s1', 'd7s2', 'd7fast', 'd7rot')
pn4 = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in tags], 0)
pn8 = np.mean([np.stack([probs8(load_net(f'models/exp_{t}.pt'), c) for c in chips]) for t in tags], 0)
for name, pn in (('4 отражения', pn4), ('8 (с поворотами)', pn8)):
    p = 0.4*PB + 0.6*pn; out = p.argmax(3).astype(np.uint8); out[~OK] = pn.argmax(3)[~OK]; out[ZERO] = 0
    r = score_bs_micro(list(T), list(out)); print(f'{name:18s} {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f}')
