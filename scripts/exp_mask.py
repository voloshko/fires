"""Гарь под маской облаков (SPEC-19).

Диагностика показала: пропущенная площадь вдвое больше ложной, и почти половина
пропусков — на чипах, маскированных на 50–97 %. Истина размечена под облаками
(источник разметки от облаков не зависит), а мы под маской принудительно ставим
фон. Сеть при этом ОБУЧАЛАСЬ предсказывать истину под маской: в потере участвуют
все пиксели, а признаки там — нули. Гипотеза: снять принудительный ноль.
Вероятности кэшируются на диск — решающие правила дальше стоят секунды.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs_micro
from src.comp.model import train, sample_chip, SEED
from src.comp.unet import load as load_net

NET = sys.argv[1] if len(sys.argv) > 1 else 'models/exp_d7w32.pt'
CACHE = 'models/tune_proba.npz'
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])

try:
    z = np.load(CACHE); PB, PN, T, OK = z['pb'], z['pn'], z['t'], z['ok']; print('вероятности из кэша')
except FileNotFoundError:
    t0=time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
    for c in fit:
        a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
    boost = train(np.concatenate(xs), np.concatenate(ys))
    net, mean_np, std_np, DEV = load_net(NET)
    mean = torch.tensor(mean_np, device=DEV).view(1,-1,1,1); std = torch.tensor(std_np, device=DEV).view(1,-1,1,1)
    PB, PN, T, OK = [], [], [], []
    with torch.no_grad():
        for c in tune:
            ch = d.load(c); feats = np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32)
            PB.append(boost.predict_proba(feats.reshape(len(NAMES), -1).T).reshape(*ch.shape, 4).astype(np.float16))
            x = (torch.from_numpy(feats).unsqueeze(0).to(DEV) - mean) / std
            lg = net(x).float()
            for dims in ([2], [3], [2, 3]): lg = lg + torch.flip(net(torch.flip(x, dims)).float(), dims)
            PN.append((lg/4).softmax(1)[0].permute(1,2,0).cpu().numpy().astype(np.float16))
            T.append(ch.mask); OK.append(ch.valid())
    PB, PN, T, OK = map(np.stack, (PB, PN, T, OK))
    np.savez(CACHE, pb=PB, pn=PN, t=T, ok=OK); print(f'вероятности посчитаны и закэшированы за {time.time()-t0:.0f}с')

burn_t = T > 0
print(f'истинной гари под маской: {int((burn_t & ~OK).sum())} из {int(burn_t.sum())} ({(burn_t & ~OK).sum()/burn_t.sum():.1%}); маскировано всего {(~OK).mean():.1%} пикселей')

def run(name, pred):
    r = score_bs_micro(list(T), list(pred))
    fp = int(((pred > 0) & ~burn_t).sum()); fn = int((burn_t & (pred == 0)).sum())
    print(f'  {name:44s} IoU_burn {r["iou_burn"]:.4f}  mIoU_sev {r["miou_sev"]:.4f}  '
          f'[{r["per_class"][1]:.3f} {r["per_class"][2]:.3f} {r["per_class"][3]:.3f}]  fp {fp:7d} fn {fn:7d}')

pn, pb = PN.astype(np.float32), PB.astype(np.float32)
mix = 0.6*pn + 0.4*pb
base = mix.argmax(3).astype(np.uint8)
z = base.copy(); z[~OK] = 0;                          run('ноль под маской (текущее)', z)
run('ансамбль везде', base)
n = base.copy(); n[~OK] = pn.argmax(3)[~OK];          run('вне маски ансамбль, под маской сеть', n)
# под маской фон по-прежнему может перевешивать; иерархическое решение: гарь если P(фон)<0.5
h = n.copy(); sub = ~OK & (pn[..., 0] < 0.5) & (n == 0); h[sub] = (pn[..., 1:].argmax(3) + 1)[sub]
run('… + под маской гарь при P(фон)<0.5', h)
# заполнение ближайшим валидным предсказанием (гарь связна)
from scipy.ndimage import distance_transform_edt
f = z.copy()
for i in range(len(f)):
    if (~OK[i]).any() and OK[i].any():
        idx = distance_transform_edt(~OK[i], return_distances=False, return_indices=True)
        f[i][~OK[i]] = z[i][tuple(idx)][~OK[i]]
run('под маской — ближайший валидный пиксель', f)
