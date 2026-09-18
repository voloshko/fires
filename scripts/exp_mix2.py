"""Ансамбли из нескольких сетей и бустинга, микро-усреднение (SPEC-19).

Смесь сети с бустингом дала +0.013 — модели ошибаются по-разному. Сети разной
глубины тоже видят разное, значит смесь из них должна работать по той же
причине. Проверяется на 35 настроечных чипах; отложенные 45 не трогаются.

Веса перебираются по сетке с шагом 0.2 — искать точнее бессмысленно, плато
предыдущего ансамбля было шире этого шага.
"""
import sys, json, hashlib, time, itertools, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.metric import score_bs_micro
from src.comp.model import train, sample_chip, SEED
from src.comp.postproc import drop_small
from src.comp.unet import load as load_net

NETS = sys.argv[1:] or ['models/exp_d5.pt', 'models/exp_d6w32.pt']

d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])

t0=time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
boost = train(np.concatenate(xs), np.concatenate(ys))
print(f'бустинг обучен за {time.time()-t0:.0f}с', flush=True)

loaded = []
for path in NETS:
    net, mean_np, std_np, DEV = load_net(path)
    loaded.append((path, net,
                   torch.tensor(mean_np, device=DEV).view(1,-1,1,1),
                   torch.tensor(std_np, device=DEV).view(1,-1,1,1), DEV))
    print(f'сеть {path}: глубина {net.depth}', flush=True)

truths, oks = [], []
probs = {p: [] for p in ['бустинг'] + NETS}
with torch.no_grad():
    for c in tune:
        ch = d.load(c)
        feats = np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32)
        probs['бустинг'].append(
            boost.predict_proba(feats.reshape(len(NAMES), -1).T).reshape(*ch.shape, 4))
        for path, net, mean, std, DEV in loaded:
            x = (torch.from_numpy(feats).unsqueeze(0).to(DEV) - mean) / std
            logits = net(x).float()
            for dims in ([2], [3], [2, 3]):
                logits = logits + torch.flip(net(torch.flip(x, dims)).float(), dims)
            probs[path].append((logits/4).softmax(1)[0].permute(1,2,0).cpu().numpy())
        truths.append(ch.mask); oks.append(ch.valid())
print('вероятности посчитаны', flush=True)

sources = ['бустинг'] + NETS
def measure(weights, min_blob):
    preds = []
    for i, ok in enumerate(oks):
        p = sum(w * probs[src][i] for src, w in zip(sources, weights) if w)
        pred = p.argmax(2).astype(np.uint8); pred[~ok] = 0
        preds.append(drop_small(pred, min_blob) if min_blob else pred)
    return score_bs_micro(truths, preds)

grid = [w for w in itertools.product(*[np.arange(0, 1.01, 0.2)]*len(sources))
        if abs(sum(w) - 1.0) < 1e-6]
print(f'\nвесов к проверке: {len(grid)}  ({" / ".join(sources)})', flush=True)
best = []
for weights in grid:
    for mb in (0, 100):
        r = measure(weights, mb)
        best.append(((0.35*r['iou_burn'] + 0.30*r['miou_sev'])/0.65, weights, mb, r))
best.sort(reverse=True, key=lambda t: t[0])
print('\nвзвеш.  веса                      фильтр  IoU_burn  mIoU_sev   кл1')
for score, weights, mb, r in best[:8]:
    print('%.4f  %-25s %-7s %8.4f %9.4f %6.3f' % (
        score, ' '.join(f'{w:.1f}' for w in weights), mb or '—',
        r['iou_burn'], r['miou_sev'], r['per_class'][1]), flush=True)
