"""Где физически сидит ошибка (SPEC-19). Метрика — микро, пул пикселей: несколько
чипов могут держать половину всей ложной площади, и тогда они и есть метрика.

Таблицы: чип → ложная / пропущенная площадь и доля в общем пуле; ложная площадь
по типу покрова; доля ложной гари рядом с маской облаков. Плюс аудит разметки
на всех обучающих чипах без модели: насколько истина вообще объяснима dNBR.
"""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch
from scipy.ndimage import binary_dilation
from src.comp.chips import BsDataset
from src.comp.features import NAMES, stack
from src.comp.model import train, sample_chip, SEED
from src.comp.unet import load as load_net

NET = sys.argv[1] if len(sys.argv) > 1 else 'models/exp_d7w32.pt'
W = 0.6
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json'))
ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())
tune, fit = sorted(rank[:35]), sorted(rank[35:])
I_DNBR, I_LC = NAMES.index('dnbr'), NAMES.index('landcover')

# --- аудит разметки: без модели, на всех 179 --------------------------------
print('АУДИТ РАЗМЕТКИ: чип, доля гари, AUC(dNBR→гарь), медиана dNBR гари, медиана dNBR фона')
audit = []
for c in fit + tune:
    ch = d.load(c); f = stack(ch); ok = ch.valid(); burn = ch.mask > 0
    dn = f[I_DNBR][ok]; b = burn[ok]
    if b.sum() < 50 or (~b).sum() < 50:
        audit.append((c, b.mean(), np.nan, np.nan, np.nan)); continue
    # AUC через ранги (Манн-Уитни)
    r = np.argsort(np.argsort(dn)).astype(np.float64)
    auc = (r[b].sum() - b.sum()*(b.sum()-1)/2) / (b.sum()*(~b).sum())
    audit.append((c, b.mean(), auc, np.median(dn[b]), np.median(dn[~b])))
audit.sort(key=lambda t: (np.nan_to_num(t[2], nan=2.0)))
for c, share, auc, mb, mf in audit[:15]:
    print(f'  {c:28s} гарь {share:6.3f}  AUC {auc:6.3f}  dNBR гари {mb:6.3f}  фона {mf:6.3f}')
aucs = np.array([a[2] for a in audit], float)
print(f'  всего {len(audit)}, AUC<0.7: {int((aucs<0.7).sum())}, AUC<0.8: {int((aucs<0.8).sum())}, медиана {np.nanmedian(aucs):.3f}', flush=True)

# --- ансамбль на 35 настроечных ----------------------------------------------
t0=time.time(); rng = np.random.default_rng(SEED); xs, ys = [], []
for c in fit:
    a, b = sample_chip(d.load(c), rng); xs.append(a); ys.append(b)
boost = train(np.concatenate(xs), np.concatenate(ys))
net, mean_np, std_np, DEV = load_net(NET)
mean = torch.tensor(mean_np, device=DEV).view(1,-1,1,1); std = torch.tensor(std_np, device=DEV).view(1,-1,1,1)
print(f'модели готовы за {time.time()-t0:.0f}с', flush=True)

rows = []; fp_lc = {}; fp_near_cloud = 0; fp_total = 0; fn_total = 0; tp_total = 0
with torch.no_grad():
    for c in tune:
        ch = d.load(c); feats = np.nan_to_num(stack(ch), posinf=0, neginf=0).astype(np.float32)
        pb = boost.predict_proba(feats.reshape(len(NAMES), -1).T).reshape(*ch.shape, 4)
        x = (torch.from_numpy(feats).unsqueeze(0).to(DEV) - mean) / std
        lg = net(x).float()
        for dims in ([2], [3], [2, 3]): lg = lg + torch.flip(net(torch.flip(x, dims)).float(), dims)
        pn = (lg/4).softmax(1)[0].permute(1,2,0).cpu().numpy()
        pred = ((1-W)*pb + W*pn).argmax(2); ok = ch.valid(); pred[~ok] = 0
        t, p = ch.mask > 0, pred > 0
        fp, fn, tp = (p & ~t), (t & ~p), (t & p)
        near = binary_dilation(~ok, iterations=3) & ok
        fp_near_cloud += int((fp & near).sum())
        lc = feats[I_LC][fp]
        for k, n in zip(*np.unique(lc, return_counts=True)): fp_lc[int(k)] = fp_lc.get(int(k), 0) + int(n)
        rows.append((c, int(t.sum()), int(fp.sum()), int(fn.sum()), int(tp.sum()), float((~ok).mean())))
        fp_total += int(fp.sum()); fn_total += int(fn.sum()); tp_total += int(tp.sum())

print(f'\nПУЛ: tp {tp_total}  fp {fp_total}  fn {fn_total}  IoU_burn {tp_total/(tp_total+fp_total+fn_total):.4f}')
print('ЧИПЫ по ложной площади: чип, истинная гарь, fp, fn, доля fp в пуле, доля fn в пуле, маскировано')
for c, t, fp, fn, tp, masked in sorted(rows, key=lambda r: -r[2]):
    print(f'  {c:28s} гарь {t:7d}  fp {fp:6d} ({fp/fp_total:5.1%})  fn {fn:6d} ({fn/fn_total:5.1%})  маска {masked:5.1%}')
top5 = sum(sorted((r[2] for r in rows), reverse=True)[:5])
print(f'  пять худших чипов держат {top5/fp_total:.1%} ложной площади')
print(f'\nЛОЖНАЯ ГАРЬ по покрову (WorldCover): ' + ', '.join(f'{k}: {v/fp_total:.1%}' for k, v in sorted(fp_lc.items(), key=lambda kv: -kv[1])))
print(f'ЛОЖНАЯ ГАРЬ в 3 пикселях от маски облаков: {fp_near_cloud/fp_total:.1%}')
