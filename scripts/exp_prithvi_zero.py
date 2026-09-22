"""SPEC-54: нулевой выстрел Prithvi-EO-2.0-300M-BurnScars на 144 групповых чипах (сцена «после»).
Вход: B2 B3 B4 B8A B11 B12 / 10000, нормализация их means/stds; вариант A — родные 20 м (512),
вариант B — пересчёт к 30 м (341 → отражённое дополнение до 512). Метрики: IoU гари на чистом небе
(пул, фолды), потерянные пожары, double-fault и Q с оптикой, сиамом v22 и бустингом."""
import sys, json, time, numpy as np, torch; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import zoom
from src.comp.chips import BsDataset
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); EXT = Path('external/prithvi-eo2-300m-burnscars'); OUT = OWN / 'prithvi-zero-v1'; OUT.mkdir(parents=True, exist_ok=True)
d = BsDataset('data/comp/train/bs')
import yaml; cfg = yaml.safe_load(open(EXT / 'burn_scars_config.yaml')); MEAN = np.array(cfg['data']['init_args']['means'], np.float32); STD = np.array(cfg['data']['init_args']['stds'], np.float32)
from terratorch.cli_tools import LightningInferenceModel
lm = LightningInferenceModel.from_config(str(EXT / 'burn_scars_config.yaml'), str(EXT / 'Prithvi_EO_V2_300M_BurnScars.pt')); model = lm.model.eval().cuda()
BANDS = [0, 1, 2, 6, 7, 8]
def prob(chip, variant):
    post = (chip.post if chip.post.size else chip.pre).astype(np.float32)[BANDS] / 10000.0   # (6, 512, 512) отражение 0..1
    if variant == 'B':
        small = np.stack([zoom(b, 2 / 3, order=1) for b in post]); h = small.shape[-1]; pad = 512 - h
        post = np.pad(small, ((0, 0), (0, pad), (0, pad)), mode='reflect')
    x = (post - MEAN[:, None, None]) / STD[:, None, None]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(torch.from_numpy(x[None]).cuda()); logits = getattr(out, 'output', out).float()
    p = torch.softmax(torch.nn.functional.interpolate(logits, size=512, mode='bilinear'), 1)[0, 1].cpu().numpy()
    if variant == 'B': p = zoom(p[:h, :h], 512 / h, order=1)[:512, :512]
    return p.astype(np.float32)
def iou(a, b): u = (a | b).sum(); return (a & b).sum() / u if u else np.nan
def pair(ea, eb, oks):
    df, qs = [], []
    for a, b, ok in zip(ea, eb, oks):
        if ok.sum() < 100: continue
        a, b = a[ok], b[ok]; dd = (a & b).mean(); aa = (~a & ~b).mean(); bb = (~a & b).mean(); cc = (a & ~b).mean(); df.append(dd); den = aa * dd + bb * cc; qs.append((aa * dd - bb * cc) / den if den > 0 else 0)
    return np.mean(df), np.mean(qs)
t0 = time.time(); ids_all, T, OK, E = [], [], [], {'оптика v21': [], 'сиам v22': [], 'бустинг SWIR': [], 'A': [], 'B': []}; P = {'A': [], 'B': []}; per_fold = {'A': [], 'B': []}
for f in range(5):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']; ids_all += ids
    PO = np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy'); PS = np.load(OWN / f'bs-confirm-siam-f{f}-sar-v1/probabilities.npy'); PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy')
    fi = {'A': [0, 0], 'B': [0, 0]}
    for i, c in enumerate(ids):
        chip = d.load(c); t = chip.mask > 0; ok = chip.valid(); T.append(t); OK.append(ok)
        E['оптика v21'].append((PO[i].astype(np.float32).argmax(2) > 0) != t); E['сиам v22'].append((PS[i].astype(np.float32).argmax(2) > 0) != t); E['бустинг SWIR'].append((PB[i].astype(np.float32).argmax(2) > 0) != t)
        for v in ('A', 'B'):
            p = prob(chip, v); P[v].append(p.astype(np.float16)); pred = p > 0.5; E[v].append(pred != t); fi[v][0] += (t & pred & ok).sum(); fi[v][1] += ((t | pred) & ok).sum()
    for v in ('A', 'B'): per_fold[v].append(fi[v][0] / max(fi[v][1], 1))
    print(f'фолд {f} готов, {int(time.time()-t0)} с', flush=True)
json.dump(dict(ids=ids_all, bands=BANDS, variants={'A': '20 м, 512', 'B': '30 м (×2/3), отражённое дополнение'}, model='Prithvi_EO_V2_300M_BurnScars.pt', config='burn_scars_config.yaml'), open(OUT / 'manifest.json', 'w'), indent=1)
for v in ('A', 'B'): np.save(OUT / f'probabilities_{v}.npy', np.stack(P[v]))
print(f'\nPrithvi-EO-2.0 BurnScars, нулевой выстрел, 144 чипа, сцена «после», {int(time.time()-t0)} с')
for name in ('оптика v21', 'сиам v22', 'бустинг SWIR', 'A', 'B'):
    preds = [np.logical_xor(t, e) for t, e in zip(T, E[name])]
    num = sum((t & p & ok).sum() for t, p, ok in zip(T, preds, OK)); den = sum(((t | p) & ok).sum() for t, p, ok in zip(T, preds, OK))
    lost = sum(iou(t, p) < 0.3 for t, p in zip(T, preds)); err = np.mean([e[ok].mean() for e, ok in zip(E[name], OK) if ok.sum() >= 100])
    dfo, qo = pair(E[name], E['оптика v21'], OK); dfs, qs = pair(E[name], E['сиам v22'], OK); dfb, qb = pair(E[name], E['бустинг SWIR'], OK)
    label = {'A': 'Prithvi A (20 м)', 'B': 'Prithvi B (30 м)'}.get(name, name)
    extra = f' | по фолдам {np.round(per_fold[name], 3).tolist()}' if name in per_fold else ''
    print(f'{label:18s} IoU гари чистое небо {num/den:.4f} | своя ошибка {err*100:.2f}% | потеряно {lost} | DF/Q с оптикой {dfo*100:.2f}%/{qo:.3f}, с сиамом {dfs*100:.2f}%/{qs:.3f}, с бустингом {dfb*100:.2f}%/{qb:.3f}{extra}')
