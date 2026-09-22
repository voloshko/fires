"""SPEC-68 ступень 1: где сидят ошибки AF. Пятикратный OOF по 336 чипам (train+val) текущим рецептом
бустинга; порог — из data/comp/af_oof_cutoff.json. По чипам: концентрация FP/FN, ночь/день,
землепользование, спектр ошибок (I1 глинт, I3/I4 факел), сенсорный зенит."""
import sys, json, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from collections import Counter
from src.comp.af import AfDataset, build_training_set, train, features, f1, NAMES, THRESHOLDS
d = AfDataset('data/comp/train/af'); sp = json.load(open('data/comp/split_af.json')); chips = sorted(sp['train'] + sp['val']); K = 5; folds = [chips[i::K] for i in range(K)]
cut = json.load(open('data/comp/af_oof_cutoff.json')).get('oof_cutoff', 0.5); print(f'порог OOF {cut}')
rows = []; t0 = time.time(); FPpix, TPpix, FNpix = [], [], []
for k in range(K):
    fit = [c for j, f in enumerate(folds) if j != k for c in f]; x, y = build_training_set(d, fit); m = train(x, y)
    for c in folds[k]:
        ch = d.load(c); ok = ch.valid(); v = ch.viirs; p = m.predict_proba(features(ch).reshape(len(NAMES), -1).T)[:, 1].reshape(ok.shape); p[~ok] = 0
        pr = p >= cut; t = ch.mask > 0; fp = pr & ~t; fn = ~pr & t; tp = pr & t
        night = float(np.nanmedian(v[5][ok])) > 90 if ok.any() else False
        rows.append(dict(chip=c, fp=int(fp.sum()), fn=int(fn.sum()), tp=int(tp.sum()), truth=int(t.sum()), night=night, senzen=float(np.nanmedian(v[6][ok])) if ok.any() else np.nan,
                         lc_fp=Counter(ch.aux[0][fp].astype(int).tolist()).most_common(2)))
        for arr, sel in ((FPpix, fp), (TPpix, tp), (FNpix, fn)):
            if sel.any(): arr.append(np.stack([v[0][sel], v[1][sel], v[2][sel], v[3][sel], v[4][sel]], 1))
    print(f'фолд {k+1}/{K} за {time.time()-t0:.0f} с', flush=True)
FP = sum(r['fp'] for r in rows); FN = sum(r['fn'] for r in rows); TP = sum(r['tp'] for r in rows)
print(f'\nвсего: TP {TP}, FP {FP}, FN {FN}, F1 {2*TP/(2*TP+FP+FN):.4f}; чипов {len(rows)}, ночных {sum(r["night"] for r in rows)}')
rows.sort(key=lambda r: -r['fp']); cum = np.cumsum([r['fp'] for r in rows]) / max(FP, 1)
print(f'FP: топ-5 чипов держат {cum[4]:.0%}, топ-10 {cum[9]:.0%}, топ-20 {cum[19]:.0%}')
for r in rows[:10]: print(f"  {r['chip']} FP {r['fp']:5d} FN {r['fn']:4d} истина {r['truth']:5d} {'ночь' if r['night'] else 'день '} зенит {r['senzen']:.0f} покров FP {r['lc_fp']}")
rows.sort(key=lambda r: -r['fn']); cum = np.cumsum([r['fn'] for r in rows]) / max(FN, 1)
print(f'FN: топ-5 чипов держат {cum[4]:.0%}, топ-10 {cum[9]:.0%}, топ-20 {cum[19]:.0%}')
for r in rows[:10]: print(f"  {r['chip']} FN {r['fn']:5d} FP {r['fp']:4d} истина {r['truth']:5d} {'ночь' if r['night'] else 'день '} зенит {r['senzen']:.0f}")
FPp, TPp, FNp = map(np.concatenate, (FPpix, TPpix, FNpix))
def q(a): return np.nanpercentile(a, [10, 50, 90]).round(3).tolist()
print('\nспектр по пикселям (перцентили 10/50/90):')
for name, arr in (('TP', TPp), ('FP', FPp), ('FN', FNp)):
    print(f'  {name}: I1 {q(arr[:,0])} I3 {q(arr[:,2])} I4 {q(arr[:,3])} I4−I5 {q(arr[:,3]-arr[:,4])} I3/I4×1000 {q(1000*arr[:,2]/np.maximum(arr[:,3],1))}')
print(f'глинт-кандидаты среди FP (I1 > 0.3 и I1 > I2): {((FPp[:,0] > 0.3) & (FPp[:,0] > FPp[:,1])).mean():.1%}; среди TP: {((TPp[:,0] > 0.3) & (TPp[:,0] > TPp[:,1])).mean():.1%}')
print(f'ночные FP: {sum(r["fp"] for r in rows if r["night"])/max(FP,1):.0%} FP при {sum(r["night"] for r in rows)/len(rows):.0%} ночных чипов; ночные FN: {sum(r["fn"] for r in rows if r["night"])/max(FN,1):.0%}')
