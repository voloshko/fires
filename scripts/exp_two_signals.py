"""SPEC-62/63: два слабых сигнала вместо одного сильного.
62 — пятно бустинга вне выхода смеси принимается, если средняя p сиама > s (и p бустинга > τ);
63 — пиксель считается гарью, если p бустинга > τb и p сиама > τs, независимо от оптики.
Рецепт v22-аналог; выбор порогов по фолдам 0–2, проверка — фолды 3–4 и 35 чипов."""
import sys, json, hashlib, itertools, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import label, find_objects
from src.comp.chips import BsDataset
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
def conf(t, p): return np.bincount((t.astype(int) * 4 + p.astype(int)).ravel(), minlength=16).reshape(4, 4)
def metric(C):
    tot = C.sum(); burn = C[1:, 1:].sum() / max(tot - C[0, 0], 1); ious = []
    for k in (1, 2, 3):
        u = C[k].sum() + C[:, k].sum() - C[k, k]
        if u > 0: ious.append(C[k, k] / u)
    return (0.35 * burn + 0.30 * np.mean(ious)) / 0.65
def recipe(pb, po, ps, ok, zero, extra=None):
    pn = 0.5 * po + 0.5 * ps; P = 0.4 * pb + 0.6 * pn; burn = P.argmax(2) > 0; burn[~ok] = (pn.argmax(2) > 0)[~ok]
    if extra is not None: burn = burn | extra
    out = np.where(burn, P[..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[zero] = 0
    return drop_far(out, anchor=(po.argmax(2) > 0) & (ps.argmax(2) > 0))
def load_fold(f):
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']
    return ids, np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32), np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32), np.load(OWN / f'bs-confirm-siam-f{f}-sar-v1/probabilities.npy').astype(np.float32)
def load_35():
    s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
    tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
    PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
    return tune, np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32), PO, np.load('research/bs-siam-sar-screen-20260918/probabilities.npy').astype(np.float32)
def table(ids, PB, PO, PS):
    rows = []
    for i, c in enumerate(ids):
        chip = d.load(c); ok, zero = chip.valid(), chip.label_zero(); out = recipe(PB[i], PO[i], PS[i], ok, zero)
        bb = (PB[i].argmax(2) > 0) & ~zero; lab, n = label(bb); comps = []
        for j, sl in enumerate(find_objects(lab), 1):
            if sl is None: continue
            m = lab == j
            if m.sum() < 20 or (m & (out > 0)).any(): continue
            comps.append(dict(mask=m, ps=float((1 - PS[i][..., 0])[m].mean()), pb=float((1 - PB[i][..., 0])[m].mean()), sev=np.where(m, PB[i][..., 1:].argmax(2) + 1, 0).astype(np.uint8)))
        rows.append(dict(T=chip.mask, out=out, comps=comps, pb=1 - PB[i][..., 0], ps=1 - PS[i][..., 0], PB=PB[i], PO=PO[i], PS=PS[i], ok=ok, zero=zero))
    return rows
def run62(rows, s, t):
    C = np.zeros((4, 4), int); Cb = np.zeros((4, 4), int); lost = 0; lost0 = 0
    for r in rows:
        o = r['out'].copy()
        for k in r['comps']:
            if k['ps'] > s and k['pb'] > t: o = np.where(k['mask'], k['sev'], o)
        C += conf(r['T'], o); Cb += conf(r['T'], r['out']); tt = r['T'] > 0; lost += (((tt & (o > 0)).sum() / max((tt | (o > 0)).sum(), 1)) < 0.3); lost0 += (((tt & (r['out'] > 0)).sum() / max((tt | (r['out'] > 0)).sum(), 1)) < 0.3)
    return metric(C) - metric(Cb), lost, lost0
def run63(rows, tb, ts):
    C = np.zeros((4, 4), int); Cb = np.zeros((4, 4), int); lost = 0
    for r in rows:
        extra = (r['pb'] > tb) & (r['ps'] > ts) & r['ok']; o = recipe(r['PB'], r['PO'], r['PS'], r['ok'], r['zero'], extra)
        C += conf(r['T'], o); Cb += conf(r['T'], r['out']); tt = r['T'] > 0; lost += (((tt & (o > 0)).sum() / max((tt | (o > 0)).sum(), 1)) < 0.3)
    return metric(C) - metric(Cb), lost
folds = [table(*load_fold(f)) for f in range(5)]; allr = sum(folds, []); r35 = table(*load_35())
lost0 = run62(allr, 9, 9)[2]; lost35 = run62(r35, 9, 9)[2]
print(f'v22-аналог: потеряно {lost0} (фолды), {lost35} (35 чипов)')
print('\nSPEC-62 — пятно по отклику сиама (s) и бустинга (τ); выбор на фолдах 0–2, Δ взв:')
S = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5); TB = (0.5, 0.7)
sel = {(s, t): run62(sum(folds[:3], []), s, t)[0] for s, t in itertools.product(S, TB)}
for t in TB: print(f'  τ={t}: ' + '  '.join(f's={s:.2f} {sel[(s, t)]:+.4f}' for s in S))
s, t = max(sel, key=sel.get); chk, _, _ = run62(sum(folds[3:], []), s, t); pool, lost, _ = run62(allr, s, t); g35, l35, _ = run62(r35, s, t)
pf = [run62(folds[f], s, t)[0] for f in range(5)]
print(f'  лучшее s={s} τ={t}: выбор {sel[(s, t)]:+.4f}, проверка ф3–4 {chk:+.4f}, пул 5 ф {pool:+.4f} (по фолдам {np.round(pf, 4).tolist()}), потеряно {lost0} → {lost} | 35 чипов {g35:+.4f}, потеряно {lost35} → {l35}')
print('  КРИТЕРИЙ SPEC-62:', 'ПРОЙДЕН' if (pool >= 0.004 and chk >= 0 and g35 >= 0 and lost < lost0) else 'НЕ ПРОЙДЕН')
print('\nSPEC-63 — пиксель: гарь при p бустинга > τb и p сиама > τs; выбор на фолдах 0–2, Δ взв:')
TBB = (0.5, 0.6, 0.7, 0.8); TS = (0.1, 0.2, 0.3, 0.4, 0.5)
sel3 = {(tb, ts): run63(sum(folds[:3], []), tb, ts)[0] for tb, ts in itertools.product(TBB, TS)}
for tb in TBB: print(f'  τb={tb}: ' + '  '.join(f'τs={ts:.1f} {sel3[(tb, ts)]:+.4f}' for ts in TS))
tb, ts = max(sel3, key=sel3.get); chk3, _ = run63(sum(folds[3:], []), tb, ts); pool3, lost3 = run63(allr, tb, ts); g353, l353 = run63(r35, tb, ts)
pf3 = [run63(folds[f], tb, ts)[0] for f in range(5)]
print(f'  лучшее τb={tb} τs={ts}: выбор {sel3[(tb, ts)]:+.4f}, проверка ф3–4 {chk3:+.4f}, пул 5 ф {pool3:+.4f} (по фолдам {np.round(pf3, 4).tolist()}), потеряно {lost0} → {lost3} | 35 чипов {g353:+.4f}, потеряно {lost35} → {l353}')
print('  КРИТЕРИЙ SPEC-63:', 'ПРОЙДЕН' if (pool3 >= 0.004 and chk3 >= 0 and g353 >= 0 and lost3 < lost0) else 'НЕ ПРОЙДЕН')
