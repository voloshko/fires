"""Правило по meta.cloud_frac (SPEC-29): cloud_frac посчитан по сцене РАЗМЕТКИ.
Если он совпадает с долей нашей SCL-маски — сцены одни и те же, под маской
истины нет; если расходится — сцены разные, под маской истина есть. Из кэша."""
import sys, json, hashlib, numpy as np, pandas as pd; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
pn = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
meta = pd.read_csv('data/comp/train/bs/meta.csv').set_index('chip_id')
ZERO = np.stack([c.label_zero() for c in chips])
P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
base = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); base[ZERO] = 0; base = np.stack([drop_far(o) for o in base])
print('чип            meta_cloud  scl_cloud  |разн|  истины под маской  fp под маской  fn под маской')
rows = []
for i, c in enumerate(tune):
    mc = meta.loc[c, 'cloud_frac']; sc = (~OK[i]).mean(); t = T[i] > 0; m = ~OK[i]
    tm = int((t & m).sum()); fpm = int(((base[i] > 0) & ~t & m).sum()); fnm = int((t & (base[i] == 0) & m).sum())
    rows.append((c, mc, sc, abs(mc - sc), tm, fpm, fnm))
for r in sorted(rows, key=lambda r: r[3]): print(f'{r[0]}  {r[1]:.3f}      {r[2]:.3f}    {r[3]:.3f}   {r[4]:8d}          {r[5]:6d}        {r[6]:6d}')
def sc_(pred): q = score_bs_micro(list(T), list(pred)); return f"{q['iou_burn']:.4f}/{q['miou_sev']:.4f} взв {(0.35*q['iou_burn']+0.30*q['miou_sev'])/0.65:.4f}"
print('\nбаза (сеть под маской везде):', sc_(base))
for tol in (0.02, 0.05, 0.1, 0.15):
    q = base.copy()
    for i, c in enumerate(tune):
        if abs(meta.loc[c, 'cloud_frac'] - (~OK[i]).mean()) <= tol: q[i][~OK[i]] = 0
    n = sum(abs(meta.loc[c, 'cloud_frac'] - (~OK[i]).mean()) <= tol for i, c in enumerate(tune))
    print(f'ноль под маской, если |meta_cloud − scl_cloud| ≤ {tol} ({n} чипов):', sc_(q))
# альтернатива: ноль под маской, если meta_cloud > 0.05 (сцена разметки сама облачная)
for thr in (0.02, 0.05, 0.1):
    q = base.copy()
    for i, c in enumerate(tune):
        if meta.loc[c, 'cloud_frac'] > thr: q[i][~OK[i]] = 0
    print(f'ноль под маской, если meta_cloud > {thr}:', sc_(q))
