"""Вариант сиама на 35 чипах против передискр. сида с тем же seed (парно), рецепт v21. Аргумент: тег каталога (2st, sar, …)."""
import sys, json, hashlib, numpy as np; VAR = sys.argv[1]; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
for r in ('bs-boost-swir-screen-v1', 'bs-siam-over-screen-20260918', f'bs-siam-{VAR}-screen-20260918'):
    assert json.load(open(f'research/{r}/data_manifest.json'))['evaluation'] == tune, r
OPT = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')], 0)
PB = np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32)
OVR = np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
S2 = np.load(f'research/bs-siam-{VAR}-screen-20260918/probabilities.npy').astype(np.float32)
t = T > 0
def measure(ps, name):
    pn = 0.5 * OPT + 0.5 * ps; P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    out = np.stack([drop_far(o, anchor=a) for o, a in zip(out, (OPT.argmax(3) > 0) & (ps.argmax(3) > 0))])
    r = score_bs_micro(list(T), list(out)); p = out > 0; ra = score_bs_micro(list(T), list(np.where(ps.argmax(3) > 0, ps[..., 1:].argmax(3) + 1, 0)))
    w = lambda r: (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
    print(f'{name:40s} сеть одна {w(ra):.4f} | рецепт {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f} | чистое небо {(t&p&OK).sum()/((t|p)&OK).sum():.4f}')
measure(OVR, 'v21: передискр. сид 20260918')
measure(S2, f'{VAR}, сид 20260918')
