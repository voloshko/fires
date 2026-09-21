"""SPEC-47 на 35 чипах: бустинг с SWIR-признаками против базового, рецепт v20-аналог."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
assert json.load(open('research/bs-boost-swir-screen-v1/data_manifest.json'))['evaluation'] == tune
T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
PO = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0)
OVR = np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
PBn = np.load(HYP / 'research/bs-boost-v1/probabilities.npy').astype(np.float32)
PBo = np.load('models/tune_proba_19.npz')['pb'].astype(np.float32)
PBs = np.load('research/bs-boost-swir-screen-v1/probabilities.npy').astype(np.float32)
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def measure(PB, name):
    pn = 0.5 * PO + 0.5 * OVR; P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    out = np.stack([drop_far(o, anchor=a) for o, a in zip(out, (PO.argmax(3) > 0) & (OVR.argmax(3) > 0))])
    r = score_bs(T.reshape(-1), out.reshape(-1)); rb = score_bs(T.reshape(-1), np.where(PB.argmax(3) > 0, PB[..., 1:].argmax(3) + 1, 0).reshape(-1)); p = out > 0; t = T > 0
    print(f'{name:34s} бустинг один {w(rb):.4f} | рецепт {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f} | чистое небо {(t&p&OK).sum()/((t|p)&OK).sum():.4f}')
measure(PBo, 'наш бустинг 19 признаков (v20)')
measure(PBn, 'бустинг соседа, база (bs-boost-v1)')
measure(PBs, 'бустинг соседа + SWIR')
