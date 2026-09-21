"""SPEC-43 на 35 настроечных чипах: передискретизованный сид против сида соседа с тем же seed (парно)."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO = np.stack([c.label_zero() for c in chips])
over_ids = json.load(open('research/bs-siam-over-screen-20260918/data_manifest.json'))['evaluation']
assert over_ids == tune, 'порядок чипов скрининга не совпал'
OPT = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0)
S18 = np.load(HYP / 'research/bs-siam-20260918-v3/probabilities.npy').astype(np.float32)
S19 = np.load(HYP / 'research/bs-siam-20260919-v3/probabilities.npy').astype(np.float32)
OVR = np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
t = T > 0
def measure(ps, name):
    pn = 0.5 * OPT + 0.5 * ps; P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    agree = (OPT.argmax(3) > 0) & (ps.argmax(3) > 0); out = np.stack([drop_far(o, anchor=a) for o, a in zip(out, agree)])
    r = score_bs_micro(list(T), list(out)); p = out > 0
    print(f'{name:44s} {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f} | чистое небо {(t&p&OK).sum()/((t|p)&OK).sum():.4f}')
measure(S18, 'сиам сид 20260918 (парная база)')
measure(OVR, 'передискр. сид 20260918')
measure(0.5 * (S18 + S19), 'v19-аналог: два сида соседа')
measure(0.5 * (OVR + S19), 'передискр. + сид 20260919')
