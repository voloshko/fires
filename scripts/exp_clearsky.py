"""IoU одинаковыми линейками (SPEC-19): бенчмарки маскируют облака и меряют
только чистое небо; метрика кейса считает все пиксели. Один ансамбль — две
шкалы, чтобы сравнение с CEMS-Wildfire было корректным."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
tags = sys.argv[1:] or ['d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1']
pn = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in tags], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO = np.stack([c.label_zero() for c in chips])
P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; out = np.stack([drop_far(o) for o in out])
r = score_bs_micro(list(T), list(out)); print(f'все пиксели (как в кейсе):             IoU_burn {r["iou_burn"]:.4f}  mIoU_sev {r["miou_sev"]:.4f}')
t, p = T > 0, out > 0
def iou(m): return (t & p & m).sum() / ((t | p) & m).sum()
def iou_c(c, m): a = (T == c) & m; b = (out == c) & m; return (a & b).sum() / (a | b).sum()
print(f'только чистое небо (как у бенчмарков): IoU_burn {iou(OK):.4f}  mIoU_sev {np.mean([iou_c(c, OK) for c in (1, 2, 3)]):.4f}   доля пикселей {OK.mean():.1%}')
print(f'только под маской SCL:                 IoU_burn {iou(~OK):.4f}')
