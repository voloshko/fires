"""Независимый замер сиамских сетей соседа (SPEC-32) в НАШЕЙ шкале: бустинг 0.4,
правило SCL, под маской сеть, фильтр чужих пожаров; все пиксели и чистое небо."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
HYP = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / 'fires-hypotheses'))
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO = np.stack([c.label_zero() for c in chips])
his = json.load(open(HYP / 'research/bs-siam-20260918-v3/per_chip.json'))
his_ids = [r['chip'] if isinstance(r, dict) else r for r in (his if isinstance(his, list) else his.get('chips', list(his.keys())))]
print('порядок чипов совпадает с нашим:', his_ids[:35] == tune if len(his_ids) >= 35 else f'? ({len(his_ids)})')
OPT = [np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')]
SIAM = [np.load(HYP / f'research/{r}/probabilities.npy').astype(np.float32) for r in ('bs-siam-20260918-v3','bs-siam-20260919-v3')]
t = T > 0
def measure(pn, name):
    P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; out = np.stack([drop_far(o) for o in out])
    r = score_bs_micro(list(T), list(out)); p = out > 0
    clear = (t & p & OK).sum() / ((t | p) & OK).sum()
    print(f'{name:36s} все: {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f} | чистое небо IoU {clear:.4f}')
measure(np.mean(OPT, 0), 'v13: оптическая пятёрка')
measure(np.mean(SIAM, 0), 'две сиамские (сосед)')
measure(np.mean(OPT + SIAM, 0), 'пятёрка + две сиамские, равные веса')
measure(0.5*np.mean(OPT, 0) + 0.5*np.mean(SIAM, 0), 'пятёрка и сиамские 50/50')
for w in (0.3, 0.4):
    measure((1-w)*np.mean(OPT, 0) + w*np.mean(SIAM, 0), f'пятёрка {1-w:.1f} + сиамские {w:.1f}')
