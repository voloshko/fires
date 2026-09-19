"""Бустинг, обученный и на пикселях под маской 8/10 (SPEC-19). Сейчас он
учится только на валидных, а применяется под маской для степени. Гипотеза:
обучение на облачных пикселях с истиной улучшит его под маской."""
import sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.features import stack, NAMES
from src.comp.metric import score_bs_micro
from src.comp.model import train, SEED, PIXELS_PER_CHIP
z = np.load('models/tune_proba_19.npz'); T, OK = z['t'], z['ok']; PB_old = z['pb'].astype(np.float32)
PN = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1')], 0)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest()); tune, fit = sorted(rank[:35]), sorted(rank[35:])
chips_t = [d.load(c) for c in tune]; ZERO = np.stack([c.label_zero() for c in chips_t])
rng = np.random.default_rng(SEED); xs, ys = [], []
t0 = time.time()
for c in fit:
    ch = d.load(c); f = stack(ch).reshape(len(NAMES), -1).T; y = ch.mask.ravel(); keep = ~ch.label_zero().ravel()   # маска НЕ исключается
    for cls in range(4):
        idx = np.flatnonzero((y == cls) & keep)
        if idx.size: pick = rng.choice(idx, min(idx.size, PIXELS_PER_CHIP // 4), replace=False); xs.append(f[pick]); ys.append(y[pick])
m = train(np.concatenate(xs), np.concatenate(ys)); print(f'бустинг с облачными пикселями обучен за {time.time()-t0:.0f}с', flush=True)
PB_new = np.stack([m.predict_proba(stack(c).reshape(len(NAMES), -1).T).reshape(*c.shape, 4) for c in chips_t]).astype(np.float32)
def rule(PB, w_mask_burn=1.0):
    P = 0.4*PB + 0.6*PN; Pm = (1-w_mask_burn)*PB + w_mask_burn*PN
    burn = P.argmax(3) > 0; burn[~OK] = (Pm.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; return out
def sc(pred):
    r = score_bs_micro(list(T), list(pred)); return f"{r['iou_burn']:.4f}/{r['miou_sev']:.4f} взв {(0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65:.4f}"
print('старый бустинг (валидные пиксели):      ', sc(rule(PB_old)))
print('новый бустинг (+ облачные), гарь: сеть: ', sc(rule(PB_new)))
for w in (0.8, 0.6): print(f'новый, гарь под маской смесь w_net={w}:   ', sc(rule(PB_new, w)))
