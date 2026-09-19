"""Отбор членов ансамбля (SPEC-19). Жадный прямой отбор по взвешенной метрике на
одной половине настроечных чипов, замер на другой — иначе отбор на тех же
чипах, где меряем, завысит результат. Пул — все сети-аналоги, обученные на 144."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from scipy.ndimage import label, distance_transform_edt as edt
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
TAGS = sys.argv[1:] or ['d7w32','d7s1','d7s2','d7fast','d7rot','d7lov','d7lov_s1','d7bnd','d7bnd_s1','d7jit','d7jit_s1','d7iz','d7iz_s1','d7ps','d7ps_s1']
PROBS = {t: np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in TAGS}
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
ZERO = np.stack([d.load(c).label_zero() for c in tune])
def predict(members):
    pn = np.mean([PROBS[t] for t in members], 0); P = 0.4*PB + 0.6*pn
    burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    return np.stack([drop_far(o) for o in out])
def sc(pred, idx):
    q = score_bs_micro([T[i] for i in idx], [pred[i] for i in idx]); return (0.35*q['iou_burn']+0.30*q['miou_sev'])/0.65
cache = {}
def score(members, idx):
    key = tuple(sorted(members))
    if key not in cache: cache[key] = predict(members)
    return sc(cache[key], idx)
def greedy(idx):
    chosen = []; best = -1
    while True:
        cand = [(score(chosen + [t], idx), t) for t in TAGS if t not in chosen]
        if not cand: break
        v, t = max(cand)
        if v <= best + 1e-4: break
        chosen.append(t); best = v
    return chosen, best
allidx = list(range(35))
print('все %d сетей: %.4f' % (len(TAGS), score(TAGS, allidx)))
full, v = greedy(allidx); print('жадный отбор на всех 35 (ОПТИМИСТИЧНО):', full, '%.4f' % v)
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed); perm = rng.permutation(35); halves = (list(perm[:17]), list(perm[17:]))
    for fit_h, ev_h in (halves, halves[::-1]):
        chosen, _ = greedy(fit_h)
        print(f'честно (сид {seed}): отобрано {chosen} → на другой половине {score(chosen, ev_h):.4f} против всех: {score(TAGS, ev_h):.4f} ({score(chosen, ev_h)-score(TAGS, ev_h):+.4f})')
