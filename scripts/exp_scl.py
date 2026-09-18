"""Правило под маской по классам SCL (SPEC-19). Под тенью облака (3) и плотным
облаком (9) истинной гари нет вовсе — разметчик там обнулял; под cirrus (10) и
средним облаком (8) гарь размечена. Значит, ноль нужен не «под маской», а под
теми классами, где его ставил разметчик."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
z = np.load('models/tune_proba_19.npz'); PB, PN, T, OK = z['pb'].astype(np.float32), z['pn'].astype(np.float32), z['t'], z['ok']
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
chips = [d.load(c) for c in tune]
pre = np.stack([c.pre[9] for c in chips]); post = np.stack([c.post[9] for c in chips])
p = (0.4*PB + 0.6*PN).argmax(3); p[~OK] = PN.argmax(3)[~OK]; t = T > 0; fp = (p > 0) & ~t
print('класс  | pre: пикс  гарь  fp | post: пикс  гарь  fp')
for k in range(12):
    a, b = pre == k, post == k
    print(f'SCL {k:2d} | {int(a.sum()):8d} {int((t&a).sum()):7d} {int((fp&a).sum()):6d} | {int(b.sum()):8d} {int((t&b).sum()):7d} {int((fp&b).sum()):6d}')
def sc(pred):
    r = score_bs_micro(list(T), list(pred))
    return f"{r['iou_burn']:.4f}/{r['miou_sev']:.4f} взв {(0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65:.4f}"
print('текущее правило (сеть под всей маской):', sc(p))
for zero in [(0,), (0, 3), (0, 3, 9), (0, 3, 9, 11), (0, 1, 3, 9, 11), (0, 3, 8, 9, 11), (0, 3, 9, 10, 11)]:
    q = p.copy(); q[np.isin(pre, zero) | np.isin(post, zero)] = 0
    print(f'ноль в SCL {zero}:', sc(q))
