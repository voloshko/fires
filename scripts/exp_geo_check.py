"""Геометрическое усреднение сетей: по фолдам (выбор/проверка) и на 35 чипах, рецепт v20."""
import sys, json, hashlib, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs, score_bs_micro
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def gmean(ps, wts):
    L = np.log(np.clip(np.stack(ps), 1e-4, 1)); g = np.exp(np.tensordot(np.asarray(wts) / np.sum(wts), L, 1)); return g / g.sum(-1, keepdims=True)
def predict(PB, PO, PS, chips, geo):
    pn = gmean([PO, PS], [0.5, 0.5]) if geo else 0.5 * PO + 0.5 * PS
    P = 0.4 * PB + 0.6 * pn; OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    agree = (PO.argmax(3) > 0) & (PS.argmax(3) > 0)
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, agree)]), OK
F = {}; SEL = {}; CHK = {}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]; T = np.stack([c.mask for c in chips])
    PB, PO = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical')); SV = np.load(OWN / f'bs-confirm-siam-f{f}-over-v1/probabilities.npy').astype(np.float32)
    for geo in (False, True):
        out, OK = predict(PB, PO, SV, chips, geo); k = 'геом.' if geo else 'арифм.'
        F.setdefault(k, []).append(w(score_bs_micro(list(T), list(out)))); (SEL if f <= 2 else CHK).setdefault(k, ([], []))[0].append(T); (SEL if f <= 2 else CHK)[k][1].append(out)
def pooled(e): return w(score_bs(np.concatenate([t.reshape(-1) for t in e[0]]), np.concatenate([o.reshape(-1) for o in e[1]])))
for k in F: print(f'фолды {k:7s} по фолдам {np.round(F[k],4).tolist()} выбор {pooled(SEL[k]):.4f} проверка {pooled(CHK[k]):.4f}')
print('Δ геом−арифм по фолдам', np.round(np.array(F['геом.']) - np.array(F['арифм.']), 4).tolist())
# 35 чипов, рецепт v20-аналог: оптика (5 сетей) + передискр. сид скрининга
z = np.load('models/tune_proba_19.npz'); PB35, T35, OK35 = z['pb'].astype(np.float32), z['t'], z['ok']
s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
OPTS = [np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')]
OVR = np.load('research/bs-siam-over-screen-20260918/probabilities.npy').astype(np.float32)
for name, PO, PS in (('арифм. (оптика арифм. внутри)', np.mean(OPTS, 0), OVR), ('геом. (оптика геом. внутри)', gmean(OPTS, [1]*5), OVR)):
    geo = name.startswith('геом'); out, _ = predict(PB35, PO, PS, chips, geo); r = score_bs(T35.reshape(-1), out.reshape(-1)); p = out > 0; t = T35 > 0
    print(f'35 чипов {name:32s} {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f} | чистое небо {(t&p&OK35).sum()/((t|p)&OK35).sum():.4f}')
