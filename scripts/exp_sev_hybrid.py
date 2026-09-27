"""SPEC-89: гибрид степени (граница C1-MM ≥ 0.5, градация dNBR внутри) против одних порогов dNBR на свежих 3 и 4 против тяжести MTBS."""
import json, sys, numpy as np, rasterio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.hls_eval import c1_probs
R = Path('research'); MM = [R / f'hls-c1mm-final-s{s}' for s in (1, 2, 3, 4, 5)]; Cs = {'гибрид': [], 'dNBR': []}
def conf(T, V, Pr):
    return np.stack([np.bincount(T[i][V[i]] * 4 + Pr[i][V[i]], minlength=16).reshape(4, 4) for i in range(len(T))])
def miou(C):
    S = C.sum(0); iou = [S[k, k] / max(S[k, :].sum() + S[:, k].sum() - S[k, k], 1) for k in range(4)]; return float(np.mean(iou[1:])), iou
for d in ('external/hls_fresh3', 'external/hls_fresh4'):
    D = Path(d); sev = {w['name'] for w in json.load(open(D / 'manifest_sev.json'))['windows']}
    man = [w for w in json.load(open(D / 'manifest_sev_pre.json'))['windows'] if w['name'] in sev]
    rd = lambda w, k: rasterio.open(D / f"{w['name']}{k}").read()
    X = np.stack([rd(w, '_merged.tif').astype(np.float32) for w in man]); XB = np.stack([rd(w, '.pre.tif').astype(np.float32) for w in man])
    M = np.stack([rd(w, '.mask.tif')[0] for w in man]); SV = np.stack([rd(w, '.sev.tif')[0] for w in man])
    T = np.select([SV == 2, SV == 3, SV == 4], [1, 2, 3], 0); V = (M >= 0) & (SV != 6) & (X != -9999).all(1) & (XB != -9999).all(1)
    X[np.broadcast_to(~V[:, None], X.shape)] = 0
    P = c1_probs(X, MM, R / f"{D.name}-sev-v1" / 'c1mm.npy') >= 0.5
    nbr = lambda A: (A[:, 3] - A[:, 5]) / (A[:, 3] + A[:, 5] + 1e-6); dn = nbr(XB) - nbr(X)
    Bd = np.digitize(dn, [0.1, 0.27, 0.66]); Hy = np.where(P, np.maximum(np.digitize(dn, [0.27, 0.66]) + 1, 1), 0)
    ch, cb = conf(T, V, Hy), conf(T, V, Bd); Cs['гибрид'].append(ch); Cs['dNBR'].append(cb)
    print(f'{D.name}: окон {len(T)} | гибрид mIoU_sev {miou(ch)[0]:.4f} {np.round(miou(ch)[1], 4).tolist()} | dNBR {miou(cb)[0]:.4f} {np.round(miou(cb)[1], 4).tolist()}', flush=True)
ch, cb = np.concatenate(Cs['гибрид']), np.concatenate(Cs['dNBR']); rng = np.random.default_rng(0); n = len(ch)
d = [miou(ch[i])[0] - miou(cb[i])[0] for i in (rng.integers(0, n, n) for _ in range(1000))]; lo, hi = np.percentile(d, [2.5, 97.5]); pt = miou(ch)[0] - miou(cb)[0]
b = lambda C: (lambda S: S[1:, 1:].sum() / (S.sum() - S[0, 0]))(C.sum(0))
print(f'вместе ({n} окон): гибрид {miou(ch)[0]:.4f} (бинарный {b(ch):.4f}) | dNBR {miou(cb)[0]:.4f} (бинарный {b(cb):.4f}) | {pt:+.4f}, 95 % [{lo:+.4f}, {hi:+.4f}] → {"ЛУЧШЕ" if pt > 0 and lo > 0 else "ХУЖЕ" if hi < 0 else "ВРОВЕНЬ"}')
