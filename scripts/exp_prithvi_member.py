"""SPEC-64: дообученный Prithvi как член смеси. Один, разнообразие (DF/Q), и смесь: R1 — вместо оптики,
R2 — третий член поровну, R3 — третий член с весом 0.25. Выбор варианта по фолдам 0–2, проверка 3–4."""
import sys, json, argparse, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs
from src.comp.postproc import drop_far
ap = argparse.ArgumentParser(); ap.add_argument('--select', default='0,1,2'); ap.add_argument('--check', default='3,4'); A = ap.parse_args(); SEL = [int(x) for x in A.select.split(',')]; CHK = [int(x) for x in A.check.split(',')]
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research'); d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def rec(PB, PO, PS, PR, OK, ZERO, var):
    pn = {'v22': 0.5 * PO + 0.5 * PS, 'R1 вместо оптики': 0.5 * PR + 0.5 * PS, 'R2 третий поровну': (PO + PS + PR) / 3, 'R3 третий 0.25': 0.25 * PO + 0.5 * PS + 0.25 * PR}[var]
    P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    anc = (PO.argmax(3) > 0) & (PS.argmax(3) > 0) if var != 'R1 вместо оптики' else (PR.argmax(3) > 0) & (PS.argmax(3) > 0)
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, anc)])
VARS = ['v22', 'R1 вместо оптики', 'R2 третий поровну', 'R3 третий 0.25']; F = {v: [] for v in VARS}; FOLDS = []; T_all, O_all, L, ALONE, E = [], {v: [] for v in VARS}, {v: 0 for v in VARS}, [], {'оптика': [], 'сиам': [], 'бустинг': [], 'prithvi': []}; OKs = []
for f in range(5):
    q = OWN / f'bs-confirm-prithvi-f{f}-v1/probabilities.npy'
    if not q.exists(): continue
    ids = json.load(open(HYP / f'research/bs-confirm-siam-f{f}-v1/data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]; T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    PB = np.load(OWN / f'bs-confirm-boost-f{f}-swir-v1/probabilities.npy').astype(np.float32); PO = np.load(HYP / f'research/bs-confirm-optical-f{f}-v1/probabilities.npy').astype(np.float32); PS = np.load(OWN / f'bs-confirm-siam-f{f}-sar-v1/probabilities.npy').astype(np.float32); PR = np.load(q).astype(np.float32)
    FOLDS.append(f); T_all.append(T); OKs.append(OK); ALONE.append(np.where(PR.argmax(3) > 0, PR[..., 1:].argmax(3) + 1, 0))
    for name, P in (('оптика', PO), ('сиам', PS), ('бустинг', PB), ('prithvi', PR)): E[name].append((P.argmax(3) > 0) != (T > 0))
    for v in VARS:
        o = rec(PB, PO, PS, PR, OK, ZERO, v); O_all[v].append(o); F[v].append(w(score_bs(T.reshape(-1), o.reshape(-1))))
        for tt, oo in zip(T, o): L[v] += ((((tt > 0) & (oo > 0)).sum() / max(((tt > 0) | (oo > 0)).sum(), 1)) < 0.3)
n = len(T_all); Tc = np.concatenate([t.reshape(-1) for t in T_all])
ra = score_bs(Tc, np.concatenate([a.reshape(-1) for a in ALONE])); print(f'prithvi дообученный, один ({n} ф.): взв {w(ra):.4f}, IoU гари {ra["iou_burn"]:.4f}, mIoU {ra["miou_sev"]:.4f}')
def pair(a, b):
    df, qs = [], []
    for A, B, OK in zip(a, b, OKs):
        for x, y, ok in zip(A, B, OK):
            if ok.sum() < 100: continue
            x, y = x[ok], y[ok]; dd = (x & y).mean(); aa = (~x & ~y).mean(); bb = (~x & y).mean(); cc = (x & ~y).mean(); df.append(dd); den = aa * dd + bb * cc; qs.append((aa * dd - bb * cc) / den if den > 0 else 0)
    return np.mean(df) * 100, np.mean(qs)
for other in ('оптика', 'сиам', 'бустинг'): df, q = pair(E['prithvi'], E[other]); print(f'  double-fault / Q с {other}: {df:.2f}% / {q:.3f}')
dfo, _ = pair(E['оптика'], E['сиам']); print(f'  (справочно: оптика–сиам {dfo:.2f}%)')
base = np.array(F['v22']); pos = {f: i for i, f in enumerate(FOLDS)}
def mean_over(dv, fl): idx = [pos[f] for f in fl if f in pos]; return float(np.mean(dv[idx])) if idx else float('nan')
poolv22 = w(score_bs(Tc, np.concatenate([o.reshape(-1) for o in O_all['v22']])))
res = {}
for v in VARS:
    r = score_bs(Tc, np.concatenate([o.reshape(-1) for o in O_all[v]])); dv = np.array(F[v] ) - base; res[v] = (w(r), dv)
    print(f'{v:20s} пул {w(r):.4f} ({w(r)-poolv22:+.4f}) | Δ по фолдам {dict(zip(FOLDS, np.round(dv, 4).tolist()))} | выбор ф{SEL} {mean_over(dv, SEL):+.4f}, проверка ф{CHK} {mean_over(dv, CHK):+.4f} | потеряно {L[v]}')
cands = ['R2 третий поровну', 'R3 третий 0.25']; best = max(cands, key=lambda v: mean_over(res[v][1], SEL)); dv = res[best][1]
chk_vals = [dv[pos[f]] for f in CHK if f in pos]
ok = len(chk_vals) == len(CHK) and np.mean(chk_vals) >= 0 and min(chk_vals) >= -0.01 and res[best][0] - poolv22 >= 0.004 and L[best] <= L['v22']
print(f'SPEC-65: выбран {best} по фолдам {SEL}; слепая проверка {CHK}: {np.round(chk_vals, 4).tolist()} | пул {res[best][0]-poolv22:+.4f} | потеряно {L[best]} против {L["v22"]} | КРИТЕРИЙ:', 'ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН')
