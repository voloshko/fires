"""SPEC-43: передискр. сиамский сид против обычного, парно по фолдам, в рецепте v19."""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs, score_bs_micro
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
def w(r): return (0.35 * r['iou_burn'] + 0.30 * r['miou_sev']) / 0.65
def run(PB, PO, PS, chips):
    pn = 0.5 * PO + 0.5 * PS; P = 0.4 * PB + 0.6 * pn; OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0
    agree = (PO.argmax(3) > 0) & (PS.argmax(3) > 0)
    return np.stack([drop_far(o, anchor=a) for o, a in zip(out, agree)])
POOL, FOLDS, LOST = {}, {}, {}
for f in range(5):
    over = OWN / f'bs-confirm-siam-f{f}-over-v1'
    if not (over / 'probabilities.npy').exists(): continue
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips])
    PB, PO, PS1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    PS2 = np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32)
    PF = np.load(over / 'probabilities.npy').astype(np.float32)
    for name, PS in (('v19: сид1+сид2', 0.5*(PS1+PS2)), ('сид2 один (парная база)', PS2), ('передискр. один', PF), ('сид1+передискр.', 0.5*(PS1+PF)), ('сид1+сид2+передискр.', (PS1+PS2+PF)/3)):
        out = run(PB, PO, PS, chips); FOLDS.setdefault(name, []).append(w(score_bs_micro(list(T), list(out))))
        e = POOL.setdefault(name, ([], [])); e[0].append(T); e[1].append(out)
        for t, o in zip(T, out):
            iou = ((t > 0) & (o > 0)).sum() / max(((t > 0) | (o > 0)).sum(), 1); LOST[name] = LOST.get(name, 0) + (iou < 0.3)
base = np.array(FOLDS['v19: сид1+сид2'])
for name, v in FOLDS.items():
    Ts, Os = POOL[name]; r = score_bs(np.concatenate([t.reshape(-1) for t in Ts]), np.concatenate([o.reshape(-1) for o in Os]))
    print(f'{name:26s} пул {len(v)} ф: {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {w(r):.4f} | фолды {np.round(v,4).tolist()} Δ к v19 {np.round(np.array(v)-base,4).tolist()} | потеряно {LOST[name]}')
