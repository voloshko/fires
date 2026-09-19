"""Ревалидация сиамских сетей (SPEC-32) на групповых фолдах соседа в НАШЕЙ шкале.

Сосед сравнивает одну сеть с одной; наш сабмит — смесь оптики и сиамских сетей
поверх бустинга с правилом SCL и фильтром чужих пожаров. Здесь на каждом фолде
из его кэшей вероятностей собирается наш рецепт: оптика одна (аналог v13),
оптика + сиам (аналог v16), сиам одна. Чипы фолда не участвовали ни в обучении,
ни в выборе сиамского варианта — это и есть честная оценка прибавки.
"""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.chips import BsDataset
from src.comp.metric import score_bs, score_bs_micro
from src.comp.postproc import drop_far
HYP = Path(sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / 'fires-hypotheses'))
d = BsDataset('data/comp/train/bs')
def measure(PB, pn, T, OK, ZERO):
    P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; out = np.stack([drop_far(o) for o in out])
    r = score_bs_micro(list(T), list(out)); p = out > 0; t = T > 0
    return r['iou_burn'], r['miou_sev'], (0.35*r['iou_burn']+0.30*r['miou_sev'])/0.65, (t & p & OK).sum() / ((t | p) & OK).sum()
rows = {}
for f in range(5):
    dirs = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    if not all((v / 'probabilities.npy').exists() for v in dirs.values()): continue
    ids = json.load(open(dirs['siam'] / 'data_manifest.json'))['evaluation']
    assert ids == json.load(open(dirs['optical'] / 'data_manifest.json'))['evaluation'] == json.load(open(dirs['boost'] / 'data_manifest.json'))['evaluation']
    chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    PB, PO, PS = (np.load(dirs[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    for name, pn in (('оптика одна (v13)', PO), ('оптика+сиам 50/50', 0.5*PO+0.5*PS), ('оптика 5/7 + сиам 2/7 (v16)', (5*PO+2*PS)/7), ('сиам одна', PS)):
        r = measure(PB, pn, T, OK, ZERO); rows.setdefault(name, []).append(r)
        print(f'фолд {f} ({len(ids)} чипов) {name:30s} {r[0]:.4f}/{r[1]:.4f} взв {r[2]:.4f} | чистое небо {r[3]:.4f}', flush=True)
print()
base = np.array(rows.get('оптика одна (v13)', []))
for name, rs in rows.items():
    a = np.array(rs); dw = a[:, 2] - base[:, 2]; dc = a[:, 3] - base[:, 3]
    print(f'{name:30s} взв среднее {a[:,2].mean():.4f}  Δ к оптике по фолдам {np.round(dw,4).tolist()}  среднее {dw.mean():+.4f}  | чистое небо Δ {dc.mean():+.4f}, min {dc.min():+.4f}')

# Пул по всем фолдам (144 чипа) — так считает проверяющая система, а не среднее по фолдам.
print()
POOL = {}
for f in range(5):
    dirs = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    if not all((v / 'probabilities.npy').exists() for v in dirs.values()): continue
    ids = json.load(open(dirs['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    T = np.stack([c.mask for c in chips]); OK = np.stack([c.valid() for c in chips]); ZERO = np.stack([c.label_zero() for c in chips])
    PB, PO, PS = (np.load(dirs[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    for name, pn in (('оптика одна (v13)', PO), ('оптика+сиам 50/50', 0.5*PO+0.5*PS), ('оптика 5/7 + сиам 2/7 (v16)', (5*PO+2*PS)/7), ('сиам одна', PS)):
        P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
        out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; out = np.stack([drop_far(o) for o in out])
        e = POOL.setdefault(name, {'T': [], 'O': [], 'OK': []}); e['T'].append(T); e['O'].append(out); e['OK'].append(OK)
        if f == 4 and name in ('оптика одна (v13)', 'сиам одна'):
            for c, t, o, ok in zip(ids, T, out, OK):
                inter = ((t > 0) & (o > 0)).sum(); union = ((t > 0) | (o > 0)).sum()
                print(f'  фолд 4 {name[:6]} {c} истина {int((t>0).sum()):7d} пред {int((o>0).sum()):7d} пересечение {int(inter):7d} объединение {int(union):7d}')
for name, e in POOL.items():
    T = np.concatenate([x.reshape(-1) for x in e['T']]); O = np.concatenate([x.reshape(-1) for x in e['O']]); OK = np.concatenate([x.reshape(-1) for x in e['OK']])
    r = score_bs(T, O); t = T > 0; p = O > 0
    print(f'ПУЛ {len(e["T"])} фолдов {name:30s} {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f} | чистое небо {(t&p&OK).sum()/((t|p)&OK).sum():.4f}')
