"""Разбор остаточных ошибок рецепта v19 на 5 групповых фолдах (144 чипа, кэши).

Куда уходит IoU: кромка или пятна, под маской или на чистом небе, какие классы
SCL под маской, какие чипы теряют пожар целиком.
"""
import sys, json, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from scipy.ndimage import distance_transform_edt, label
from src.comp.chips import BsDataset
from src.comp.postproc import drop_far
HYP = Path.home() / 'fires-hypotheses'; OWN = Path('research')
d = BsDataset('data/comp/train/bs')
conf = np.zeros((4, 4), np.int64); rows = []
cat = {k: np.zeros(3, np.int64) for k in ('чистое небо', 'под маской')}   # TP FP FN по гари
edge = {k: np.zeros(2, np.int64) for k in ('кромка ≤2 пикс', '3–10 пикс', '>10 пикс (пятна)')}  # FP FN
scl_fn = {}; scl_fp = {}
for f in range(5):
    base = {k: HYP / f'research/bs-confirm-{k}-f{f}-v1' for k in ('boost', 'optical', 'siam')}
    ids = json.load(open(base['siam'] / 'data_manifest.json'))['evaluation']; chips = [d.load(c) for c in ids]
    PB, PO, PS1 = (np.load(base[k] / 'probabilities.npy').astype(np.float32) for k in ('boost', 'optical', 'siam'))
    PS = 0.5 * (PS1 + np.load(OWN / f'bs-confirm-siam-f{f}-s2-v1/probabilities.npy').astype(np.float32))
    pn = 0.5 * PO + 0.5 * PS; P = 0.6 * pn + 0.4 * PB
    for i, c in enumerate(chips):
        OK = c.valid(); burn = P[i].argmax(2) > 0; burn[~OK] = (pn[i].argmax(2) > 0)[~OK]
        out = np.where(burn, P[i][..., 1:].argmax(2) + 1, 0).astype(np.uint8); out[c.label_zero()] = 0
        out = drop_far(out, anchor=(PO[i].argmax(2) > 0) & (PS[i].argmax(2) > 0))
        T = c.mask; t = T > 0; p = out > 0
        conf += np.bincount((T.astype(int) * 4 + out).reshape(-1), minlength=16).reshape(4, 4)
        for k, m in (('чистое небо', OK), ('под маской', ~OK)):
            cat[k] += [(t & p & m).sum(), (~t & p & m).sum(), (t & ~p & m).sum()]
        dist = np.where(t, distance_transform_edt(t), distance_transform_edt(~t))  # расстояние до кромки истины
        for k, lo, hi in (('кромка ≤2 пикс', 0, 2), ('3–10 пикс', 2, 10), ('>10 пикс (пятна)', 10, 1e9)):
            m = (dist > lo) & (dist <= hi); edge[k] += [(~t & p & m).sum(), (t & ~p & m).sum()]
        for s in np.unique(c.post[9][~OK]):
            m = (~OK) & (c.post[9] == s); scl_fn[s] = scl_fn.get(s, 0) + (t & ~p & m).sum(); scl_fp[s] = scl_fp.get(s, 0) + (~t & p & m).sum()
        inter = (t & p).sum(); union = (t | p).sum()
        rows.append((c.chip_id, f, int(t.sum()), int(p.sum()), inter / max(union, 1), (~OK & t).mean() if t.any() else 0))
tp, fp, fn = (cat['чистое небо'] + cat['под маской'])
print(f'IoU гари всего {tp/(tp+fp+fn):.4f}; FP {fp} ({fp/(fp+fn):.0%} ошибок), FN {fn}')
for k, v in cat.items(): print(f'  {k:12s} TP {v[0]:8d} FP {v[1]:7d} FN {v[2]:7d}  IoU {v[0]/max(v.sum(),1):.4f}  доля всех ошибок {(v[1]+v[2])/(fp+fn):.0%}')
print('по расстоянию до кромки истины (FP / FN):'); 
for k, v in edge.items(): print(f'  {k:18s} FP {v[0]:7d} FN {v[1]:7d}  доля ошибок {(v[0]+v[1])/(fp+fn):.0%}')
print('под маской по классу SCL post (FN / FP):', {int(k): (int(scl_fn[k]), int(scl_fp[k])) for k in sorted(scl_fn) if scl_fn[k] + scl_fp[k] > 2000})
iou_cls = [conf[k, k] / (conf[k].sum() + conf[:, k].sum() - conf[k, k]) for k in range(4)]
print('IoU классов:', np.round(iou_cls, 3), ' mIoU', round(float(np.mean(iou_cls)), 4))
print('матрица (истина строки → предсказание), тыс.:'); print((conf // 1000))
sev = conf[1:, 1:]; print(f'внутри найденной гари верная степень {np.trace(sev)/sev.sum():.3f}; класс 1 уходит в 2: {conf[1,2]/conf[1].sum():.2f}, в фон: {conf[1,0]/conf[1].sum():.2f}')
rows.sort(key=lambda r: r[4]); bad = [r for r in rows if r[4] < 0.3]
print(f'чипов с IoU<0.3: {len(bad)} из {len(rows)}; их FN+FP пикс: {sum(abs(r[2]-r[3]) for r in bad)}')
for r in bad: print(f'  {r[0]} ф{r[1]} истина {r[2]:6d} пред {r[3]:6d} IoU {r[4]:.2f} доля истины под маской {r[5]:.2f}')
print('медиана IoU по чипам', round(float(np.median([r[4] for r in rows])), 3), '; квартили', np.round(np.percentile([r[4] for r in rows], [25, 75]), 3))
