"""Решающие правила под новым правилом маски — из кэша вероятностей exp_mask.py.
Вес ансамбля и фильтр пятен подбирались, когда под маской стоял ноль; теперь
там сеть, и оптимум мог сдвинуться. Стоит секунды."""
import sys, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_small
z = np.load('models/tune_proba.npz'); PB, PN, T, OK = z['pb'].astype(np.float32), z['pn'].astype(np.float32), z['t'], z['ok']
def pred(w, blob=0, hier=False):
    p = ((1-w)*PB + w*PN); out = p.argmax(3).astype(np.uint8)
    blind = ~OK; out[blind] = PN.argmax(3)[blind]
    if hier:  # гарь, если P(фон) < 0.5, класс — argmax среди 1..3
        pm = p.copy(); pm[blind] = PN[blind]
        burn = pm[..., 0] < 0.5; out = np.where(burn, pm[..., 1:].argmax(3) + 1, 0).astype(np.uint8)
    return [drop_small(o, blob) if blob else o for o in out]
print('вес   фильтр  иерарх  IoU_burn  mIoU_sev   кл1    кл2    кл3   взвеш.')
for w in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
    for blob in (0, 100):
        for hier in (False, True):
            if hier and blob: continue
            r = score_bs_micro(list(T), pred(w, blob, hier))
            print(f'{w:4.1f}  {blob or "—":>6}  {"да" if hier else "нет":>6}  {r["iou_burn"]:.4f}    {r["miou_sev"]:.4f}  '
                  f'{r["per_class"][1]:.3f}  {r["per_class"][2]:.3f}  {r["per_class"][3]:.3f}  {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f}')
