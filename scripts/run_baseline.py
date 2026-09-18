import sys, json; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import numpy as np
from src.comp.chips import BsDataset, split_by_fire, write_split
from src.comp.baseline import predict
from src.comp.metric import score_bs

d = BsDataset('data/comp/train/bs')
split = split_by_fire(d.meta)
write_split(split, 'data/comp/split_bs.json')
val = [c for c in split['val'] if d.has_post(c)]
print(f"val-чипов со сценой post: {len(val)} из {len(split['val'])}")
scores = []
for cid in val:
    chip = d.load(cid)
    s = score_bs(chip.mask, predict(chip))
    scores.append(s)
burn = np.nanmean([s['iou_burn'] for s in scores])
miou = np.nanmean([s['miou_sev'] for s in scores])
print(f"BASELINE USGS: IoU_burn={burn:.4f} mIoU_sev={miou:.4f} n={len(scores)}")
for c in (1,2,3):
    print(f"  класс {c}: IoU={np.nanmean([s['per_class'][c] for s in scores]):.4f}")
json.dump({'n':len(scores),'iou_burn':float(burn),'miou_sev':float(miou)}, open('data/comp/baseline_bs.json','w'), indent=2)
