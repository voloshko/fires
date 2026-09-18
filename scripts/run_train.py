import sys, json, time; sys.path.insert(0,'/Users/mc/projects/fires')
import numpy as np
from src.comp.chips import BsDataset
from src.comp.model import build_training_set, train, predict, save
from src.comp.metric import score_bs

t0=time.time()
d = BsDataset('data/comp/train/bs')
split = json.load(open('data/comp/split_bs.json'))
tr = [c for c in split['train'] if d.has_post(c)]
va = [c for c in split["val"] if d.has_post(c)]
import rasterio.errors
def _ok(c):
    try: d.load(c); return True
    except rasterio.errors.RasterioIOError: return False
va = [c for c in va if _ok(c)]
print(f'train-чипов с post: {len(tr)}, val: {len(va)}')
x, y = build_training_set(d, tr)
print('обучающих пикселей', x.shape, 'классы', np.bincount(y).tolist(), f'{time.time()-t0:.0f}с')
m = train(x, y)
print(f'обучено за {time.time()-t0:.0f}с')
save(m, 'models/bs_hgb.pkl')
scores=[score_bs(d.load(c).mask, predict(m, d.load(c))) for c in va]
burn=np.nanmean([s['iou_burn'] for s in scores]); miou=np.nanmean([s['miou_sev'] for s in scores])
print(f'МОДЕЛЬ: IoU_burn={burn:.4f} mIoU_sev={miou:.4f} n={len(scores)}')
for c in (1,2,3): print(f'  класс {c}: IoU={np.nanmean([s["per_class"][c] for s in scores]):.4f}')
b=json.load(open('data/comp/baseline_bs.json'))
print(f'BASELINE был: IoU_burn={b["iou_burn"]:.4f} mIoU_sev={b["miou_sev"]:.4f}')
json.dump({'iou_burn':float(burn),'miou_sev':float(miou),'n':len(scores),'n_train_chips':len(tr)}, open('data/comp/model_bs.json','w'), indent=2)
