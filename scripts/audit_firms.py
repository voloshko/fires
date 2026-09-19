"""SPEC-29, FIRMS-часть: детекции VIIRS (SP) в окне «до–после» на ОБУЧАЮЩИХ
чипах. Ключ — только из окружения FIRMS_MAP_KEY. Вопросы: какая доля истинной
гари лежит дальше 1 км от любой детекции окна; какая доля нашей ложной гари —
рядом с детекциями (то есть горело, но не размечено); есть ли чипы с истиной без
детекций. Тестовые чипы не трогаются."""
import os, sys, io, json, hashlib, time, datetime as dt, numpy as np, pandas as pd, requests
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds, transform
from scipy.ndimage import distance_transform_edt as edt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.postproc import drop_far
KEY = os.environ['FIRMS_MAP_KEY']
d = BsDataset('data/comp/train/bs'); meta = pd.read_csv('data/comp/train/bs/meta.csv').set_index('chip_id')
s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
pn = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0)
P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
def detections(bbox, start, end):
    rows = []; day = start
    while day <= end:
        span = min(5, (end - day).days + 1)
        for src in ('VIIRS_SNPP_SP', 'VIIRS_NOAA20_SP'):
            url = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{src}/{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}/{span}/{day.isoformat()}'
            for attempt in range(3):
                r = requests.get(url, timeout=60)
                if r.ok and r.text.strip() and not r.text.startswith('Invalid'): break
                time.sleep(2)
            if r.ok and r.text.startswith('latitude'): rows.append(pd.read_csv(io.StringIO(r.text)))
        day += dt.timedelta(days=span)
    return pd.concat(rows) if rows else pd.DataFrame(columns=['latitude', 'longitude'])
print('чип           окно            детекций  гарь  доля гари >1 км от детекций  fp   доля fp <1 км от детекций  детекций вне гари (>1 км)')
agg = dict(truth=0, truth_far=0, fp=0, fp_near=0, det=0, det_far=0)
for i, c in enumerate(tune):
    ch = d.load(c); t = T[i] > 0
    pred = np.where(burn[i], P[i][..., 1:].argmax(2) + 1, 0).astype(np.uint8); pred[ch.label_zero()] = 0; pred = drop_far(pred); fp = (pred > 0) & ~t
    with rasterio.open(f'data/comp/train/bs/sentinel2_post/{c}_Sentinel-2_post.tif') as ref: crs, tr, bounds = ref.crs, ref.transform, ref.bounds
    bbox = transform_bounds(crs, 'EPSG:4326', *bounds); pad = 0.02
    start, end = dt.date.fromisoformat(meta.loc[c, 'date_pre']), dt.date.fromisoformat(meta.loc[c, 'date_post'])
    det = detections((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), start, end)
    grid = np.zeros(t.shape, bool)
    if len(det):
        xs, ys = transform('EPSG:4326', crs, det.longitude.values, det.latitude.values)
        cols, rows = ~tr * (np.array(xs), np.array(ys)); cols, rows = np.round(cols).astype(int), np.round(rows).astype(int)
        ok = (cols >= 0) & (cols < 512) & (rows >= 0) & (rows < 512); grid[rows[ok], cols[ok]] = True
    dist = edt(~grid) * 20 if grid.any() else np.full(t.shape, 1e9)   # метры
    far_t = (t & (dist > 1000)).sum(); near_fp = (fp & (dist <= 1000)).sum()
    dist_t = edt(~t) * 20 if t.any() else np.full(t.shape, 1e9); det_far = int((grid & (dist_t > 1000)).sum())
    agg['truth'] += t.sum(); agg['truth_far'] += far_t; agg['fp'] += fp.sum(); agg['fp_near'] += near_fp; agg['det'] += grid.sum(); agg['det_far'] += det_far
    print(f'{c}  {start}–{end}  {len(det):5d}   {int(t.sum()):6d}  {far_t/max(1,t.sum()):.2f}                        {int(fp.sum()):6d}  {near_fp/max(1,fp.sum()):.2f}                     {det_far}', flush=True)
print(f"\nИТОГО: истинной гари дальше 1 км от детекций окна: {agg['truth_far']/agg['truth']:.1%}; нашей ложной гари ближе 1 км к детекциям: {agg['fp_near']/agg['fp']:.1%}; детекций дальше 1 км от истины: {agg['det_far']}/{agg['det']}")
