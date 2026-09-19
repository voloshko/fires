"""SPEC-29: происхождение разметки BS — сопоставление с MCD64A1 (Planetary Computer,
анонимно, только ОБУЧАЮЩИЕ чипы). Для каждого чипа: дата выгорания MCD64A1 в
сетке чипа; IoU истины с «выгорело в окне [date_pre, date_post]»; доля нашей
ложной гари, лежащей на MCD64-гари с датой ВНЕ окна (соседние пожары)."""
import sys, json, hashlib, numpy as np, pandas as pd, datetime as dt
from pathlib import Path
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import pystac_client, planetary_computer as pc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.postproc import drop_far
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)
d = BsDataset('data/comp/train/bs'); meta = pd.read_csv('data/comp/train/bs/meta.csv').set_index('chip_id')
s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35])
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
pn = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in ('d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1')], 0)
P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
chips = {c: d.load(c) for c in tune}
pred = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8)
for i, c in enumerate(tune): pred[i][chips[c].label_zero()] = 0; pred[i] = drop_far(pred[i])
def doy(date): t = dt.date.fromisoformat(date); return t.timetuple().tm_yday, t.year
def mcd64_burn_date(chip, year, months):
    """Дата выгорания (день года) MCD64A1 в сетке чипа; 0 — не горело; объединение по месяцам и тайлам."""
    with rasterio.open(f'data/comp/train/bs/sentinel2_post/{chip.chip_id}_Sentinel-2_post.tif') as ref:
        crs, transform, shape, bounds = ref.crs, ref.transform, (ref.height, ref.width), ref.bounds
    from rasterio.warp import transform_bounds
    bbox = transform_bounds(crs, 'EPSG:4326', *bounds)
    out = np.zeros(shape, np.int16)
    for m in months:
        items = list(cat.search(collections=['modis-64A1-061'], bbox=bbox, datetime=f'{year}-{m:02d}-01/{year}-{m:02d}-28').items())
        for it in items:
            href = it.assets['Burn_Date'].href
            with rasterio.open(href) as src, WarpedVRT(src, crs=crs, transform=transform, width=shape[1], height=shape[0], resampling=Resampling.nearest) as vrt:
                arr = vrt.read(1)
            good = arr > 0; out[good] = arr[good]
    return out
print('чип            окно doy      истина  fp   |  IoU(истина, MCD64 в окне)  IoU(истина, MCD64 всё)  доля fp на MCD64 вне окна  доля истины без MCD64')
summary = []
for i, c in enumerate(tune):
    ch = chips[c]; t = T[i] > 0; fp = (pred[i] > 0) & ~t
    d0, y0 = doy(meta.loc[c, 'date_pre']); d1, y1 = doy(meta.loc[c, 'date_post'])
    months = sorted({dt.date.fromisoformat(meta.loc[c, 'date_pre']).month, dt.date.fromisoformat(meta.loc[c, 'date_post']).month, max(1, dt.date.fromisoformat(meta.loc[c, 'date_pre']).month - 1)})
    try: bd = mcd64_burn_date(ch, y0, months)
    except Exception as e: print(f'{c}  ошибка {type(e).__name__}: {str(e)[:80]}'); continue
    inwin = (bd >= d0 - 3) & (bd <= d1 + 3); anyb = bd > 0; outwin = anyb & ~inwin
    iou_in = (t & inwin).sum() / max(1, (t | inwin).sum()); iou_any = (t & anyb).sum() / max(1, (t | anyb).sum())
    fp_out = (fp & outwin).sum() / max(1, fp.sum()); t_no = (t & ~anyb).sum() / max(1, t.sum())
    summary.append((c, iou_in, iou_any, fp_out, t_no, int(fp.sum())))
    print(f'{c}  {d0:3d}-{d1:3d} {y0}  {int(t.sum()):6d} {int(fp.sum()):6d} |  {iou_in:.3f}                    {iou_any:.3f}                {fp_out:.2f}                      {t_no:.2f}', flush=True)
S = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in summary], float)
print(f'\nсреднее по чипам: IoU в окне {S[:,0].mean():.3f}, IoU всё {S[:,1].mean():.3f}; доля fp на MCD64 вне окна, взвешенно по fp: {(S[:,2]*S[:,4]).sum()/S[:,4].sum():.2f}; истины без MCD64: {S[:,3].mean():.2f}')
