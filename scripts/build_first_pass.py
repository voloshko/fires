"""SPEC-85: снимок «первого пролёта» — первый Sentinel-2 L2A (Earth Search) после начала пожара с облачностью сцены ≤ 80 %,
облака и дым допускаются. Сетка окна, те же полосы и масштаб, что у двойников SPEC-83. Пишет <окно>.fp.s2.tif, .fp.s2scl.tif, .fp.json.
Запуск: build_first_pass.py <каталог> <манифест>."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import numpy as np, rasterio, pystac_client
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datetime import datetime, timezone
from src.comp.fire_dates import start_of
UTC = timezone.utc
D, MF = Path(sys.argv[1]), sys.argv[2]
ES = pystac_client.Client.open('https://earth-search.aws.element84.com/v1'); BANDS = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']

def build(w):
    base = D / w['name']
    if Path(f'{base}.fp.json').exists(): return w
    with rasterio.open(f'{base}_merged.tif') as r: crs, tr, shape = r.crs, r.transform, r.shape
    bb = transform_bounds(crs, 4326, *rasterio.transform.array_bounds(*shape, tr)); t0 = start_of(w)
    c = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
    its = sorted(ES.search(collections=['sentinel-2-c1-l2a'], intersects={'type': 'Point', 'coordinates': c},
                           datetime=f'{t0:%Y-%m-%dT%H:%M:%SZ}/{t0 + timedelta(days=15):%Y-%m-%dT%H:%M:%SZ}', max_items=60).items(), key=lambda i: i.properties['datetime'])
    it = next((i for i in its if i.properties.get('eo:cloud_cover', 100) <= 80), None)
    if it is None: return None
    X = np.zeros((6,) + shape, np.float32); ok = np.ones(shape, bool)
    for k, b in enumerate(BANDS):
        with rasterio.open(it.assets[b].href) as s, WarpedVRT(s, crs=crs, transform=tr, width=shape[1], height=shape[0], resampling=Resampling.average, src_nodata=0, nodata=np.nan) as v:
            a = v.read(1).astype(np.float32)
        ok &= np.isfinite(a); X[k] = np.nan_to_num(a) * 0.0001 - 0.1
    with rasterio.open(it.assets['scl'].href) as s, WarpedVRT(s, crs=crs, transform=tr, width=shape[1], height=shape[0], resampling=Resampling.nearest, src_nodata=0, nodata=0) as v:
        scl = v.read(1).astype(np.uint8)
    X[:, ~ok] = -9999; prof = dict(driver='GTiff', height=shape[0], width=shape[1], crs=crs, transform=tr, compress='deflate')
    with rasterio.open(f'{base}.fp.s2.tif', 'w', count=6, dtype='float32', nodata=-9999, **prof) as o: o.write(X)
    with rasterio.open(f'{base}.fp.s2scl.tif', 'w', count=1, dtype='uint8', nodata=0, **prof) as o: o.write(scl[None])
    bad = np.isin(scl, [0, 1, 3, 8, 9, 10]).mean()
    json.dump(dict(id=it.id, datetime=it.properties['datetime'], start=str(t0), hours_after_start=round((datetime.fromisoformat(it.properties['datetime'][:19]).replace(tzinfo=UTC) - t0).total_seconds() / 3600, 1),
                   scene_cloud=it.properties.get('eo:cloud_cover'), window_bad=round(float(bad), 4)), open(f'{base}.fp.json', 'w'))
    return w

def retry(w):
    import time
    for k in range(3):
        try: return build(w)
        except Exception as e: print(w['name'], 'попытка', k + 1, repr(e)[:160], flush=True); time.sleep(20)
    return None

man = json.load(open(D / MF))
with ThreadPoolExecutor(4) as ex: res = list(ex.map(retry, man['windows']))
keep = [r for r in res if r]
json.dump(dict(man, windows=keep, fp_from=len(man['windows'])), open(D / f"{MF[:-5]}_fp.json", 'w'), ensure_ascii=False, indent=1)
print(D, 'первый пролёт', len(keep), 'из', len(man['windows']), flush=True)
