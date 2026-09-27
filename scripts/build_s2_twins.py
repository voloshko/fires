"""SPEC-83: двойники окон на Sentinel-2 L2A (Earth Search, sentinel-2-c1-l2a) — тот же тайл MGRS и та же дата, что снимок HLS окна.
Полосы blue green red nir08 swir16 swir22 → сетка окна 30 м (усреднение), отражение = DN × 0.0001 − 0.1; SCL — ближайший сосед.
Пишет <окно>.s2.tif (6 полос, −9999 — нодата) и <окно>.s2scl.tif; манифест двойников — manifest_s2.json."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, rasterio, pystac_client
from rasterio.warp import reproject, Resampling

ES = pystac_client.Client.open('https://earth-search.aws.element84.com/v1')
BANDS = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']


def twin(d, w):
    base = d / w['name']
    if Path(f'{base}.s2.tif').exists(): return dict(w, s2=json.load(open(f'{base}.s2.json')))
    _, _, tile, stamp = w['item'].split('.')[:4]; day = datetime.strptime(stamp[:7], '%Y%j')
    its = list(ES.search(collections=['sentinel-2-c1-l2a'], query={'grid:code': {'eq': f'MGRS-{tile[1:]}'}},
                         datetime=f'{day:%Y-%m-%d}T00:00:00Z/{day + timedelta(days=1):%Y-%m-%d}T00:00:00Z').items())
    if not its: return None
    it = min(its, key=lambda i: i.properties.get('eo:cloud_cover', 100))
    with rasterio.open(f'{base}_merged.tif') as r: crs, tr, shape = r.crs, r.transform, r.shape
    X = np.zeros((6,) + shape, np.float32); ok = np.ones(shape, bool)
    for k, b in enumerate(BANDS):
        with rasterio.open(it.assets[b].href) as s:
            arr = np.full(shape, np.nan, np.float32)
            reproject(rasterio.band(s, 1), arr, dst_transform=tr, dst_crs=crs, resampling=Resampling.average, src_nodata=0, dst_nodata=np.nan)
        ok &= np.isfinite(arr); X[k] = np.nan_to_num(arr) * 0.0001 - 0.1
    scl = np.zeros(shape, np.uint8)
    with rasterio.open(it.assets['scl'].href) as s:
        reproject(rasterio.band(s, 1), scl, dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    X[:, ~ok] = -9999
    prof = dict(driver='GTiff', height=shape[0], width=shape[1], crs=crs, transform=tr, compress='deflate')
    with rasterio.open(f'{base}.s2.tif', 'w', count=6, dtype='float32', nodata=-9999, **prof) as o: o.write(X)
    with rasterio.open(f'{base}.s2scl.tif', 'w', count=1, dtype='uint8', nodata=0, **prof) as o: o.write(scl[None])
    meta = dict(id=it.id, datetime=it.properties['datetime'], created=it.properties.get('created'), product_uri=it.properties.get('s2:product_uri'))
    json.dump(meta, open(f'{base}.s2.json', 'w')); return dict(w, s2=meta)


def retry(args):
    import time
    for k in range(3):
        try: return twin(*args)
        except Exception as e: print(args[1]['name'], 'попытка', k + 1, repr(e)[:160], flush=True); time.sleep(20)
    return None


for d, mf in (('external/floga_hls', 'manifest_test.json'), ('external/ems_hls', 'manifest_test.json')):
    d = Path(d); man = json.load(open(d / mf))
    with ThreadPoolExecutor(4) as ex: res = list(ex.map(retry, [(d, w) for w in man['windows']]))
    keep = [r for r in res if r]
    json.dump(dict(man, windows=keep, twins_from=len(man['windows'])), open(d / 'manifest_s2.json', 'w'), ensure_ascii=False, indent=1)
    print(d, 'двойников', len(keep), 'из', len(man['windows']), flush=True)
