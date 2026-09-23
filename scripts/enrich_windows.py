"""SPEC-76/77: к каждому окну каталога (manifest.json с полем item) дописываются слои того же снимка HLS — Fmask,
SZA, SAA (× 0.01°) — и рельеф Copernicus DEM GLO-30, пиксель в пиксель с окном. С флагом --mount пишет
manifest_mount.json: окна с медианным уклоном внутри контура ≥ 10° (отбор до замера моделей)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, rasterio, planetary_computer, pystac_client
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.terrain import slope_deg

D = Path(sys.argv[1]); MOUNT = '--mount' in sys.argv
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)
man = json.load(open(D / 'manifest.json'))


def dem_for(crs, tr, shape):
    b = rasterio.transform.array_bounds(*shape, tr); lon0, lat0, lon1, lat1 = transform_bounds(crs, 4326, *b); dem = np.full(shape, np.nan, np.float32)
    for it in cat.search(collections=['cop-dem-glo-30'], bbox=[lon0, lat0, lon1, lat1]).items():
        with rasterio.open(it.assets['data'].href) as s:
            w = from_bounds(lon0 - 0.01, lat0 - 0.01, lon1 + 0.01, lat1 + 0.01, s.transform).round_offsets().round_lengths()
            arr = s.read(1, window=w, boundless=True, fill_value=np.nan).astype(np.float32); part = np.full(shape, np.nan, np.float32)
            reproject(arr, part, src_transform=s.window_transform(w), src_crs=s.crs, dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            dem = np.where(np.isnan(dem), part, dem)
    return dem


def enrich(w):
    base = D / w['name']
    if all(Path(f'{base}.{k}.tif').exists() for k in ('fmask', 'sza', 'saa', 'dem')): return
    with rasterio.open(f'{base}_merged.tif') as r: crs, tr, shape, bounds, prof = r.crs, r.transform, r.shape, r.bounds, r.profile
    coll = 'hls2-s30' if '.S30.' in w['item'] else 'hls2-l30'; it = cat.get_collection(coll).get_item(w['item'])
    prof = dict(driver='GTiff', height=shape[0], width=shape[1], crs=crs, transform=tr, compress='deflate', count=1)
    for k, a in (('fmask', 'Fmask'), ('sza', 'SZA'), ('saa', 'SAA')):
        with rasterio.open(it.assets[a].href) as s:
            win = from_bounds(*bounds, transform=s.transform).round_offsets().round_lengths(); arr = s.read(1, window=win)
        assert arr.shape == shape, (w['name'], k, arr.shape)
        with rasterio.open(f'{base}.{k}.tif', 'w', dtype=arr.dtype, **prof) as o: o.write(arr[None])
    dem = dem_for(crs, tr, shape)
    with rasterio.open(f'{base}.dem.tif', 'w', dtype='float32', nodata=np.nan, **prof) as o: o.write(dem[None])


with ThreadPoolExecutor(4) as ex: list(ex.map(enrich, man['windows']))
print('слои дописаны:', len(man['windows']), 'окон', flush=True)
if MOUNT:
    keep = []
    for w in man['windows']:
        m = rasterio.open(D / f"{w['name']}.mask.tif").read()[0]; sl = slope_deg(rasterio.open(D / f"{w['name']}.dem.tif").read()[0].astype(np.float64))
        med = float(np.nanmedian(sl[m == 1])); w = dict(w, slope_median_burn=round(med, 2))
        if med >= 10: keep.append(w)
    json.dump(dict(man, windows=keep, mount_rule='медианный уклон внутри контура ≥ 10°, выбрано до замера моделей', mount_from=len(man['windows'])),
              open(D / 'manifest_mount.json', 'w'), ensure_ascii=False, indent=1)
    print('горных окон:', len(keep), 'из', len(man['windows']), flush=True)
