"""SPEC-85: радарные слои Sentinel-1 RTC (Planetary Computer) в сетке окна: VV, VH после (дБ), dVV, dVH (после − до, дБ).
«До» — последний пролёт в пределах 30 суток до начала пожара (той же орбиты, что «после», если есть); «после» — пролёт, ближайший
к опорной дате, не раньше начала пожара. Режимы: обычный (опорная дата — снимок HLS окна, выход <окно>.s1.tif) и --first-pass
(опорная дата — снимок первого пролёта <окно>.fp.json, выход <окно>.fp.s1.tif). Пишет manifest_<суффикс>.json с окнами, где слой есть.
Запуск: build_s1_layers.py <каталог> <манифест> [--first-pass]."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, planetary_computer, pystac_client
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, Resampling
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.fire_dates import start_of

D, MF = Path(sys.argv[1]), sys.argv[2]; FP = '--first-pass' in sys.argv; SUF = 'fp.s1' if FP else 's1'
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)
man = json.load(open(D / MF)); UTC = timezone.utc
def ref_of(w, base):
    if FP: return datetime.fromisoformat(json.load(open(f'{base}.fp.json'))['datetime'][:19]).replace(tzinfo=UTC)
    return datetime.fromisoformat(w['item_date']).replace(tzinfo=UTC) + timedelta(hours=12)

def read_db(it, crs, tr, shape):
    out = []
    for pol in ('vv', 'vh'):
        with rasterio.open(it.assets[pol].href) as s, WarpedVRT(s, crs=crs, transform=tr, width=shape[1], height=shape[0], resampling=Resampling.average, src_nodata=0, nodata=np.nan) as v:
            a = v.read(1).astype(np.float32)
        out.append(np.where(np.isfinite(a) & (a > 0), 10 * np.log10(np.maximum(a, 1e-4)), np.nan))
    return np.stack(out)

def build(w):
    base = D / w['name']; outp = Path(f'{base}.{SUF}.tif')
    if outp.exists(): return w
    if FP and not Path(f'{base}.fp.json').exists(): return None
    with rasterio.open(f'{base}_merged.tif') as r: crs, tr, shape = r.crs, r.transform, r.shape
    bb = transform_bounds(crs, 4326, *rasterio.transform.array_bounds(*shape, tr)); t0, ref = start_of(w), ref_of(w, base)
    post = [i for i in cat.search(collections=['sentinel-1-rtc'], bbox=list(bb), datetime=f'{t0:%Y-%m-%dT%H:%M:%SZ}/{ref + timedelta(days=10):%Y-%m-%dT%H:%M:%SZ}').items()]
    if not post: return None
    po = min(post, key=lambda i: abs((datetime.fromisoformat(i.properties['datetime'][:19]).replace(tzinfo=UTC) - ref).total_seconds()))
    pre = list(cat.search(collections=['sentinel-1-rtc'], bbox=list(bb), datetime=f'{t0 - timedelta(days=30):%Y-%m-%dT%H:%M:%SZ}/{t0:%Y-%m-%dT%H:%M:%SZ}').items())
    if not pre: return None
    same = [i for i in pre if i.properties.get('sat:orbit_state') == po.properties.get('sat:orbit_state')] or pre
    pr = max(same, key=lambda i: i.properties['datetime'])
    A, B = read_db(po, crs, tr, shape), read_db(pr, crs, tr, shape); X = np.concatenate([A, A - B])
    X = np.where(np.isfinite(X).all(0)[None], X, -9999).astype(np.float32)
    with rasterio.open(outp, 'w', driver='GTiff', height=shape[0], width=shape[1], count=4, dtype='float32', crs=crs, transform=tr, nodata=-9999, compress='deflate') as o: o.write(X)
    json.dump(dict(post=po.id, pre=pr.id, start=str(t0), ref=str(ref)), open(f'{base}.{SUF}.json', 'w')); return w

def retry(w):
    import time
    for k in range(3):
        try: return build(w)
        except Exception as e: print(w['name'], 'попытка', k + 1, repr(e)[:160], flush=True); time.sleep(20)
    return None

with ThreadPoolExecutor(4) as ex: res = list(ex.map(retry, man['windows']))
keep = [r for r in res if r]
json.dump(dict(man, windows=keep, s1_from=len(man['windows'])), open(D / f"{MF[:-5]}_{SUF.replace('.', '')}.json", 'w'), ensure_ascii=False, indent=1)
print(D, MF, SUF, 'слоёв', len(keep), 'из', len(man['windows']), flush=True)
