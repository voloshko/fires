"""SPEC-86: растр тяжести MTBS (ImageServer USFS_EDW_MTBS_CONUS, год пожара) в сетке окна → <окно>.sev.tif (классы MTBS 0–6).
Запуск: build_mtbs_sev.py <каталог> [<каталог> …]; пишет manifest_sev.json с окнами, где растр получен (годы ≤ 2024)."""
import io, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, rasterio, requests
URL = 'https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_MTBS_CONUS/ImageServer/exportImage'

def sev(d, w):
    p = d / f"{w['name']}.sev.tif"
    if p.exists(): return w
    y = int(w['ig_date'][:4])
    if y > 2024: return None
    with rasterio.open(d / f"{w['name']}_merged.tif") as r: crs, tr, (h, wd), b = r.crs, r.transform, r.shape, r.bounds
    q = dict(bbox=f'{b.left},{b.bottom},{b.right},{b.top}', bboxSR=crs.to_epsg(), imageSR=crs.to_epsg(), size=f'{wd},{h}', format='tiff', pixelType='U8',
             interpolation='RSP_NearestNeighbor', mosaicRule=json.dumps({'mosaicMethod': 'esriMosaicNone', 'where': f"name='mtbs_CONUS_{y}'"}), f='image')
    for k in range(5):
        try:
            r = requests.get(URL, params=q, timeout=120)
            if r.content[:2] in (b'II', b'MM'): break
        except Exception: pass
        time.sleep(5 * (k + 1))
    else: print(w['name'], 'MTBS не ответил', flush=True); return None
    with rasterio.open(io.BytesIO(r.content)) as s: a = s.read(1)
    assert a.shape == (h, wd), (w['name'], a.shape)
    m = rasterio.open(d / f"{w['name']}.mask.tif").read(1)
    if not np.isin(a[m == 1], [1, 2, 3, 4, 5]).any(): return None   # MTBS тяжесть по пожару не выпустил
    with rasterio.open(p, 'w', driver='GTiff', height=h, width=wd, count=1, dtype='uint8', crs=crs, transform=tr, compress='deflate') as o: o.write(a[None])
    return w

for d in sys.argv[1:]:
    d = Path(d); man = json.load(open(d / 'manifest.json'))
    with ThreadPoolExecutor(4) as ex: keep = [x for x in ex.map(lambda w: sev(d, w), man['windows']) if x]
    json.dump(dict(man, windows=keep, sev_from=len(man['windows'])), open(d / 'manifest_sev.json', 'w'), ensure_ascii=False, indent=1)
    print(d, 'растров тяжести', len(keep), 'из', len(man['windows']), flush=True)
