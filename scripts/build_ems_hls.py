"""SPEC-79: средиземноморский набор — контуры Copernicus EMS Rapid Mapping (Wildfire, 2023+) на HLS v2.
Продукт DEL/GRA с последним снимком на AOI; гарь — observedEventA с «burn»/«fire»; вне AOI — −1. Снимок hls2-s30
от −10 до +30 суток от снимка EMS, наименьшая доля облака/тени/нодаты в окне (< 10 %). Разбиение по активации:
sha256('ems:' + код) чётный — проверка, нечётный — резерв на обучение. Выход — external/ems_hls/."""
import hashlib, io, json, os, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, requests, planetary_computer, pystac_client
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import mapping
from shapely.ops import unary_union

SIZE = 512; OUT = Path('external/ems_hls'); OUT.mkdir(parents=True, exist_ok=True); ZIPS = Path('external/ems/zips'); ZIPS.mkdir(parents=True, exist_ok=True)
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']; API = 'https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api'
MED = {'Greece', 'Spain', 'Portugal', 'France', 'Italy', 'Croatia', 'Albania', 'Montenegro', 'Bosnia and Herzegovina', 'Tunisia', 'Cyprus', 'Turkey',
       'Türkiye', 'North Macedonia', 'Slovenia', 'Israel', 'Lebanon', 'Morocco', 'Algeria', 'Malta'}
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

def get_json(url, cache=None):
    """EMS ограничивает частоту: при «throttled» или пустом ответе ждём и повторяем; ответ кэшируется на диск."""
    import re, time
    if cache and Path(cache).exists(): return json.load(open(cache))
    for k in range(10):
        r = requests.get(url, timeout=60)
        try:
            d = r.json()
            if 'results' in d:
                if cache: Path(cache).parent.mkdir(parents=True, exist_ok=True); json.dump(d, open(cache, 'w'))
                time.sleep(1.5); return d
        except ValueError: d = {}
        wait = re.search(r'(\d+) seconds', r.text[:200]); time.sleep(int(wait.group(1)) + 2 if wait else 10)
    raise RuntimeError(f'EMS не ответил: {url}')

acts = [a for a in get_json(f'{API}/public-activations-info/?limit=1000', 'external/ems/activations.json')['results']
        if a['category'] == 'Wildfire' and a['eventTime'] >= '2023' and set(a['countries']) & MED and set(a['countries']) <= MED]
print('активаций', len(acts), flush=True)

def split(code): return 'проверка' if int(hashlib.sha256(f'ems:{code}'.encode()).hexdigest(), 16) % 2 == 0 else 'резерв'

def aoi_jobs(act):
    d = get_json(f"{API}/public-activations/?code={act['code']}", f"external/ems/acts/{act['code']}.json")['results'][0]; jobs = []
    for aoi in d['aois']:
        prods = [p for p in aoi['products'] if p['type'] in ('DEL', 'GRA') and p.get('downloadPath') and p.get('images')]
        if not prods: continue
        key = lambda p: (max(i['acquisitionTime'] for i in p['images']), p['type'] == 'GRA', p['version']['number'] if p.get('version') else 0)
        p = max(prods, key=key); jobs.append(dict(code=act['code'], country=','.join(act['countries']), aoi=aoi['number'], type=p['type'],
                                                  image_time=key(p)[0], url=p['downloadPath'], split=split(act['code'])))
    return jobs

def vectors(job):
    z = ZIPS / Path(job['url']).name
    if not z.exists(): return None, None   # скачивается заранее по одному (download_all)
    zf = zipfile.ZipFile(z); names = zf.namelist()
    ev = [n for n in names if 'observedEventA' in n and n.endswith('.json')]; ao = [n for n in names if 'areaOfInterestA' in n and n.endswith('.json')]
    if not ev or not ao: return None, None
    e = gpd.read_file(io.BytesIO(zf.read(ev[0]))); a = gpd.read_file(io.BytesIO(zf.read(ao[0])))
    col = lambda k: e[k].astype(str) if k in e.columns else pd.Series('', index=e.index)
    txt = (col('notation') + ' ' + col('obj_desc')).str.lower()
    e = e[txt.str.contains('burn') | txt.str.contains('fire')]
    if e.empty: return None, None
    return unary_union(e.to_crs(4326).geometry), unary_union(a.to_crs(4326).geometry)

def build(job):
    name0 = f"ems_{job['code']}_AOI{job['aoi']:02d}"
    if list(OUT.glob(f'{name0}_*.json')): return
    burn, aoi = vectors(job)
    if burn is None: return
    t = pd.Timestamp(job['image_time']).tz_localize(None)
    items = list(cat.search(collections=['hls2-s30'], intersects=mapping(burn.centroid), datetime=f'{(t - pd.Timedelta(days=10)).date()}/{(t + pd.Timedelta(days=30)).date()}').items())
    best = None
    for it in items:
        with rasterio.open(it.assets['Fmask'].href) as src:
            gb = gpd.GeoSeries([burn], crs=4326).to_crs(src.crs).iloc[0]; c = gb.centroid; row, col = src.index(c.x, c.y)
            w = Window(int(np.clip(col - SIZE // 2, 0, src.width - SIZE)), int(np.clip(row - SIZE // 2, 0, src.height - SIZE)), SIZE, SIZE)
            fm = src.read(1, window=w); bad = (fm == 255) | ((fm >> 1) & 1).astype(bool) | ((fm >> 3) & 1).astype(bool)
            key = (float(bad.mean()), abs((it.datetime.replace(tzinfo=None) - t).total_seconds()))
            if best is None or key < best[0]: best = (key, it, w, src.crs, src.window_transform(w), bad, gb)
    if best is None or best[0][0] >= 0.10: return
    (frac, _), it, w, crs, tr, bad, gb = best
    X = []
    for bnd in BANDS:
        with rasterio.open(it.assets[bnd].href) as src: X.append(src.read(1, window=w).astype(np.float32))
    X = np.stack(X); nod = (X == -9999).any(0) | bad; X = np.where(nod[None], -9999, X * 0.0001).astype(np.float32)
    ga = gpd.GeoSeries([aoi], crs=4326).to_crs(crs).iloc[0]
    inside = rasterize([(ga, 1)], out_shape=(SIZE, SIZE), transform=tr, fill=0, dtype='uint8').astype(bool)
    m = rasterize([(gb, 1)], out_shape=(SIZE, SIZE), transform=tr, fill=0, dtype='int16'); m[~inside] = -1; m[nod] = -1
    if (m == 1).sum() == 0: return
    name = f'{name0}_{it.id}'; prof = dict(driver='GTiff', height=SIZE, width=SIZE, crs=crs, transform=tr, compress='deflate')
    with rasterio.open(OUT / f'{name}_merged.tif', 'w', count=6, dtype='float32', nodata=-9999, **prof) as o: o.write(X)
    with rasterio.open(OUT / f'{name}.mask.tif', 'w', count=1, dtype='int16', nodata=-1, **prof) as o: o.write(m[None])
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    json.dump(dict(name=name, event_id=f"{job['code']}_AOI{job['aoi']:02d}", code=job['code'], country=job['country'], product=job['type'],
                   ems_image=job['image_time'], item=it.id, item_date=str(it.datetime.date()), split=job['split'], bad_fraction=round(frac, 4),
                   burn_pixels=int((m == 1).sum()), outside_aoi=round(float((~inside).mean()), 4),
                   sha256_merged=h(OUT / f'{name}_merged.tif'), sha256_mask=h(OUT / f'{name}.mask.tif')),
              open(OUT / f'{name}.json', 'w'), ensure_ascii=False)

def safe(job):
    try: build(job)
    except Exception as e: print(job['code'], job['aoi'], 'ошибка:', repr(e)[:200], flush=True)

jobs = [j for a in acts for j in aoi_jobs(a)]   # по одной: EMS ограничивает частоту
print('AOI с DEL/GRA:', len(jobs), flush=True)
def download_all(jobs):
    """EMS ограничивает частоту запросов и отвечает JSON «throttled» с кодом 200 — файл кэшируется только если это zip."""
    import re, time
    for j in jobs:
        z = ZIPS / Path(j['url']).name
        if z.exists() and zipfile.is_zipfile(z): continue
        for k in range(8):
            r = requests.get(j['url'], timeout=300)
            if r.content[:2] == b'PK': z.write_bytes(r.content); break
            wait = re.search(r'(\d+) seconds', r.text[:200]); time.sleep(int(wait.group(1)) + 2 if wait else 10)
        else: print(j['code'], j['aoi'], 'архив не скачан', flush=True)
        time.sleep(1.5)

download_all(jobs); print('архивов:', sum(zipfile.is_zipfile(ZIPS / Path(j['url']).name) for j in jobs if (ZIPS / Path(j['url']).name).exists()), 'из', len(jobs), flush=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(safe, jobs))
done = [json.load(open(f)) for f in sorted(OUT.glob('ems_*.json'))]
json.dump(dict(spec='SPEC-79', source='Copernicus EMS Rapid Mapping (© European Union) + Planetary Computer hls2-s30', activations=len(acts), aois=len(jobs), windows=done),
          open(OUT / 'manifest.json', 'w'), ensure_ascii=False, indent=1)
json.dump(dict(spec='SPEC-79', split='проверка', windows=[w for w in done if w['split'] == 'проверка']), open(OUT / 'manifest_test.json', 'w'), ensure_ascii=False, indent=1)
print('готово:', len(done), 'окон; проверка', sum(w['split'] == 'проверка' for w in done), '| резерв', sum(w['split'] == 'резерв' for w in done), flush=True)
