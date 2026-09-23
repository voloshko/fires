"""SPEC-72: свежий лесной тест по рецепту HLS Burn Scars — контуры MTBS с 2022-01-01 (CONUS, рамка ≤ 14 км,
без пересечения со сценами HLS Burn Scars), снимок hls2-s30 через 30–150 суток после возгорания с наименьшей
долей облака/тени/нодаты в окне (< 10 %), окно 512 × 512 с центром в центроиде контура. Выход — external/hls_fresh/."""
import glob, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, planetary_computer, pystac_client
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.ops import unary_union

N, SIZE = int(__import__("os").environ.get("FRESH_N", 300)), 512   # FRESH_N — только для пробы
# SPEC-74: второй набор — FRESH_OUT, FRESH_SKIP
OUT = Path(__import__('os').environ.get('FRESH_OUT', 'external/hls_fresh')); OUT.mkdir(parents=True, exist_ok=True)
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']

# 1. Рамки сцен HLS Burn Scars (training + validation) в градусах.
foot = []
for f in sorted(glob.glob('external/hls_burn_scars/*/*_merged.tif')):
    with rasterio.open(f) as r: foot.append(gpd.GeoSeries([box(*r.bounds)], crs=r.crs).to_crs(4269).iloc[0])
foot = unary_union(foot)

# 2. Контуры MTBS.
g = gpd.read_file('external/mtbs/mtbs_perims_DD.shp'); g['d'] = pd.to_datetime(g.ig_date)
g = g[(g.d >= '2022-01-01') & ~g.event_id.str[:2].isin(['AK', 'HI', 'PR'])]
b = g.to_crs(5070).bounds; g = g[((b.maxx - b.minx) <= 14000) & ((b.maxy - b.miny) <= 14000)]
n_size = len(g); g = g[~g.intersects(foot)]
g = g.assign(h=g.event_id.map(lambda e: hashlib.sha256(f'fresh:{e}'.encode()).hexdigest())).sort_values('h')
print(f'пожаров после фильтров: {n_size} по размеру, {len(g)} без пересечения с HLS Burn Scars', flush=True)
SKIP = int(__import__('os').environ.get('FRESH_SKIP', 0)); g = g.iloc[SKIP:]; print('пропущено первых', SKIP, flush=True)
ST = __import__('os').environ.get('FRESH_STATES')   # SPEC-76: горный набор — только западные штаты
if ST: g = g[g.event_id.str[:2].isin(ST.split(','))]; print('штаты', ST, '→ пожаров', len(g), flush=True)

cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

def window_for(src, geom_proj):
    c = geom_proj.centroid; row, col = src.index(c.x, c.y)
    r0 = int(np.clip(row - SIZE // 2, 0, src.height - SIZE)); c0 = int(np.clip(col - SIZE // 2, 0, src.width - SIZE))
    return Window(c0, r0, SIZE, SIZE)

def build(rec):
    ev, geom, d = rec.event_id, rec.geometry, rec.d
    items = list(cat.search(collections=['hls2-s30'], intersects=mapping(geom.centroid),
                            datetime=f'{(d + pd.Timedelta(days=30)).date()}/{(d + pd.Timedelta(days=150)).date()}').items())
    best = None
    for it in items:
        try:
            with rasterio.open(it.assets['Fmask'].href) as src:
                gp = gpd.GeoSeries([geom], crs=4269).to_crs(src.crs).iloc[0]; w = window_for(src, gp); fm = src.read(1, window=w)
                bad = (fm == 255) | ((fm >> 1) & 1).astype(bool) | ((fm >> 3) & 1).astype(bool)
                key = (float(bad.mean()), it.datetime)
                if best is None or key < best[0]: best = (key, it, w, src.crs, src.window_transform(w), bad, gp)
        except Exception as e:
            print(ev, it.id, 'Fmask не прочитан:', e, flush=True)
    if best is None or best[0][0] >= 0.10: return None
    (frac, dt), it, w, crs, tr, bad, gp = best
    X = []
    for bnd in BANDS:
        with rasterio.open(it.assets[bnd].href) as src: a = src.read(1, window=w).astype(np.float32)
        X.append(a)
    X = np.stack(X); nod = (X == -9999).any(0) | bad; X = np.where(nod[None], -9999, X * 0.0001).astype(np.float32)
    m = rasterize([(gp, 1)], out_shape=(SIZE, SIZE), transform=tr, fill=0, dtype='int16'); m[nod] = -1
    if (m == 1).sum() == 0: return None
    name = f'fresh_{ev}_{it.id}'; prof = dict(driver='GTiff', height=SIZE, width=SIZE, crs=crs, transform=tr, compress='deflate')
    with rasterio.open(OUT / f'{name}_merged.tif', 'w', count=6, dtype='float32', nodata=-9999, **prof) as o: o.write(X)
    with rasterio.open(OUT / f'{name}.mask.tif', 'w', count=1, dtype='int16', nodata=-1, **prof) as o: o.write(m[None])
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return dict(name=name, event_id=ev, incid_name=rec.incid_name, incid_type=rec.incid_type, ig_date=str(d.date()), item=it.id,
                item_date=str(dt.date()), bad_fraction=round(frac, 4), burn_fraction=round(float((m == 1).mean()), 4),
                sha256_merged=h(OUT / f'{name}_merged.tif'), sha256_mask=h(OUT / f'{name}.mask.tif'))

done, tried = [], 0
with ThreadPoolExecutor(8) as ex:
    for start in range(0, len(g), 32):
        if len(done) >= N: break
        batch = list(g.iloc[start:start + 32].itertuples()); tried += len(batch)
        for r in ex.map(build, batch):
            if r and len(done) < N: done.append(r)
        print(f'просмотрено {tried}, годных {len(done)}', flush=True)
keep = {r['name'] for r in done}
for f in OUT.glob('fresh_*'):   # лишние окна последнего батча сверх N
    if f.name.replace('_merged.tif', '').replace('.mask.tif', '') not in keep: f.unlink()
json.dump(dict(spec='SPEC-72' if SKIP == 0 else ('SPEC-76' if ST else 'SPEC-74'), skip=SKIP, states=ST, source='MTBS mtbs_perims_DD + Planetary Computer hls2-s30', eligible=len(g), tried=tried,
               windows=done), open(OUT / 'manifest.json', 'w'), ensure_ascii=False, indent=1)
print('готово:', len(done), 'окон', flush=True)
