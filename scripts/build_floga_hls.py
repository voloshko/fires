"""SPEC-73: горы — контуры FLOGA (Пожарная служба Греции, 2020–2021) на HLS v2 (Planetary Computer, hls2-s30).
Снимок: ±10 суток от их снимка после пожара (S2_e), иначе 1–60 суток после конца пожара; наименьшая доля
облака/тени/нодаты в окне (< 10 %). Окно 512 × 512 с центром в центроиде; маска 1 — событие, −1 — другие контуры
того же года и плохие пиксели; уклон по cop-dem-glo-30 на сетке окна. Выход — external/floga_hls/."""
import hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio, planetary_computer, pystac_client
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, mapping

SIZE = 512; OUT = Path('external/floga_hls'); OUT.mkdir(parents=True, exist_ok=True)
BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

ev = []
for y in (2020, 2021):
    g = gpd.read_file(f'external/floga_ann/fb_{y}.shp')
    g = g.dissolve(by='ID', aggfunc='first').reset_index(); g['year'] = y; ev.append(g)
ev = pd.concat(ev, ignore_index=True); ev = gpd.GeoDataFrame(ev, crs=4326)
import os
N = int(os.environ.get('FLOGA_N', 0))   # только для пробы
if N: ev = ev.iloc[:N]
# ponytail: память течёт внутри одного процесса (дважды OOM), поэтому события идут пачками FLOGA_RANGE=a:b в отдельных
# процессах, каждое окно пишет свой .json, а FLOGA_ASSEMBLE=1 собирает общий манифест. Утечку не искали.
RNG = os.environ.get('FLOGA_RANGE'); ASSEMBLE = os.environ.get('FLOGA_ASSEMBLE') == '1'
print('событий', len(ev), flush=True)

def _bounds(tr):
    top, left = tr.f, tr.c; return left, top + tr.e * SIZE, left + tr.a * SIZE, top   # (minx, miny, maxx, maxy) окна

def window_for(src, geom_proj):
    c = geom_proj.centroid; row, col = src.index(c.x, c.y)
    return Window(int(np.clip(col - SIZE // 2, 0, src.width - SIZE)), int(np.clip(row - SIZE // 2, 0, src.height - SIZE)), SIZE, SIZE)

def search(geom, a, b):
    return list(cat.search(collections=['hls2-s30'], intersects=mapping(geom.centroid), datetime=f'{a.date()}/{b.date()}').items())

def slope_for(crs, tr):
    bounds = rasterio.transform.array_bounds(SIZE, SIZE, tr); lon0, lat0, lon1, lat1 = transform_bounds(crs, 4326, *bounds)
    dem = np.full((SIZE, SIZE), np.nan, np.float32)
    for it in cat.search(collections=['cop-dem-glo-30'], bbox=[lon0, lat0, lon1, lat1]).items():
        with rasterio.open(it.assets['data'].href) as s:
            part = np.full((SIZE, SIZE), np.nan, np.float32); pad = 0.01   # читаем только окно DEM вокруг снимка, не тайл целиком
            win = from_bounds(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad, s.transform).round_offsets().round_lengths()
            arr = s.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            reproject(arr, part, src_transform=s.window_transform(win), src_crs=s.crs, dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            dem = np.where(np.isnan(dem), part, dem)
    gy, gx = np.gradient(dem, 30.0); return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

def build(rec):
    geom = rec.geometry; s2e = pd.Timestamp(re.search(r'_(\d{8})T', rec.S2_e).group(1)); end = pd.Timestamp(rec.End_date)
    items = search(geom, s2e - pd.Timedelta(days=10), s2e + pd.Timedelta(days=10)); rule = 'S2_e±10'
    if not items: items = search(geom, end + pd.Timedelta(days=1), end + pd.Timedelta(days=60)); rule = 'end+1..60'
    best = None
    for it in items:
        try:
            with rasterio.open(it.assets['Fmask'].href) as src:
                gp = gpd.GeoSeries([geom], crs=4326).to_crs(src.crs).iloc[0]; w = window_for(src, gp); fm = src.read(1, window=w)
                bad = (fm == 255) | ((fm >> 1) & 1).astype(bool) | ((fm >> 3) & 1).astype(bool)
                key = (float(bad.mean()), abs((it.datetime.replace(tzinfo=None) - s2e).total_seconds()))
                if best is None or key < best[0]: best = (key, it, w, src.crs, src.window_transform(w), bad, gp)
        except Exception as e:
            print(rec.ID, it.id, 'Fmask не прочитан:', e, flush=True)
    if best is None or best[0][0] >= 0.10: return None
    (frac, _), it, w, crs, tr, bad, gp = best
    X = []
    for bnd in BANDS:
        with rasterio.open(it.assets[bnd].href) as src: X.append(src.read(1, window=w).astype(np.float32))
    X = np.stack(X); nod = (X == -9999).any(0) | bad; X = np.where(nod[None], -9999, X * 0.0001).astype(np.float32)
    m = rasterize([(gp, 1)], out_shape=(SIZE, SIZE), transform=tr, fill=0, dtype='int16')
    others = ev[(ev.year == rec.year) & (ev.ID != rec.ID)].to_crs(crs)
    wbox = box(*_bounds(tr))
    others = [g for g in others.geometry if g.intersects(wbox)]   # рамка окна вместо буфера на 15 км: буфер сложного контура съедал 12 ГБ
    if others:
        o = rasterize([(g, 1) for g in others], out_shape=(SIZE, SIZE), transform=tr, fill=0, dtype='uint8').astype(bool); m[o & (m == 0)] = -1
    m[nod] = -1
    if (m == 1).sum() == 0: return None
    sl = slope_for(crs, tr)
    name = f'floga_{rec.year}_{rec.ID}_{it.id}'; prof = dict(driver='GTiff', height=SIZE, width=SIZE, crs=crs, transform=tr, compress='deflate')
    with rasterio.open(OUT / f'{name}_merged.tif', 'w', count=6, dtype='float32', nodata=-9999, **prof) as o: o.write(X)
    with rasterio.open(OUT / f'{name}.mask.tif', 'w', count=1, dtype='int16', nodata=-1, **prof) as o: o.write(m[None])
    with rasterio.open(OUT / f'{name}.slope.tif', 'w', count=1, dtype='float32', nodata=np.nan, **prof) as o: o.write(sl[None])
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return dict(name=name, id=str(rec.ID), year=int(rec.year), item=it.id, item_date=str(it.datetime.date()), s2_e=str(s2e.date()), rule=rule,
                bad_fraction=round(frac, 4), burn_pixels=int((m == 1).sum()), slope_median_burn=float(np.nanmedian(sl[m == 1])),
                sha256_merged=h(OUT / f'{name}_merged.tif'), sha256_mask=h(OUT / f'{name}.mask.tif'))

ev = ev.rename(columns={'End date': 'End_date', 'Start date': 'Start_date'})
if ASSEMBLE:
    done = [json.load(open(f)) for f in sorted(OUT.glob('floga_*.json'))]
    json.dump(dict(spec='SPEC-73', source='FLOGA-annotations polygons/v2 (MIT) + Planetary Computer hls2-s30, cop-dem-glo-30', events=len(ev), windows=done),
              open(OUT / 'manifest.json', 'w'), ensure_ascii=False, indent=1)
    print('готово:', len(done), 'окон из', len(ev), 'событий', flush=True); raise SystemExit
a, b = (int(x) for x in RNG.split(':')) if RNG else (0, len(ev))
part = ev.iloc[a:b]
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(build, part.itertuples()):
        if r: json.dump(r, open(OUT / f"{r['name']}.json", 'w'), ensure_ascii=False)
print(f'события {a}:{b} готовы', flush=True)
