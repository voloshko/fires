"""SPEC-80: карта реального кипрского пожара рецептом для гор (C1-F + маска «Fmask и NDWI», порог 0.5), рядом — dNBR > 0.1.
Пожар — крупнейшее событие 2025 года на Кипре по детекциям VIIRS (архив FIRMS). Ключ FIRMS — только из окружения.
Выход — research/cyprus-fire-v1/ (GeoTIFF, обзорная картинка, summary.json с площадями и masked_fraction)."""
import json, os, sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, rasterio, planetary_computer, pystac_client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import firms
from src.events import build_events, EventConfig
from src.burn import masked_fraction
from src.comp.hls_eval import c1_probs
from src.comp.terrain import water_ndwi

OUT = Path('research/cyprus-fire-v1'); OUT.mkdir(parents=True, exist_ok=True); S = (1, 2, 3, 4, 5)
KEY = os.environ.get('FIRMS_MAP_KEY', '')
if not KEY: sys.exit('FIRMS_MAP_KEY не задан в окружении')
CYPRUS = firms.Region(name='Cyprus', bbox=(32.2, 34.5, 34.7, 35.8)); SRC = ['VIIRS_SNPP_SP', 'VIIRS_NOAA20_SP']

# 1. Детекции июнь–сентябрь 2025, окнами по 5 суток.
dets, fails = [], {}
d = datetime(2025, 6, 5)
while d <= datetime(2025, 9, 30):
    r = firms.fetch(CYPRUS, SRC, 5, KEY, date=d.strftime('%Y-%m-%d')); dets += r.detections; fails.update(r.failures); d += timedelta(days=5)
dets = list({(x.sensor, x.latitude, x.longitude, x.acquired_at): x for x in dets}.values())
print('детекций', len(dets), '| сбои источников', {k: v[:80] for k, v in fails.items()}, flush=True)
ev = max(build_events(dets, EventConfig()), key=lambda e: len(e.detections)); D = ev.detections
t0, t1 = min(x.acquired_at for x in D), max(x.acquired_at for x in D)
lat = [x.latitude for x in D]; lon = [x.longitude for x in D]
aoi = (min(lon) - 0.022, min(lat) - 0.018, max(lon) + 0.022, max(lat) + 0.018)   # ≈ 2 км
print(f'событие: {len(D)} детекций, {t0:%Y-%m-%d} — {t1:%Y-%m-%d}, AOI {np.round(aoi, 3).tolist()}', flush=True)

# 2. Снимки HLS до и после, один тайл MGRS.
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)
BANDS = {'hls2-s30': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'], 'hls2-l30': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07']}
def bad_of(it):
    with rasterio.open(it.assets['Fmask'].href) as s:
        w = from_bounds(*transform_bounds(4326, s.crs, *aoi), transform=s.transform).round_offsets().round_lengths(); fm = s.read(1, window=w, boundless=True, fill_value=255)
    return float(((fm == 255) | ((fm >> 1) & 1).astype(bool) | ((fm >> 3) & 1).astype(bool)).mean())
def pick(a, b, tile=None, last=False):
    its = [it for c in BANDS for it in cat.search(collections=[c], bbox=list(aoi), datetime=f'{a:%Y-%m-%d}/{b:%Y-%m-%d}').items() if tile is None or it.id.split('.')[2] == tile]
    its = sorted(its, key=lambda it: it.datetime, reverse=last)
    for it in its:
        if bad_of(it) < 0.10: return it
    return None
post = pick(t1 + timedelta(days=3), t1 + timedelta(days=60)); assert post, 'нет чистого снимка после'
pre = pick(t0 - timedelta(days=60), t0 - timedelta(days=1), tile=post.id.split('.')[2], last=True)
print('после', post.id, '| до', pre.id if pre else 'нет', flush=True)

def read(it):
    with rasterio.open(it.assets['Fmask'].href) as s:
        w = from_bounds(*transform_bounds(4326, s.crs, *aoi), transform=s.transform).round_offsets().round_lengths()
        fm = s.read(1, window=w, boundless=True, fill_value=255); tr, crs = s.window_transform(w), s.crs
    X = []
    for b in BANDS[it.collection_id]:
        with rasterio.open(it.assets[b].href) as s: X.append(s.read(1, window=w, boundless=True, fill_value=-9999).astype(np.float32))
    X = np.stack(X); bad = (X == -9999).any(0) | (fm == 255) | ((fm >> 1) & 1).astype(bool) | ((fm >> 3) & 1).astype(bool)
    return np.where(bad[None], 0, X * 0.0001).astype(np.float32), fm, ~bad, tr, crs
Xp, fmp, vp, tr, crs = read(post); H, W = vp.shape

# 3. Модель: окна 512 × 512 по AOI, ансамбль C1-F, маска «Fmask и NDWI».
ph, pw = -H % 512, -W % 512; Xpad = np.pad(Xp, ((0, 0), (0, ph), (0, pw)), mode='reflect')
tiles = [Xpad[:, i:i + 512, j:j + 512] for i in range(0, H + ph, 512) for j in range(0, W + pw, 512)]
P = c1_probs(np.stack(tiles), [Path(f'research/hls-c1f-final-s{s}') for s in S], OUT / 'c1f_tiles.npy'); full = np.zeros(Xpad.shape[1:], np.float32); k = 0
for i in range(0, H + ph, 512):
    for j in range(0, W + pw, 512): full[i:i + 512, j:j + 512] = P[k]; k += 1
prob = full[:H, :W]; wat = water_ndwi(fmp, Xp[1], Xp[3]); burn = (prob >= 0.5) & vp & ~wat

# 4. dNBR по паре (NBR = B8A/B05 и B12/B07).
if pre is not None:
    Xb, _, vb, trb, crsb = read(pre); assert vb.shape == vp.shape and trb == tr, 'снимки до и после на разной сетке'
    nbr = lambda X: (X[3] - X[5]) / (X[3] + X[5] + 1e-6); dnbr = nbr(Xb) - nbr(Xp); vpair = vp & vb; burn_d = (dnbr > 0.1) & vpair & ~wat
else: dnbr, vpair, burn_d = None, None, None

px_ha = abs(tr.a * tr.e) / 1e4
summary = dict(event=dict(detections=len(D), first=str(t0), last=str(t1), aoi_lonlat=aoi, sources=SRC), post=post.id, pre=pre.id if pre else None,
               model=dict(recipe='C1-F (5 сидов, 8 преобразований) + маска «Fmask и NDWI», порог 0.5', area_ha=round(float(burn.sum()) * px_ha, 1),
                          masked_fraction=round(masked_fraction(vp, vp.size), 4)),
               dnbr=None if pre is None else dict(rule='dNBR > 0.1', area_ha=round(float(burn_d.sum()) * px_ha, 1), masked_fraction=round(masked_fraction(vpair, vpair.size), 4),
                                                  agreement_iou=round(float((burn & burn_d).sum() / max((burn | burn_d).sum(), 1)), 4)),
               pixel_ha=px_ha, grid=dict(crs=str(crs), height=H, width=W))
prof = dict(driver='GTiff', height=H, width=W, crs=crs, transform=tr, compress='deflate', count=1)
with rasterio.open(OUT / 'burn_prob.tif', 'w', dtype='float32', **prof) as o: o.write(prob[None])
with rasterio.open(OUT / 'burn_model.tif', 'w', dtype='uint8', nodata=255, **prof) as o: o.write(np.where(vp, burn, 255).astype(np.uint8)[None])
if pre is not None:
    with rasterio.open(OUT / 'dnbr.tif', 'w', dtype='float32', nodata=np.nan, **prof) as o: o.write(np.where(vpair, dnbr, np.nan).astype(np.float32)[None])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rgb = np.clip(np.stack([Xp[5], Xp[3], Xp[2]], -1) / np.percentile(Xp[[5, 3, 2]][:, vp], 98), 0, 1)
fig, ax = plt.subplots(1, 2 if pre is not None else 1, figsize=(14 if pre is not None else 7, 7), squeeze=False)
ax[0, 0].imshow(rgb); ax[0, 0].contour(burn, levels=[0.5], colors='yellow', linewidths=0.8); ax[0, 0].set_title(f'модель C1-F: {summary["model"]["area_ha"]} га')
if pre is not None:
    ax[0, 1].imshow(np.where(vpair, dnbr, np.nan), cmap='inferno', vmin=0, vmax=0.8); ax[0, 1].contour(burn, levels=[0.5], colors='cyan', linewidths=0.6)
    ax[0, 1].set_title(f'dNBR (> 0.1: {summary["dnbr"]["area_ha"]} га), контур — модель')
for a in ax.flat: a.axis('off')
fig.suptitle(f'Кипр, {t0:%Y-%m-%d} — {t1:%Y-%m-%d}; снимок после {post.datetime:%Y-%m-%d}'); fig.tight_layout(); fig.savefig(OUT / 'overview.png', dpi=110)
json.dump(summary, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=1, default=str); print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
