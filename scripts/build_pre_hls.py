"""SPEC-86: снимок «до» для базовой линии dNBR — hls2-s30 того же тайла за 90 суток до возгорания, плохих пикселей в окне < 10 %
(последний подходящий). Выход <окно>.pre.tif (6 полос, −9999 — нодата). Запуск: build_pre_hls.py <каталог> <манифест>."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, rasterio, rasterio.warp, planetary_computer, pystac_client
from rasterio.windows import from_bounds
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.fire_dates import start_of   # SPEC-87: EMS и FLOGA без ig_date
D, MF = Path(sys.argv[1]), sys.argv[2]; BANDS = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
cat = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=planetary_computer.sign_inplace)

def pre(w):
    p = D / f"{w['name']}.pre.tif"
    if p.exists(): return w
    with rasterio.open(D / f"{w['name']}_merged.tif") as r: b, sh, prof = r.bounds, r.shape, r.profile
    tile = w['item'].split('.')[2]; ig = pd.Timestamp(start_of(w)).tz_localize(None)
    cand = [i for i in cat.search(collections=['hls2-s30'], bbox=list(rasterio.warp.transform_bounds(prof['crs'], 4326, *b)),
                                   datetime=f'{(ig - pd.Timedelta(days=90)).date()}/{(ig - pd.Timedelta(days=1)).date()}').items() if i.id.split('.')[2] == tile]
    for it in sorted(cand, key=lambda i: i.datetime, reverse=True):
        with rasterio.open(it.assets['Fmask'].href) as s:
            win = from_bounds(*b, transform=s.transform).round_offsets().round_lengths(); fm = s.read(1, window=win)
        bad = (fm == 255) | ((fm >> 1) & 1).astype(bool) | ((fm >> 3) & 1).astype(bool)
        if fm.shape != sh or bad.mean() >= 0.10: continue
        X = []
        for bn in BANDS:
            with rasterio.open(it.assets[bn].href) as s: X.append(s.read(1, window=win).astype(np.float32))
        X = np.stack(X); X = np.where(((X == -9999).any(0) | bad)[None], -9999, X * 0.0001).astype(np.float32)
        with rasterio.open(p, 'w', **dict(prof, count=6, dtype='float32', nodata=-9999, compress='deflate')) as o: o.write(X)
        return w
    return None

man = json.load(open(D / MF))
with ThreadPoolExecutor(4) as ex: keep = [x for x in ex.map(pre, man['windows']) if x]
json.dump(dict(man, windows=keep, pre_from=len(man['windows'])), open(D / f'{MF[:-5]}_pre.json', 'w'), ensure_ascii=False, indent=1)
print(D, 'снимков «до»', len(keep), 'из', len(man['windows']), flush=True)
