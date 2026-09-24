"""SPEC-81: резерв EMS для обучения без окон, чья рамка пересекает окна проверки EMS или FLOGA → external/ems_hls/manifest_train.json."""
import json
from pathlib import Path
import geopandas as gpd, rasterio
from shapely.geometry import box
from shapely.ops import unary_union

def boxes(d, windows):
    out = []
    for w in windows:
        with rasterio.open(Path(d) / f"{w['name']}_merged.tif") as r: out.append(gpd.GeoSeries([box(*r.bounds)], crs=r.crs).to_crs(4326).iloc[0])
    return out

E = Path('external/ems_hls'); man = json.load(open(E / 'manifest.json'))
test = [w for w in man['windows'] if w['split'] == 'проверка']; res = [w for w in man['windows'] if w['split'] == 'резерв']
fl = json.load(open('external/floga_hls/manifest.json'))['windows']
block = unary_union(boxes(E, test) + boxes('external/floga_hls', fl))
keep = [w for w, b in zip(res, boxes(E, res)) if not b.intersects(block)]
json.dump(dict(spec='SPEC-81', split='резерв', rule='рамка не пересекает окна проверки EMS и FLOGA', reserve=len(res), windows=keep),
          open(E / 'manifest_train.json', 'w'), ensure_ascii=False, indent=1)
print(f'резерв {len(res)} → чистых {len(keep)} (выброшено {len(res) - len(keep)})')
