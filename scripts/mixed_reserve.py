"""SPEC-82: FLOGA делится пополам по sha256('floga:' + id); обучающая половина без окон, пересекающих проверку FLOGA и EMS."""
import hashlib, json
from pathlib import Path
import geopandas as gpd, rasterio
from shapely.geometry import box
from shapely.ops import unary_union

def boxes(d, windows):
    out = []
    for w in windows:
        with rasterio.open(Path(d) / f"{w['name']}_merged.tif") as r: out.append(gpd.GeoSeries([box(*r.bounds)], crs=r.crs).to_crs(4326).iloc[0])
    return out

F = Path('external/floga_hls'); fl = json.load(open(F / 'manifest.json'))['windows']
test = [w for w in fl if int(hashlib.sha256(f"floga:{w['id']}".encode()).hexdigest(), 16) % 2 == 0]; train = [w for w in fl if w not in test]
ems_test = json.load(open('external/ems_hls/manifest_test.json'))['windows']
block = unary_union(boxes(F, test) + boxes('external/ems_hls', ems_test))
keep = [w for w, b in zip(train, boxes(F, train)) if not b.intersects(block)]
json.dump(dict(spec='SPEC-82', split='проверка', windows=test), open(F / 'manifest_test.json', 'w'), ensure_ascii=False, indent=1)
json.dump(dict(spec='SPEC-82', split='обучение', rule='рамка не пересекает проверку FLOGA и EMS', half=len(train), windows=keep), open(F / 'manifest_train.json', 'w'), ensure_ascii=False, indent=1)
print(f'FLOGA: проверка {len(test)} | обучающая половина {len(train)} → чистых {len(keep)}')
