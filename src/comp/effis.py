"""SPEC-84: контур гарей EFFIS (публичный WMS, слои modis.ba.<год>) в сетке нашего окна. EFFIS не отдаёт UTM, поэтому
картинка берётся в EPSG:3035 с шагом 10 м и пересчитывается в окно ближайшим соседом; гарь — заливка с альфой ≥ 120."""
import io, math, re
import numpy as np, requests
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling

WMS = 'https://maps.effis.emergency.copernicus.eu/effis'


def effis_mask(crs, transform, shape, year, step=10.0):
    h, w = shape; b = (transform.c, transform.f + transform.e * h, transform.c + transform.a * w, transform.f)
    x0, y0, x1, y1 = transform_bounds(crs, 'EPSG:3035', *b); W, H = min(4000, math.ceil((x1 - x0) / step)), min(4000, math.ceil((y1 - y0) / step))
    r = requests.get(WMS, params=dict(SERVICE='WMS', VERSION='1.1.1', REQUEST='GetMap', LAYERS=f'modis.ba.{year}', SRS='EPSG:3035', BBOX=f'{x0},{y0},{x1},{y1}',
                                      WIDTH=W, HEIGHT=H, FORMAT='image/png', TRANSPARENT='true', STYLES=''), timeout=180)
    if not r.headers.get('content-type', '').startswith('image'): raise RuntimeError(f'EFFIS: {r.text[:200]}')
    src = (np.array(Image.open(io.BytesIO(r.content)).convert('RGBA'))[..., 3] >= 120).astype(np.uint8)
    out = np.zeros(shape, np.uint8)
    reproject(src, out, src_transform=from_bounds(x0, y0, x1, y1, W, H), src_crs='EPSG:3035', dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest)
    return out.astype(bool)


def effis_info(lon, lat, year, d=0.01):
    """Атрибуты пожара EFFIS в точке: даты, страна, место, площадь, га (HTML GetFeatureInfo)."""
    r = requests.get(WMS, params=dict(SERVICE='WMS', VERSION='1.1.1', REQUEST='GetFeatureInfo', LAYERS=f'modis.ba.{year}', QUERY_LAYERS=f'modis.ba.{year}', SRS='EPSG:4326',
                                      BBOX=f'{lon - d},{lat - d},{lon + d},{lat + d}', WIDTH=101, HEIGHT=101, X=50, Y=50, INFO_FORMAT='text/html', FEATURE_COUNT=1, STYLES=''), timeout=60)
    cells = re.findall(r'<td>([^<]*)</td>\s*<td>([^<]*)</td>', r.text)
    return {k.strip(): v.strip() for k, v in cells} or None
