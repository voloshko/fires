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
    r.encoding = 'utf-8'; cells = re.findall(r'<td>([^<]*)</td>\s*<td>([^<]*)</td>', r.text)
    return {k.strip(): v.strip() for k, v in cells} or None


def effis_mask_retry(crs, transform, shape, year, tries=8):
    """Сервер EFFIS обрывает ответ (IncompleteRead) на части запросов: сначала повтор, потом другие шаги,
    потом четыре четверти окна по отдельности. Каждая ступень — та же геометрия, меняется только растровый запрос."""
    import time
    from rasterio.transform import Affine
    last = None
    for step in (10.0, 20.0, 15.0, 25.0, 12.0):
        for k in range(2):
            try: return effis_mask(crs, transform, shape, year, step=step)
            except Exception as e: last = e; time.sleep(3 * (k + 1))
    h, w = shape; h2, w2 = h // 2, w // 2; out = np.zeros(shape, bool)
    for r0, r1 in ((0, h2), (h2, h)):
        for c0, c1 in ((0, w2), (w2, w)):
            t = transform * Affine.translation(c0, r0)
            for k in range(4):
                try: out[r0:r1, c0:c1] = effis_mask(crs, t, (r1 - r0, c1 - c0), year, step=10.0); break
                except Exception as e: last = e; time.sleep(5 * (k + 1))
            else: raise RuntimeError(f'EFFIS не ответил даже по четвертям: {last!r}'[:300])
    return out
