"""SPEC-83 ч. Б: задержка карты против Copernicus EMS. (1) Задержка источника за последние 60 суток над Кипром, Грецией,
Испанией: HLS (CMR updated − time_start), Earth Search L2A (created − datetime), генерация ESA (время в s2:product_uri −
datetime). (2) По каждой активации EMS: первый пролёт после eventTime с облачностью сцены < 20 % + медиана задержки
источника + 10 минут счёта против первого продукта EMS; медиана (EMS − мы) и 95 % бутстреп по активациям."""
import glob, json, re
from datetime import datetime, timedelta, timezone
import numpy as np, requests, pystac_client

ES = pystac_client.Client.open('https://earth-search.aws.element84.com/v1'); NOW = datetime.now(timezone.utc)
PTS = {'Кипр': (33.0, 34.8), 'Греция': (23.7, 38.0), 'Испания': (-4.0, 40.0)}
P = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')) if '+' in s or 'Z' in s else datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
lag = {'HLS (CMR)': [], 'L2A Earth Search': [], 'L2A генерация ESA': []}
for name, (lon, lat) in PTS.items():
    t0 = (NOW - timedelta(days=60)).strftime('%Y-%m-%dT%H:%M:%SZ'); t1 = NOW.strftime('%Y-%m-%dT%H:%M:%SZ')
    for sn in ('HLSS30', 'HLSL30'):
        e = requests.get(f'https://cmr.earthdata.nasa.gov/search/granules.json?short_name={sn}&point={lon},{lat}&temporal={t0},{t1}&page_size=100', timeout=60).json()['feed']['entry']
        lag['HLS (CMR)'] += [(P(x['updated']) - P(x['time_start'])).total_seconds() / 3600 for x in e]
    for it in ES.search(collections=['sentinel-2-c1-l2a'], intersects={'type': 'Point', 'coordinates': [lon, lat]}, datetime=f'{t0}/{t1}', max_items=100).items():
        dt = P(it.properties['datetime'])
        if it.properties.get('created'): lag['L2A Earth Search'].append((P(it.properties['created']) - dt).total_seconds() / 3600)
        g = re.search(r'_(\d{8}T\d{6})\.SAFE$', it.properties.get('s2:product_uri', ''))
        if g:
            h = (datetime.strptime(g.group(1), '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc) - dt).total_seconds() / 3600
            if 0 < h < 72: lag['L2A генерация ESA'].append(h)
med = {}
for k, v in lag.items():
    v = np.array(v); med[k] = float(np.median(v)); print(f'задержка источника {k:20s} n={len(v):4d} | медиана {np.median(v):6.1f} ч | 90 % {np.percentile(v, 90):6.1f} ч', flush=True)

rows = []
for f in sorted(glob.glob('external/ems/acts/*.json')):
    r = json.load(open(f))['results'][0]
    if r.get('category') != 'Wildfire': continue
    dels = [pr['version']['deliveryTime'] for a in r['aois'] for pr in a['products'] if pr.get('version') and pr['version'].get('deliveryTime') and pr['type'] in ('FEP', 'DEL', 'GRA')]
    if not dels: continue
    ev = P(r['eventTime']); ems = min(P(x) for x in dels); lon, lat = (float(x) for x in re.findall(r'-?\d+\.?\d*', r['centroid'])[:2])
    its = sorted(ES.search(collections=['sentinel-2-c1-l2a'], intersects={'type': 'Point', 'coordinates': [lon, lat]},
                           datetime=f"{ev:%Y-%m-%dT%H:%M:%SZ}/{ev + timedelta(days=10):%Y-%m-%dT%H:%M:%SZ}", max_items=50).items(), key=lambda i: i.properties['datetime'])
    first = next((P(i.properties['datetime']) for i in its if i.properties.get('eo:cloud_cover', 100) < 20), None)
    if first is None: continue
    rows.append(dict(code=r['code'], ems_h=(ems - ev).total_seconds() / 3600, pass_h=(first - ev).total_seconds() / 3600))
print(f'\nактиваций с пролётом < 20 % облаков за 10 суток: {len(rows)}', flush=True)
e = np.array([x['ems_h'] for x in rows]); ps = np.array([x['pass_h'] for x in rows]); rng = np.random.default_rng(0)
print(f'EMS: событие → первый продукт, медиана {np.median(e):.1f} ч | первый чистый пролёт после события, медиана {np.median(ps):.1f} ч')
for k in lag:
    ours = ps + med[k] + 10 / 60; d = e - ours; b = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]; lo, hi = np.percentile(b, [2.5, 97.5])
    print(f'  путь {k:20s}: наша карта, медиана {np.median(ours):6.1f} ч | EMS − мы, медиана {np.median(d):+6.1f} ч, 95 % [{lo:+.1f}, {hi:+.1f}] | быстрее EMS в {np.mean(d > 0):.0%} активаций → {"БЫСТРЕЕ" if np.median(d) > 0 and lo > 0 else "не быстрее"}')
json.dump(dict(source_lag_median_h=med, rows=rows), open('research/latency-bench-v1.json', 'w'), ensure_ascii=False, indent=1)
