"""SPEC-117: freeze bounded public activation listings without collecting imagery."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urljoin, urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe
from src.comp.accuracy_contract import sha256_file
from src.comp.cems_source_availability import summarize

BUDGET = 16*1024**2
FLOOR = 3*1024**3
ENDPOINTS = {
    'rapid': 'https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/?limit=1000',
    'historical': 'https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=2000&activationTime=2012-01%2C2026-10-04',
}


def write_once(directory, path, content):
    used = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if used+len(content) > BUDGET or shutil.disk_usage(directory).free-len(content) < FLOOR:
        raise ValueError('metadata budget/free-space floor unavailable')
    with path.open('xb') as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to(Path('research').resolve()):
        raise ValueError('fresh own metadata output required')
    if shutil.disk_usage(args.out.parent).free < FLOOR+BUDGET:
        raise ValueError('metadata free-space floor unavailable')
    args.out.mkdir()
    recipe = freeze_recipe(args.out/'recipe.json', execution_sources(Path(__file__),
        'src/comp/cems_source_availability.py'), dict(endpoints=ENDPOINTS, images=False,
        filter_basis='declared country labels only; activation timestamps not ignition'))
    snapshots = {}
    for name, first in ENDPOINTS.items():
        url, pages, rows, count, visited = first, [], [], None, set()
        while url is not None:
            if url in visited or len(visited) >= 20 or urlsplit(url).netloc != urlsplit(first).netloc:
                raise ValueError('invalid public pagination chain')
            visited.add(url)
            with requests.get(url, timeout=30, stream=True) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > BUDGET:
                        raise ValueError('public metadata page too large')
                    chunks.append(chunk)
                content = b''.join(chunks)
                payload = json.loads(content)
                if not isinstance(payload.get('results'), list) or type(payload.get('count')) is not int:
                    raise ValueError('malformed public listing')
                if count is None: count = payload['count']
                if payload['count'] != count:
                    raise ValueError('pagination count changed during snapshot')
                path = args.out/(name+f'-page{len(pages)}.json')
                write_once(args.out, path, content)
                pages.append(dict(url=url, response_url=response.url, status=response.status_code,
                    captured_at=datetime.now(timezone.utc).isoformat(), path=str(path), sha256=sha256_file(path)))
            rows.extend(payload['results'])
            following = payload.get('next')
            if following is not None and not isinstance(following, str):
                raise ValueError('malformed next page')
            url = urljoin(url, following) if following else None
        snapshots[name] = dict(pages=pages, declared_count=count, complete=True, summary=summarize(rows, count))
    result = dict(spec_id='SPEC-117', recipe_id=recipe, snapshots=snapshots,
        non_claims=['Snapshot of two public activation listings, not every mapping source or absence of fire.',
            'Country tags and activation timestamps do not prove physical ignition, labels, imagery admission or independent fire counts.',
            'No data split/training, Russian accuracy, model promotion or original goal completion.'])
    write_once(args.out, args.out/'result.json', json.dumps(result, indent=2).encode())
    print(json.dumps({k: dict(records=v['summary']['records'], russian=len(v['summary']['declared_russian_records']),
        russian_wildfires=len(v['summary']['declared_russian_wildfires']), boreal=len(v['summary']['boreal_wildfire_activations']),
        post2015_boreal=len(v['summary']['post2015_boreal_activations']), unknown_country=len(v['summary']['unknown_country_records']))
        for k,v in snapshots.items()}), flush=True)


if __name__ == '__main__':
    main()
