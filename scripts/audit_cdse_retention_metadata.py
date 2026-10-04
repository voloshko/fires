"""SPEC-120: public metadata for exactly the16 failed protected-pixel windows."""
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
from src.comp.cdse_retention_metadata import ENDPOINT, MANIFEST_SHA, descriptors, lost_population, query

BUDGET, FLOOR = 8*1024**2, 3*1024**3


def anonymous_session():
    client=requests.Session()
    client.trust_env=False  # Disable .netrc, proxy authentication and environment configuration.
    return client


def write_once(root, path, value):
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used+len(value)>BUDGET or shutil.disk_usage(root).free-len(value)<FLOOR:
        raise ValueError('metadata budget/free-space unavailable')
    with path.open('xb') as stream:stream.write(value)


def fetch(root, index, plan, pages):
    params={'$filter':plan['filter'],'$select':'Id,Name,Online,ContentDate,PublicationDate',
            '$count':'true','$top':100,'$orderby':'PublicationDate desc'}
    url=ENDPOINT;seen=set();rows=[];count=None
    while url is not None:
        if url in seen or len(seen)>=20 or urlsplit(url).netloc!=urlsplit(ENDPOINT).netloc:
            raise ValueError('invalid metadata pagination')
        seen.add(url)
        with anonymous_session() as client, client.get(url,params=params,timeout=30,stream=True) as response:
            chunks=[];size=0
            for part in response.iter_content(65536):
                size+=len(part)
                if size>BUDGET:raise ValueError('public metadata page too large')
                chunks.append(part)
            raw=b''.join(chunks)
            path=root/f'window{index:02d}-page{len(pages)}.json';write_once(root,path,raw)
            pages.append(dict(url=response.url,status=response.status_code,captured_at=datetime.now(timezone.utc).isoformat(),
                              path=str(path),sha256=sha256_file(path)))
            response.raise_for_status();value=json.loads(raw)
            if not isinstance(value.get('value'),list) or type(value.get('@odata.count')) is not int:
                raise ValueError('catalogue count/schema unavailable')
            if count is None:count=value['@odata.count']
            if value['@odata.count']!=count:raise ValueError('catalogue count changed')
        rows.extend(value['value']);following=value.get('@odata.nextLink')
        if following is not None and not isinstance(following,str):raise ValueError('malformed next link')
        url=urljoin(url,following) if following else None;params=None
    return dict(complete=True,pages=pages,count=count,
                products=descriptors(rows,count,plan['tile'],plan['acquisition_day']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if sha256_file(args.manifest)!=MANIFEST_SHA or args.out.exists() or not args.out.resolve().is_relative_to(Path('research').resolve()):
        raise ValueError('immutable final manifest/fresh own output required')
    manifest=json.loads(args.manifest.read_text());losses=lost_population(manifest)
    if shutil.disk_usage(args.out.parent).free<FLOOR+BUDGET:raise ValueError('metadata free floor unavailable')
    args.out.mkdir()
    recipe=freeze_recipe(args.out/'recipe.json',execution_sources(Path(__file__),'src/comp/cdse_retention_metadata.py'),
        dict(manifest_sha256=MANIFEST_SHA,endpoint=ENDPOINT,losses=losses,credential_access=False,pixel_download=False))
    results=[]
    for i,loss in enumerate(losses):
        plan=query(loss['sample_id']);pages=[]
        try:
            report=fetch(args.out,i,plan,pages)
        except Exception as error:
            status=error.response.status_code if isinstance(error,requests.HTTPError) and error.response is not None else None
            report=dict(complete=False,count=None,products=None,pages=pages,error_type=type(error).__name__,status=status)
        results.append(dict(loss,**plan,catalogue=report))
        print('QUERIED',loss['sample_id'],report['complete'],report['count'],flush=True)
    complete=[r for r in results if r['catalogue']['complete']]
    result=dict(spec_id='SPEC-120',recipe_id=recipe,manifest_sha256=MANIFEST_SHA,records=results,
        summary=dict(windows=len(results),complete_queries=len(complete),failed_queries=len(results)-len(complete),
            complete_empty_queries=sum(r['catalogue']['count']==0 for r in complete),
            online_windows=sum(any(p['online_state']=='online' for p in r['catalogue']['products']) for r in complete)),
        source_admission='NOT_TESTED',l2a_ready=False,original_readiness_unchanged=True,
        non_claims=['Online metadata is not source pixels, cloud-free coverage or protected-label retention.',
            'Query failure is unknown, not no product/fire. No credential access or image downloads.',
            'No original admission change, L2A useful gain, model promotion or original goal completion.'])
    write_once(args.out,args.out/'result.json',json.dumps(result,indent=2).encode());print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
