"""Bounded LIVE provenance probes for SPEC-34/35; training chips only."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import write_json,digest
from scripts.hypothesis_lab import manifest


def get_json(url):
    with urllib.request.urlopen(url,timeout=45) as f: return json.load(f)


def external(args):
    import requests
    import rasterio
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    report=manifest(args); repo='links-ads/wildfires-cems'; base=f'https://huggingface.co/datasets/{repo}'
    api=get_json(f'https://huggingface.co/api/datasets/{repo}')
    report.update(repository=repo,revision=api['sha'],license=api.get('cardData',{}).get('license'))
    tree=get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{api["sha"]}?recursive=true&limit=1000')
    report['files']=[{k:r[k] for k in ('path','size')} for r in tree if r['type']=='file']
    write_json(out/'manifest.json',report)
    url=f'{base}/resolve/{api["sha"]}/data/train/train.tar.0000.gz.part'
    # Only first train fragment; never fetch external val/test.
    seen=[]; sample=[]; event=None; start=time.time()
    try:
        with requests.get(url,stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw,mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile(): continue
                    seen.append(member.name)
                    if event is None and '_S2L2A.tif' in member.name:
                        event=Path(member.name).name.removesuffix('_S2L2A.tif')
                    # Archive order may put masks before imagery: retain bounded first files.
                    if len(sample)<30 and member.size<100_000_000 and member.name.endswith(('.tif','.json')):
                        target=out/Path(member.name).name
                        if target.exists(): raise ValueError('duplicate basename')
                        src=tar.extractfile(member)
                        with target.open('wb') as dst:
                            while chunk:=src.read(2**20): dst.write(chunk)
                        item=dict(name=member.name,bytes=member.size,sha256=digest(target))
                        if target.suffix=='.tif':
                            with rasterio.open(target) as ds:
                                arr=ds.read(out_shape=(ds.count,32,32)); item.update(count=ds.count,shape=[ds.height,ds.width],crs=str(ds.crs),bounds=list(ds.bounds),descriptions=list(ds.descriptions),tags=ds.tags(),min=float(arr.min()),max=float(arr.max()))
                        else:
                            try: item['metadata']=json.loads(target.read_text())
                            except (ValueError,UnicodeError): item['metadata']='unreadable JSON'
                        sample.append(item)
                    if any(x['name'].endswith('_S2L2A.tif') for x in sample) and any(x['name'].endswith('_DEL.tif') for x in sample): break
                    if len(seen)>200 or time.time()-start>300: break
        result=dict(status='sample_downloaded',sample=sample,archive_entries=seen,seconds=time.time()-start)
    except Exception as e:
        # Signed download URLs may contain credentials; store class/status only.
        result=dict(status='source_error',exception=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),sample=sample,archive_entries=seen)
    write_json(out/'result.json',result); print(json.dumps(result,indent=2),flush=True)


def temporal(args):
    import requests
    import rasterio
    from rasterio.warp import transform_bounds
    from scripts.train_unet import load_split
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); write_json(out/'manifest.json',manifest(args))
    d,fit,tune=load_split(args.data,args.split); meta=d.meta.set_index('chip_id'); rows=[]
    for c in fit[:8]:
        row=meta.loc[c]; path=Path(args.data)/'sentinel2_pre'/f'{c}_Sentinel-2_pre.tif'
        with rasterio.open(path) as ds:
            bbox=transform_bounds(ds.crs,'EPSG:4326',*ds.bounds)
            record=dict(chip=c,source_sha256=digest(path),bbox=bbox,dates={k:str(row[k]) for k in ('date_pre','date_post')},items={})
        for kind in ('pre','post'):
            date=dt.date.fromisoformat(str(row['date_'+kind])); lo=date-dt.timedelta(days=30) if kind=='pre' else date; hi=date if kind=='pre' else date+dt.timedelta(days=30)
            query=dict(collections=['sentinel-2-l2a'],bbox=list(bbox),datetime=f'{lo}T00:00:00Z/{hi}T23:59:59Z',limit=100)
            try:
                r=requests.post('https://planetarycomputer.microsoft.com/api/stac/v1/search',json=query,timeout=45); r.raise_for_status(); data=r.json()
                items=[]
                for f in data['features']:
                    prop=f['properties']; item=dict(id=f['id'],datetime=prop['datetime'],mgrs=prop.get('s2:mgrs_tile'),cloud=prop.get('eo:cloud_cover'))
                    if item['mgrs'] is None: item['mgrs']=f['id'].split('_')[1] if '_' in f['id'] else None
                    items.append(item)
                record['items'][kind]=items
                record[kind+'_query']=query
            except Exception as e: record['items'][kind]=dict(error=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None))
        rows.append(record); print(c,'pre/post',*[len(record['items'][x]) for x in ('pre','post')],flush=True)
    write_json(out/'catalog.json',rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['external','temporal']); p.add_argument('--out',required=True); p.add_argument('--data',default='data/comp/train/bs'); p.add_argument('--split',default='data/comp/split_bs.json'); args=p.parse_args(); (external if args.task=='external' else temporal)(args)
if __name__=='__main__': main()
