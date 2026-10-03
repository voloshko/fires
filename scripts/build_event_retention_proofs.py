"""SPEC-97: partition original HLS positives into known and protected pixels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file, canonical_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    args=p.parse_args();source=args.source_root.resolve();root=args.data_root.resolve()
    registry=json.loads(args.registry.read_text())
    vector=source/'external/mtbs/mtbs_perims_DD.shp'
    for name,digest in registry['vector_hashes'].items():
        if sha256_file(vector.parent/name)!=digest:raise ValueError('source vector changed')
    frame=gpd.read_file(vector)
    outputs=[];out=root/'event-retention';out.mkdir(parents=True,exist_ok=True)
    for record in registry['records']:
        image=source/record['source'];mask=Path(str(image).replace('_merged.tif','.mask.tif'))
        if sha256_file(image)!=record['source_sha256'] or sha256_file(mask)!=record['label_sha256']:
            raise ValueError('original event-audit input changed')
        with rasterio.open(image) as raster,rasterio.open(mask) as labels:
            array=raster.read();truth=labels.read(1)
            positive=(truth==1)&(array!=-9999).all(0)&np.isfinite(array).all(0)
            known=np.zeros(raster.shape,bool)
            if record['event_id']:
                polygons=frame[frame.event_id==record['event_id']].to_crs(raster.crs)
                if polygons.empty:raise ValueError('audited event disappeared')
                known=rasterize([(g,1) for g in polygons.geometry],out_shape=raster.shape,
                                transform=raster.transform,fill=0,dtype='uint8').astype(bool)
            proof=np.where(positive,np.where(known,1,2),0).astype(np.uint8)
            path=out/(canonical_hash(record['sample_id'])+'.tif')
            if path.exists():
                with rasterio.open(path) as existing:
                    if not np.array_equal(existing.read(1),proof):raise ValueError('proof changed')
            else:
                with rasterio.open(path,'w',**dict(raster.profile,count=1,dtype='uint8',nodata=0)) as target:
                    target.write(proof[None])
        outputs.append(dict(sample_id=record['sample_id'],event_key=None if not record['event_id'] else 'MTBS:'+record['event_id'],
                            proof=str(path.relative_to(root)),proof_sha256=sha256_file(path),
                            known_pixels=int((proof==1).sum()),protected_pixels=int((proof==2).sum()),
                            source_sha256=record['source_sha256'],label_sha256=record['label_sha256']))
    payload=dict(spec_id='SPEC-97',registry_sha256=sha256_file(args.registry),records=outputs)
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=payload:raise ValueError('retention proof manifest changed')
    else:
        with manifest.open('x') as stream:json.dump(payload,stream,indent=2)
    print('RESULT',len(outputs),'windows',sum(r['protected_pixels'] for r in outputs),'protected pixels',flush=True)


if __name__=='__main__':main()
