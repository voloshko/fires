"""SPEC-35: match source pixels, then fetch pre-only augmentation with fixed QC."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import write_json,digest,harmonize_s2_dn


def main():
    import planetary_computer as pc
    import rasterio
    import requests
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from src.comp.chips import BsDataset,SCL_INVALID
    parser=argparse.ArgumentParser(); parser.add_argument('--catalog',default='research/temporal-v1'); parser.add_argument('--out',default='research/temporal-pre-v1'); args=parser.parse_args()
    root=Path(args.catalog); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((root/'catalog.json').read_text()); dataset=BsDataset('data/comp/train/bs'); records=[]
    api='https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/'
    bands=['B02','B03','B04','B05','B06','B07','B8A','B11','B12','SCL']
    for entry in catalog:
        c=entry['chip']; ch=dataset.load(c); record=dict(chip=c,accepted=False,checks=[])
        with rasterio.open(f'data/comp/train/bs/sentinel2_pre/{c}_Sentinel-2_pre.tif') as ds: crs,transform=ds.crs,ds.transform
        def read(item_id,selected):
            response=requests.get(api+item_id,timeout=45); response.raise_for_status(); item=response.json(); arrays=[]
            for band in selected:
                href=pc.sign(item['assets'][band]['href'])
                with rasterio.Env(GDAL_HTTP_TIMEOUT='45',GDAL_HTTP_MAX_RETRY='1',GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                    with rasterio.open(href) as ds:
                        with WarpedVRT(ds,crs=crs,transform=transform,width=512,height=512,resampling=Resampling.nearest if band=='SCL' else Resampling.bilinear,nodata=0) as vrt: arrays.append(vrt.read(1))
            return harmonize_s2_dn(np.stack(arrays),selected,item['properties'].get('s2:processing_baseline')),item
        if not isinstance(entry['items'].get('pre'),list):
            record['reason']='catalog_source_error'; records.append(record); print(json.dumps(record),flush=True); continue
        originals=[x for x in entry['items']['pre'] if x['datetime'][:10]==entry['dates']['date_pre']]
        originals=sorted(originals,key=lambda x:x['id']); matched=None
        try:
            for item in originals:
                a,meta=read(item['id'],['B04','B8A','B12','SCL']); ok=~np.isin(a[3],SCL_INVALID)&~np.isin(ch.pre[9],SCL_INVALID)&(a[:3]>0).all(0)
                coverage=float((a[:3]>0).all(0).mean())
                if ok.sum()<100: continue
                x=a[:3,ok].astype(float)/10000; y=ch.pre[[2,6,8]][:,ok].astype(float)/10000
                corr=float(np.corrcoef(x.ravel(),y.ravel())[0,1]); rmse=float(np.sqrt(np.mean((x-y)**2)))
                record['checks'].append(dict(item=item['id'],processing_baseline=meta['properties'].get('s2:processing_baseline'),harmonized_dn=True,coverage=coverage,correlation=corr,rmse=rmse))
                if corr>=.95 and rmse<=.03 and coverage>=.9: matched=item; break
            if matched is None: record['reason']='original_scene_not_matched'
            else:
                record['original_item']=matched['id']; tile=matched['id'].split('_')[4]
                extra=[x for x in entry['items']['pre'] if x['datetime'][:10]<entry['dates']['date_pre'] and x['id'].split('_')[4]==tile]
                extra=sorted(extra,key=lambda x:(x['datetime'],x['id']),reverse=True)
                for item in extra[:3]:
                    a,extra_meta=read(item['id'],bands); valid=~np.isin(a[9],SCL_INVALID); cloudy=1-float(valid.mean())
                    mask=(ch.mask>0)&valid&~np.isin(ch.pre[9],SCL_INVALID)
                    def nbr(v):
                        x,y=v[6].astype(float),v[8].astype(float); return (x-y)/np.maximum(x+y,1)
                    delta=float(np.median(np.abs(nbr(a)[mask]-nbr(ch.pre)[mask]))) if mask.sum() else None
                    record['checks'].append(dict(extra_item=item['id'],processing_baseline=extra_meta['properties'].get('s2:processing_baseline'),harmonized_dn=True,masked_fraction=cloudy,burn_pixels_compared=int(mask.sum()),median_nbr_change=delta))
                    if cloudy<=.3 and delta is not None and delta<=.05 and mask.sum()>=.5*np.count_nonzero(ch.mask):
                        a=np.clip(a,0,65535).astype(np.uint16); np.savez_compressed(out/f'{c}.npz',pre=a)
                        record.update(accepted=True,extra_item=item['id'],extra_sha256=digest(out/f'{c}.npz')); break
                if not record['accepted']: record['reason']='no_extra_passes_frozen_qc'
        except Exception as e: record.update(reason='source_error',exception=type(e).__name__)
        records.append(record); print(json.dumps(record),flush=True)
    write_json(out/'result.json',dict(records=records,accepted=sum(r['accepted'] for r in records),minimum_required=4,post_changed=False))
if __name__=='__main__': main()
