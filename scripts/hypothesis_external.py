"""SPEC-34: bounded extraction of official CEMS TRAIN and shared-encoder pretrain."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import digest,write_json
from scripts.hypothesis_sources import get_json


class MultipartReader(io.RawIOBase):
    def __init__(self,urls): self.urls=iter(urls); self.response=None; self.bytes_read=0
    def readable(self): return True
    def read(self,size=-1):
        import requests
        if size<0: raise ValueError('bounded reads required')
        chunks=[]; remaining=size
        while remaining:
            if self.response is None:
                url=next(self.urls,None)
                if url is None: break
                self.response=requests.get(url,stream=True,timeout=(30,90)); self.response.raise_for_status()
            data=self.response.raw.read(remaining)
            if not data: self.response.close(); self.response=None; continue
            chunks.append(data); remaining-=len(data); self.bytes_read+=len(data)
        return b''.join(chunks)
    def close(self):
        if self.response is not None: self.response.close()
        super().close()


def download(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    repo='links-ads/wildfires-cems'; info=get_json(f'https://huggingface.co/api/datasets/{repo}')
    tree=get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{info["sha"]}?recursive=true&limit=1000')
    parts=sorted(r['path'] for r in tree if r['path'].startswith('data/train/train.tar.'))
    urls=[f'https://huggingface.co/datasets/{repo}/resolve/{info["sha"]}/{p}' for p in parts]
    write_json(out/'source.json',dict(repository=repo,revision=info['sha'],license=info.get('cardData',{}).get('license'),parts=parts,only_split='train',max_events=args.events))
    chosen={}; files=[]; stream=MultipartReader(urls); started=time.time()
    try:
        with tarfile.open(fileobj=stream,mode='r|gz') as tar:
            for member in tar:
                if not member.isfile(): continue
                path=Path(member.name); segments=path.parts
                if segments[0]!='train': raise ValueError('unexpected split')
                if not path.name.endswith(('_S2L2A.tif','_S2L2A.json','_DEL.tif','_CM.tif')): continue
                event=segments[1]; prefix=path.name.rsplit('_',1)[0]
                if event not in chosen:
                    if len(chosen)>=args.events: break
                    chosen[event]=[]
                if prefix not in chosen[event]:
                    if len(chosen[event])>=4: continue
                    chosen[event].append(prefix)
                if member.size>300_000_000: raise ValueError('unexpectedly large member')
                target=out/'samples'/path.name; target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists(): raise FileExistsError(target)
                with target.open('wb') as f:
                    src=tar.extractfile(member)
                    while chunk:=src.read(2**20): f.write(chunk)
                files.append(dict(file=str(target),source_name=member.name,sha256=digest(target),event=event))
                if len(files)%20==0: print('events',len(chosen),'files',len(files),'downloaded_MB',stream.bytes_read//10**6,flush=True)
    finally: stream.close()
    write_json(out/'download.json',dict(events=chosen,files=files,bytes_read=stream.bytes_read,seconds=time.time()-started))


def prepare(args):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds,calculate_default_transform
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from src.comp.chips import BsDataset
    from scripts.train_unet import load_split
    out=Path(args.out); downloaded=json.loads((out/'download.json').read_text()); source=json.loads((out/'source.json').read_text())
    if source['license']!='cc-by-4.0': raise ValueError('license changed')
    d,fit,tune=load_split('data/comp/train/bs','data/comp/split_bs.json'); local=[]
    for c in fit+tune:
        with rasterio.open(f'data/comp/train/bs/sentinel2_pre/{c}_Sentinel-2_pre.tif') as ds: local.append(transform_bounds(ds.crs,'EPSG:4326',*ds.bounds))
    def overlaps(a,b): return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]
    records=[]; patches=[]; root=out/'patches-v2'; root.mkdir(exist_ok=True); per_event={}
    for info in downloaded['files']:
        p=Path(info['file'])
        if not p.name.endswith('_S2L2A.tif'): continue
        event=info['event']; rec=dict(image=str(p),event=event,accepted=False)
        if per_event.get(event,0)>=4: continue
        stem=p.name.removesuffix('_S2L2A.tif'); mask=p.with_name(stem+'_DEL.tif'); meta=p.with_suffix('.json')
        if not mask.exists() or not meta.exists(): rec['reason']='missing_mask_or_metadata'; records.append(rec); continue
        metadata=json.loads(meta.read_text()); payload=metadata['payload']; script=payload['evalscript']
        expected=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']
        if not all(f'"{band}"' in script for band in expected): raise ValueError('band schema changed')
        with rasterio.open(p) as ds,rasterio.open(mask) as ms:
            bbox=transform_bounds(ds.crs,'EPSG:4326',*ds.bounds); rec['bbox']=bbox; rec['dates']=payload.get('acquisition_date')
            if not rec['dates']: rec['reason']='missing_date'; records.append(rec); continue
            if any(overlaps(bbox,b) for b in local): rec['reason']='overlap_local_development'; records.append(rec); continue
            lon,lat=(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2; epsg=(32600 if lat>=0 else 32700)+int((lon+180)//6)+1
            transform,w,h=calculate_default_transform(ds.crs,f'EPSG:{epsg}',ds.width,ds.height,*ds.bounds,resolution=20)
            with WarpedVRT(ds,crs=f'EPSG:{epsg}',transform=transform,width=w,height=h,resampling=Resampling.bilinear) as image, WarpedVRT(ms,crs=f'EPSG:{epsg}',transform=transform,width=w,height=h,resampling=Resampling.nearest) as label:
                labels=label.read(1)
                if not set(np.unique(labels)).issubset({0,1}): raise ValueError('nonbinary DEL')
                rec['grid_20m']=[h,w]
                candidates=[]
                for y in range(0,max(1,h-511),256):
                    for x in range(0,max(1,w-511),256):
                        if y+512<=h and x+512<=w:
                            area=int(labels[y:y+512,x:x+512].sum())
                            if area>=100: candidates.append((y,x))
                if not candidates and np.count_nonzero(labels)>=100:
                    yy,xx=np.nonzero(labels)
                    candidates=[(max(0,min(max(0,h-512),int(np.median(yy))-256)),max(0,min(max(0,w-512),int(np.median(xx))-256)))]
                for y,x in candidates:
                    if per_event.get(event,0)>=4: break
                    hh,ww=min(512,h-y),min(512,w-x)
                    a=np.zeros((9,512,512),np.float32); b=np.full((512,512),255,np.uint8)
                    a[:,:hh,:ww]=image.read([2,3,4,5,6,7,9,11,12],window=Window(x,y,ww,hh)); b[:hh,:ww]=labels[y:y+hh,x:x+ww]
                    b[np.all(a==0,axis=0)]=255
                    if np.count_nonzero(b!=255)<4096: continue
                    if not np.isfinite(a).all() or np.quantile(a,.99)>2 or np.quantile(a,.01)<-.1: raise ValueError('unverified reflectance scale')
                    patch=root/f'{stem}_{x}_{y}.npz'; np.savez_compressed(patch,image=a.astype(np.float16),mask=b.astype(np.uint8))
                    patches.append(dict(file=str(patch),sha256=digest(patch),event=event)); per_event[event]=per_event.get(event,0)+1
                rec['accepted']=per_event.get(event,0)>0
        records.append(rec)
    write_json(out/'prepared-v2.json',dict(records=records,patches=patches,events=len(per_event),minimum_events=20,eligible=len(per_event)>=20))
    print('prepared',len(patches),'patches',len(per_event),'events',flush=True)


def pretrain(args):
    import torch
    import torch.nn.functional as F
    from scripts.train_unet import UNet,normalise,make_batch
    from scripts.hypothesis_lab import manifest
    from src.comp.hypothesis_models import masked_binary_loss
    out=Path(args.out); prepared=json.loads((out/'prepared-v2.json').read_text())
    if not prepared['eligible']: raise RuntimeError('fewer than 20 eligible events')
    result=out/'pretrain'; result.mkdir(exist_ok=True); write_json(result/'manifest.json',manifest(args))
    torch.manual_seed(20260918); torch.set_num_threads(4); rng=np.random.default_rng(20260918)
    xs=[];ys=[]
    for r in prepared['patches']:
        z=np.load(r['file']);xs.append(z['image']);ys.append(z['mask'].astype(np.int64))
    flat=np.concatenate([x[:,y!=255][:,::37].astype(np.float32) for x,y in zip(xs,ys)],axis=1); mean=flat.mean(1); std=np.maximum(flat.std(1),1e-3); del flat
    X=torch.stack([torch.from_numpy(((x.astype(np.float32)-mean[:,None,None])/std[:,None,None]).astype(np.float16)) for x in xs]).cuda();Y=torch.as_tensor(np.stack(ys),device='cuda')
    net=UNet(9,2,w=32,depth=7).cuda(); opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=50*(len(xs)//8)); scaler=torch.amp.GradScaler('cuda'); losses=[]
    for ep in range(50):
        net.train(); order=rng.permutation(len(xs)); batch_losses=[]
        for i in range(0,len(order)-7,8):
            x,y=make_batch(X,Y,torch.as_tensor(order[i:i+8],device='cuda'),rng,512); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                logits=net(x); loss=masked_binary_loss(logits,y)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();sched.step();batch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(batch_losses)));print('pretrain',ep+1,losses[-1],flush=True)
    torch.save(dict(encoder=net.down.state_dict(),mean=mean,std=std,events=prepared['events']),result/'encoder.pt');write_json(result/'summary.json',dict(loss=losses,events=prepared['events'],patches=len(xs),non_claims=['Training loss is not a held-out accuracy measurement.']))


def smoke(args):
    import torch
    from scripts.train_unet import UNet
    from src.comp.hypothesis_models import masked_binary_loss
    torch.set_num_threads(2)
    out=Path(args.out); prepared=json.loads((out/'prepared-v2.json').read_text())
    arrays=[np.load(r['file']) for r in prepared['patches'][:8]]
    x=torch.as_tensor(np.stack([z['image'].astype(np.float32) for z in arrays]))
    y=torch.as_tensor(np.stack([z['mask'].astype(np.int64) for z in arrays]))
    net=UNet(9,2,w=2,depth=3); loss=masked_binary_loss(net(x),y); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    result=dict(status='PASS',smoke=True,real_patches=len(arrays),ignored_pixels=int((y==255).sum()),loss_finite=True,non_claim='CPU forward/backward only; not model quality.')
    write_json(out/'smoke.json',result); print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['download','prepare','pretrain','smoke']);p.add_argument('--out',default='research/external-train-v1');p.add_argument('--events',type=int,default=30);args=p.parse_args();{'download':download,'prepare':prepare,'pretrain':pretrain,'smoke':smoke}[args.task](args)
if __name__=='__main__':main()
