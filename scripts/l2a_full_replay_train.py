"""SPEC-101: complete original supervision, matched source and mask schedules."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import experiment_fingerprint,sha256_file
from src.comp.accuracy_artifacts import execution_sources,load_checkpoint,save_checkpoint
from src.comp.forest_pair import make_forest_model,augment_forest,forest_loss
from src.comp.l2a_full_replay import epoch_view,weighted_order,matched_view,original_valid
from src.comp.l2a_finetuning import l2a_valid
from scripts.l2a_full_replay_io import load_manifest,files_for


def cache_samples(root,records,mean,std):
    images=[];targets=[];twins=[];common=[]
    for record in records:
        with rasterio.open(root/record["files"]["hls"]) as source:image=source.read().astype(np.float32)
        with rasterio.open(root/record["files"]["mask"]) as source:labels=source.read(1)
        valid=original_valid(image,labels)
        with rasterio.open(root/record["files"]["original_valid"]) as source:stored=source.read(1).astype(bool)
        if not np.array_equal(valid,stored) or int(valid.sum())!=record["original_pixels"]:
            raise ValueError("original supervision changed")
        if not valid.any():raise ValueError("original sample has no observable supervision")
        x,y,_=matched_view(image,labels,valid)
        x-=mean[:,None,None];x/=std[:,None,None]
        images.append(torch.from_numpy(x));targets.append(torch.from_numpy(y))
        twin=record["twin"]
        if twin is None:
            twins.append(None);common.append(None);continue
        with rasterio.open(root/twin["files"]["l2a"]) as source:l2a=source.read().astype(np.float32)
        with rasterio.open(root/twin["files"]["valid"]) as source:mask=source.read(1).astype(bool)
        with rasterio.open(root/twin["files"]["scl"]) as source:scl=source.read(1)
        if not np.array_equal(mask,l2a_valid(l2a,scl,labels)&valid):raise ValueError("common mask differs from strict physical validity")
        lx,_,mask=matched_view(l2a,labels,valid,mask)
        if int(mask.sum())!=twin["common_pixels"]:raise ValueError("common supervision changed")
        lx-=mean[:,None,None];lx/=std[:,None,None]
        twins.append(torch.from_numpy(lx));common.append(torch.from_numpy(mask))
    return images,targets,twins,common


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","manifest","checkpoint","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--seed",type=int,choices=(1,2,3),required=True)
    p.add_argument("--variant",choices=("control","replay"),required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required onk8plus")
    if args.out.exists():raise ValueError("immutable training output exists")
    manifest=load_manifest(args.data_root,args.manifest)
    initializer=next(r for r in manifest["initializers"] if r["seed"]==args.seed)
    if sha256_file(args.checkpoint)!=initializer["sha256"]:raise ValueError("wrong C1-MM initializer")
    bundle=load_checkpoint(args.checkpoint,initializer["sha256"],export_root=Path("research/tensor-exports"))
    if bundle.get("seed")!=args.seed or bundle.get("w")!=32 or bundle.get("cin",6)!=6:
        raise ValueError("wrong initializer architecture/seed")
    mean=np.asarray(bundle["mean"],np.float32);std=np.asarray(bundle["std"],np.float32)
    if mean.shape!=(6,) or std.shape!=(6,) or not np.isfinite([mean,std]).all() or (std<=0).any():
        raise ValueError("invalid initializer normalization")
    fit=[r for r in manifest["records"] if r["partition"]=="fit"]
    if args.smoke:
        fit=[r for r in fit if r["twin"] is not None][:2]+[r for r in fit if r["twin"] is None][:1]
    epochs,batch=(2,2) if args.smoke else (30,8)
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    net=make_forest_model("post",32,7);net.load_state_dict(bundle["state"]);del bundle
    net.cuda().train()
    sources=execution_sources("scripts/l2a_full_replay_train.py","scripts/l2a_full_replay_io.py",
        "src/comp/l2a_full_replay.py","src/comp/l2a_finetuning.py","scripts/queues/queue_l2a_full_replay.sh",tta=False)
    inputs=[args.data_root/path for record in manifest["records"] for _,path in files_for(record)]
    signature=experiment_fingerprint(manifest["records"],inputs,[args.checkpoint],sources,
        dict(variant=args.variant,seed=args.seed,epochs=epochs,batch=batch,smoke=args.smoke,
             normalization="initializer-fit-locked",schedule="odd-full/even-common/fallback-full"))
    config=dict(spec_id="SPEC-101",variant=args.variant,seed=args.seed,w=32,width=32,depth=7,cin=6,classes=2,
        epochs=epochs,batch=batch,lr=3e-5,mean=mean.tolist(),std=std.tolist(),smoke=args.smoke,
        data_manifest=sha256_file(args.manifest),fit_ids=[r["sample_id"] for r in fit],fingerprint=signature,
        initialization_sha256=initializer["sha256"],initialization_family="C1-MM-seeds1-3",
        initializer_pool=[r["sha256"] for r in manifest["initializers"]],
        source_schedule="odd-full/even-common/fallback-full",normalization_mode="initializer-fit-locked",
        source_hashes={str(path):sha256_file(path) for path in sources},
        software={"torch":str(torch.__version__),"numpy":str(np.__version__)})
    args.out.mkdir(parents=True)
    with (args.out/"data_manifest.json").open("x") as stream:json.dump(config,stream,indent=2)
    images,targets,twins,common=cache_samples(args.data_root,fit,mean,std)
    optimizer=torch.optim.AdamW(net.parameters(),lr=3e-5,weight_decay=1e-4)
    template_size=sum(r["sampling_weight"] for r in fit)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*math.ceil(template_size/batch))
    full_exposures=np.zeros(len(fit),int);common_exposures=full_exposures.copy();l2a_exposures=full_exposures.copy()
    order_hash=hashlib.sha256();losses=[];start=time.monotonic()
    fill=torch.as_tensor(-mean/std,device="cuda")[None,:,None,None]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        order=weighted_order(fit,rng);order_hash.update(order.astype("<i8").tobytes());values=[]
        for pos in range(0,len(order),batch):
            indices=order[pos:pos+batch];batch_images=[];batch_targets=[];batch_valid=[]
            for index in indices:
                source,mask=epoch_view(epoch,twins[index] is not None,args.variant)
                batch_images.append(twins[index] if source=="l2a" else images[index])
                batch_targets.append(targets[index])
                batch_valid.append(common[index] if mask=="common" else targets[index]>=0)
                if mask=="original":full_exposures[index]+=1
                else:common_exposures[index]+=1
                if source=="l2a":l2a_exposures[index]+=1
            x=torch.stack(batch_images).cuda();y=torch.stack(batch_targets).cuda().long();valid=torch.stack(batch_valid).cuda()
            x=torch.where(valid[:,None],x,fill);y[~valid]=-1
            x,y=augment_forest(x,y,rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):logits=net(x)
            loss=forest_loss(logits,y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):raise FloatingPointError("nonfinite training")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step();scheduler.step();values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
        print("epoch",epoch,"loss",losses[-1],"seconds",int(time.monotonic()-start),flush=True)
    weights=np.array([r["sampling_weight"] for r in fit])
    expected_full=np.array([(epochs//2 if r["twin"] is not None else epochs)*w for r,w in zip(fit,weights)])
    if not np.array_equal(full_exposures,expected_full):raise ValueError("original full supervision lost exposures")
    if not np.array_equal(full_exposures+common_exposures,epochs*weights):raise ValueError("original samples lost optimization visits")
    expected_l2a=common_exposures if args.variant=="replay" else np.zeros(len(fit),int)
    if not np.array_equal(l2a_exposures,expected_l2a):raise ValueError("actual L2A source exposures differ")
    save_checkpoint(args.out/"model.pt",net.cpu().state_dict(),config)
    summary=dict(config,completed=True,loss=losses,seconds=time.monotonic()-start,
        full_hls_exposures=full_exposures.tolist(),common_exposures=common_exposures.tolist(),
        l2a_exposures=l2a_exposures.tolist(),order_sha256=order_hash.hexdigest(),
        peak_vram_bytes=torch.cuda.max_memory_allocated())
    with (args.out/"summary.json").open("x") as stream:json.dump(summary,stream,indent=2)
    print("COMPLETE",args.variant,args.seed,flush=True)


if __name__=="__main__":main()
