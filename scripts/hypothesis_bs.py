"""Paired BS screening, fixed split/steps/postprocessing (SPEC-32/36)."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import bs_inputs,bs_scores,confusion,digest,write_json
from scripts.hypothesis_lab import manifest


def data_split(args):
    from scripts.train_unet import load_split
    # --final: все 224 чипа, настроечной части нет, замер не производится (SPEC-32 → сабмит).
    d,fit,tune=load_split(args.data,args.split,use_all=getattr(args,'final',False))
    if args.fold is not None:
        from src.comp.hypothesis_lab import bs_confirmation_split
        fit,tune=bs_confirmation_split(d.meta,fit,tune,args.fold)
    return d,fit,tune


def data_manifest(args,fit,tune):
    root=Path(args.data); hashes={}
    for folder in ('sentinel2_pre','sentinel2_post','aux','sentinel1_pre','masks'):
        for c in fit+tune:
            for p in sorted((root/folder).glob(c+'_*.tif')): hashes[str(p)]=digest(p)
    return dict(fit=fit,evaluation=tune,files=hashes,split_sha256=digest(args.split),meta_sha256=digest(root/'meta.csv'))


def boost(args):
    from src.comp.model import build_training_set,train,save
    from src.comp.features import stack
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'manifest.json',manifest(args)); d,fit,tune=data_split(args)
    write_json(out/'data_manifest.json',data_manifest(args,fit,tune))
    x,y=build_training_set(d,fit); model=train(x,y); save(model,out/'boost.pkl')
    probabilities=[]
    for c in tune:
        chip=d.load(c); f=np.nan_to_num(stack(chip),posinf=0,neginf=0)
        probabilities.append(model.predict_proba(f.reshape(len(f),-1).T).reshape(*chip.shape,4).astype(np.float16))
    np.save(out/'probabilities.npy',np.stack(probabilities)); print('boost ready',len(fit),len(tune),flush=True)


def evaluate(net,mean,std,d,tune,variant,out,boost_dir=None,device='cuda',precision='fp16'):
    import torch
    from src.comp.postproc import drop_far
    pb=None
    if boost_dir:
        bm=json.loads((Path(boost_dir)/'data_manifest.json').read_text())
        if bm['evaluation']!=tune or set(bm['fit'])&set(tune): raise ValueError('boost split mismatch')
        # Bind the shared boost cache to the same source pixels, not merely IDs.
        for filename,h in bm['files'].items():
            if digest(filename)!=h: raise ValueError('boost source data changed')
        pb=np.load(Path(boost_dir)/'probabilities.npy',mmap_mode='r')
    counts={k:[] for k in ('network','network_clear','mixed','mixed_clear')}; pn=[]; records=[]
    mean_t=torch.as_tensor(mean,device=device)[None,:,None,None]; std_t=torch.as_tensor(std,device=device)[None,:,None,None]
    net.eval()
    with torch.no_grad():
        for i,c in enumerate(tune):
            chip=d.load(c); f=np.nan_to_num(bs_inputs(chip,variant),posinf=0,neginf=0)
            x=(torch.from_numpy(f)[None].to(device)-mean_t)/std_t
            with torch.autocast(device_type=device,enabled=device=='cuda' and precision!='fp32',dtype=torch.bfloat16 if precision=='bf16' else torch.float16):
                logits=net(x).float()
                for dims in ([2],[3],[2,3]): logits+=torch.flip(net(torch.flip(x,dims)).float(),dims)
            if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite evaluation logits')
            p=(logits/4).softmax(1)[0].permute(1,2,0).cpu().numpy(); pn.append(p.astype(np.float16))
            rec={'chip':c}
            for name,mix in [('network',p)]+([] if pb is None else [('mixed',.6*p+.4*pb[i].astype(np.float32))]):
                pred=mix.argmax(2).astype(np.uint8); blind=~chip.valid()
                pred[blind]=np.where(p.argmax(2)>0,mix[...,1:].argmax(2)+1,0)[blind]
                pred[chip.label_zero()]=0; pred=drop_far(pred,125)
                cm=confusion(chip.mask,pred); clear=confusion(chip.mask,pred,valid=chip.valid())
                counts[name].append(cm); counts[name+'_clear'].append(clear); rec[name]=cm.tolist(); rec[name+'_clear']=clear.tolist()
            records.append(rec)
    np.save(out/'probabilities.npy',np.stack(pn))
    summary={name:bs_scores(np.sum(cms,axis=0)) for name,cms in counts.items() if cms}
    write_json(out/'per_chip.json',records); return summary


def run(args):
    import torch
    import torch.nn.functional as F
    from src.comp.hypothesis_models import make_model
    from scripts.train_unet import normalise,make_batch
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    write_json(out/'manifest.json',manifest(args)); d,fit,tune=data_split(args)
    write_json(out/'data_manifest.json',data_manifest(args,fit,tune))
    if getattr(args,'final',False): args.boost=None
    if not args.smoke and args.boost:
        bm=json.loads((Path(args.boost)/'data_manifest.json').read_text())
        if bm['fit']!=fit or bm['evaluation']!=tune: raise ValueError('boost training/evaluation split mismatch')
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device='cpu' if args.smoke else ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.precision=='bf16' and device=='cuda' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported')
    if device=='cpu' and not args.smoke: raise RuntimeError('GPU required for full experiment')
    if args.smoke: fit,tune=fit[:8],tune[:2]
    xs=[]; ys=[]
    for c in fit:
        ch=d.load(c); xs.append(np.nan_to_num(bs_inputs(ch,args.variant),posinf=0,neginf=0).astype(np.float16)); ys.append(ch.mask.astype(np.int64))
    mean,std=normalise(xs)
    # Shared statistics prevent artificial pre/post differences in shared encoder.
    if args.variant=='siam':
        for b in range(9):
            flat=np.concatenate([a[[b,b+9]].reshape(2,-1)[:,::37].astype(np.float32).ravel() for a in xs])
            mean[b]=mean[b+9]=flat.mean(); std[b]=std[b+9]=max(flat.std(),1e-3)
    extra_map={}
    if args.extra:
        from dataclasses import replace
        extra_root=Path(args.extra); qc=json.loads((extra_root/'result.json').read_text())
        accepted={r['chip']:r for r in qc['records'] if r['accepted']}
        if len(accepted)<args.min_extra: raise ValueError('insufficient accepted temporal chips for this experiment')
        if args.fold is None and not set(accepted)<=set(fit): raise ValueError('extra scenes outside fit')
        for i,c in enumerate(fit):
            if c in accepted:
                path=extra_root/f'{c}.npz'
                if digest(path)!=accepted[c]['extra_sha256']: raise ValueError('extra hash mismatch')
                ch=replace(d.load(c),pre=np.load(path)['pre']); extra_map[i]=len(xs)
                xs.append(np.nan_to_num(bs_inputs(ch,args.variant),posinf=0,neginf=0).astype(np.float16)); ys.append(ch.mask.astype(np.int64))
        write_json(out/'extra_manifest.json',qc)
    if args.fade:
        # SPEC-41: выцветание — сцена «после» сдвинута к «до», SCL и метка прежние.
        from dataclasses import replace
        frng=np.random.default_rng(args.seed+1); alphas={}
        for i,c in enumerate(fit):
            ch=d.load(c); a=float(frng.uniform(args.fade[0],args.fade[1])); alphas[c]=a
            post=ch.post.copy(); post[:9]=ch.pre[:9]+a*(ch.post[:9]-ch.pre[:9])
            ch=replace(ch,post=post); extra_map[i]=len(xs)
            xs.append(np.nan_to_num(bs_inputs(ch,args.variant),posinf=0,neginf=0).astype(np.float16)); ys.append(ch.mask.astype(np.int64))
        write_json(out/'fade_manifest.json',dict(range=args.fade,alphas=alphas))
    # Normalize chip-wise to avoid an N*C*H*W float32 transient on the GPU.
    X=torch.stack([torch.from_numpy(((a.astype(np.float32)-mean[:,None,None])/std[:,None,None]).astype(np.float16)) for a in xs]).to(device)
    Y=torch.as_tensor(np.stack(ys),device=device); del xs,ys
    if not torch.isfinite(X).all(): raise FloatingPointError('nonfinite normalized inputs')
    net=make_model(args.variant,args.width,args.depth).to(device)
    if args.encoder:
        if args.variant!='siam': raise ValueError('external encoder requires siam variant')
        encoder=torch.load(args.encoder,map_location=device,weights_only=False)
        net.down.load_state_dict(encoder['encoder'])
        write_json(out/'encoder_manifest.json',dict(file=args.encoder,sha256=digest(args.encoder),events=encoder['events']))
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
    steps=args.epochs*(len(fit)//args.batch)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,total_steps=steps)
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda' and args.precision=='fp16'); rng=np.random.default_rng(args.seed)
    weights=torch.tensor([.25,1.,1.,1.],device=device); t0=time.time(); losses=[]
    hooks=[]; numerical_failure={}
    if args.debug_numerics:
        def check_output(name):
            def hook(module,inputs,output):
                if not torch.isfinite(output).all():
                    value=float(inputs[0].detach().abs().max())
                    numerical_failure.update(module=name,type=type(module).__name__,input_dtype=str(inputs[0].dtype),output_dtype=str(output.dtype),input_finite=bool(torch.isfinite(inputs[0]).all()),input_max=value if np.isfinite(value) else str(value),nonfinite_outputs=int((~torch.isfinite(output)).sum()),parameters_finite=all(bool(torch.isfinite(p).all()) for p in module.parameters(recurse=False)))
                    raise FloatingPointError('nonfinite module output: '+name)
            return hook
        for name,module in net.named_modules():
            if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.BatchNorm2d)):
                hooks.append(module.register_forward_hook(check_output(name)))
    for epoch in range(args.epochs):
        net.train(); order=rng.permutation(len(fit)); epoch_losses=[]
        for k in range(0,len(order)-args.batch+1,args.batch):
            batch_ids=order[k:k+args.batch].copy()
            for bi,index in enumerate(batch_ids):
                replace_input=rng.random()<.5
                if int(index) in extra_map and replace_input: batch_ids[bi]=extra_map[int(index)]
            x,y=make_batch(X,Y,torch.as_tensor(batch_ids,device=device),rng,X.shape[-1])
            opt.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device,enabled=device=='cuda' and args.precision!='fp32',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16):
                    logits=net(x); p=1-logits.softmax(1)[:,0]; truth=(y>0).float()
                    loss=F.cross_entropy(logits,y,weight=weights)+1-(2*(p*truth).sum()+1)/(p.sum()+truth.sum()+1)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            except FloatingPointError:
                numerical_failure.update(epoch=epoch+1,batch=k//args.batch,batch_indices=[int(i) for i in batch_ids],batch_ids=[fit[int(i)] if int(i)<len(fit) else 'extra' for i in batch_ids],lr=float(opt.param_groups[0]['lr']))
                write_json(out/'numerical_failure.json',numerical_failure)
                # Retain the failed batch without requiring expensive module hooks.
                torch.save(dict(state=net.state_dict(),x=x.detach().cpu(),y=y.detach().cpu(),variant=args.variant,width=args.width,depth=args.depth,fusion_norm=args.variant=='siam',precision=args.precision),out/'debug_state.pt')
                raise
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        if (epoch+1)%10==0 or args.smoke: print(args.variant,args.seed,'epoch',epoch+1,'loss',losses[-1],'seconds',int(time.time()-t0),flush=True)
    bundle=dict(state=net.state_dict(),variant=args.variant,width=args.width,depth=args.depth,fusion_norm=args.variant=='siam',precision=args.precision,mean=mean,std=std,seed=args.seed,epochs=args.epochs,fit=fit)
    torch.save(bundle,out/'model.pt'); del X,Y; torch.cuda.empty_cache()
    if getattr(args,'final',False):
        write_json(out/'summary.json',dict(final=True,chips=len(fit),seconds=time.time()-t0,loss=losses,quality_evaluated=False)); print('final saved',len(fit),flush=True); return
    summary=evaluate(net,mean,std,d,tune,args.variant,out,None if args.smoke else args.boost,device,args.precision)
    summary.update(seconds=time.time()-t0,loss=losses,gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,screening_only=args.fold is None,smoke=args.smoke)
    write_json(out/'summary.json',summary); print(json.dumps({k:v for k,v in summary.items() if k!='loss'},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['boost','train']); p.add_argument('--data',default='data/comp/train/bs'); p.add_argument('--split',default='data/comp/split_bs.json'); p.add_argument('--out',required=True); p.add_argument('--variant',choices=['optical','raw','siam'],default='optical'); p.add_argument('--seed',type=int,default=20260918); p.add_argument('--epochs',type=int,default=200); p.add_argument('--width',type=int,default=32); p.add_argument('--depth',type=int,default=7); p.add_argument('--batch',type=int,default=8); p.add_argument('--boost',default='research/bs-boost-v1'); p.add_argument('--min-extra',type=int,default=4); p.add_argument('--fold',type=int); p.add_argument('--extra'); p.add_argument('--fade',nargs=2,type=float,help='SPEC-41: диапазон α выцветания сцены после'); p.add_argument('--encoder'); p.add_argument('--precision',choices=['fp16','bf16','fp32'],default='fp16'); p.add_argument('--debug-numerics',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--final',action='store_true',help='обучение на всех чипах без замера'); args=p.parse_args()
    (boost if args.task=='boost' else run)(args)
if __name__=='__main__': main()
