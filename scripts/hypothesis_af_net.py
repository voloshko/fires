"""SPEC-27/36: small AF U-Net, nested calibration and fixed mixtures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import GRID as BOOST_GRID,binary_counts,binary_scores,digest,nested_folds,paired_interval,write_json
from scripts.hypothesis_lab import manifest

# Neural probabilities need not share the HGB calibration scale. Frozen before GPU trials.
GRID=np.unique(np.r_[[.01,.02,.05,.1,.2,.3,.4],BOOST_GRID])


def run(args):
    import torch
    import torch.nn.functional as F
    from src.comp.af import AfDataset,build_training_set,train,features,THRESHOLDS
    from scripts.train_unet import UNet,make_batch,normalise
    torch.set_num_threads(3)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); run_manifest=manifest(args); run_manifest['threshold_grid']=GRID.tolist(); run_manifest['mixture_grid']=[0.,.3,.4,.5,.6,.7,1.]; write_json(out/'manifest.json',run_manifest)
    d=AfDataset(args.data); sp=json.loads(Path(args.split).read_text()); ids=sorted(sp['train']+sp['val'])
    if set(ids)&set(sp['holdout']): raise ValueError('holdout overlap')
    device='cpu' if args.smoke else 'cuda'
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
    # Cache only development chips; no holdout/test labels read.
    raw={}; truth={}; eligible={}; hashes={}
    for c in ids:
        ch=d.load(c); raw[c]=np.nan_to_num(np.concatenate([ch.viirs,ch.aux]),posinf=0,neginf=0).astype(np.float32); truth[c]=ch.mask.astype(np.int64)
        eligible[c]=ch.valid()&~np.isin(ch.aux[0],THRESHOLDS['exclude_landcover'])
        hashes[c]={str(p):digest(p) for folder in ('viirs','aux','masks') for p in (Path(args.data)/folder).glob(c+'_*.tif')}
    write_json(out/'data_manifest.json',dict(files=hashes,split_sha256=digest(args.split)))
    weights=[0.,.3,.4,.5,.6,.7,1.]; results=[]; started=time.time()
    for k,(fit,cal,test) in enumerate(nested_folds(ids)):
        if args.smoke: fit,cal,test=fit[:8],cal[:2],test[:2]
        torch.manual_seed(args.seed+k); rng=np.random.default_rng(args.seed+k)
        mean,std=normalise([raw[c] for c in fit]); mt=torch.as_tensor(mean,device=device)[None,:,None,None]; st=torch.as_tensor(std,device=device)[None,:,None,None]
        X=torch.stack([torch.from_numpy(((raw[c].astype(np.float32)-mean[:,None,None])/std[:,None,None]).astype(np.float16)) for c in fit]).to(device)
        Y=torch.as_tensor(np.stack([truth[c] for c in fit]),device=device)
        net=UNet(13,2,w=args.width,depth=args.depth).to(device); opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4)
        steps=args.epochs*(len(fit)//8); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=steps); scaler=torch.amp.GradScaler(device,enabled=device=='cuda')
        class_weight=torch.tensor([1.,200.],device=device)
        for ep in range(args.epochs):
            net.train(); order=rng.permutation(len(fit)); losses=[]
            for i in range(0,len(order)-7,8):
                x,y=make_batch(X,Y,torch.as_tensor(order[i:i+8],device=device),rng,256); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,enabled=device=='cuda'):
                    logits=net(x); p=logits.float().softmax(1)[:,1]; t=(y>0).float()
                    loss=F.cross_entropy(logits,y,weight=class_weight)+1-(2*(p*t).sum()+1)/(p.sum()+t.sum()+1)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach()))
            if (ep+1)%10==0 or args.smoke: print('fold',k,'epoch',ep+1,'loss',np.mean(losses),'seconds',int(time.time()-started),flush=True)
        torch.save(dict(state=net.state_dict(),mean=mean,std=std,width=args.width,depth=args.depth,fit=fit,seed=args.seed+k),out/f'net-fold-{k}.pt')
        del X,Y; torch.cuda.empty_cache(); net.eval()
        bx,by=build_training_set(d,fit,args.seed); hgb=train(bx,by,args.seed); del bx,by
        all_counts={}
        with torch.no_grad():
            for section,selected in [('calibration',cal),('evaluation',test)]:
                rows=[]
                for c in selected:
                    ch=d.load(c); x=(torch.from_numpy(raw[c].astype(np.float32))[None].to(device)-mt)/st
                    with torch.autocast(device_type=device,enabled=device=='cuda'): logits=net(x).float()
                    pn=logits.softmax(1)[0,1].cpu().numpy()
                    pb=hgb.predict_proba(features(ch).reshape(16,-1).T)[:,1].reshape(ch.shape)
                    pn[~eligible[c]]=0; pb[~eligible[c]]=0
                    if section=='evaluation': np.savez_compressed(out/f'prob-{c}.npz',network=pn,boost=pb.astype(np.float32))
                    rows.append([[binary_counts(truth[c],w*pn+(1-w)*pb>=cut) for cut in GRID] for w in weights])
                all_counts[section]=np.asarray(rows)
        cc,ee=all_counts['calibration'],all_counts['evaluation']; row=dict(fold=k,fit=fit,calibration=cal,evaluation=test)
        for name,w_indices in [('boost',[0]),('network',[6]),('mixture',list(range(1,6)))]:
            candidates=[(binary_scores(cc[:,w,j])['f1'],w,j) for w in w_indices for j in range(len(GRID))]
            # Stable fixed order resolves ties without external evaluation.
            _,wi,ti=max(candidates,key=lambda v:v[0]); cnt=ee[:,wi,ti]
            row[name]=dict(weight=weights[wi],cutoff=float(GRID[ti]),counts=cnt.tolist(),metrics=binary_scores(cnt))
            print(k,name,row[name],flush=True)
        write_json(out/f'fold-{k}.json',row); results.append(row)
        del net,hgb; torch.cuda.empty_cache()
        if args.smoke: break
    counts={name:np.concatenate([np.asarray(r[name]['counts']) for r in results]) for name in ('boost','network','mixture')}
    summary={name:binary_scores(c) for name,c in counts.items()}
    for name in ('network','mixture'):
        summary[name]['delta']=summary[name]['f1']-summary['boost']['f1']; summary[name]['ci95_chip_bootstrap']=paired_interval(counts['boost'],counts[name]); summary[name]['accepted']=summary[name]['delta']>=.005 and summary[name]['ci95_chip_bootstrap'][0]>0
    summary.update(seconds=time.time()-started,threshold_grid=GRID.tolist(),smoke=args.smoke,non_claims=['No hidden test evaluation.','Chip grouping cannot guarantee independent fire events.','No final420 model or submission produced by this screening run.'])
    write_json(out/'summary.json',summary); print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/comp/train/af'); p.add_argument('--split',default='data/comp/split_af.json'); p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=20260918); p.add_argument('--epochs',type=int,default=100); p.add_argument('--width',type=int,default=16); p.add_argument('--depth',type=int,default=5); p.add_argument('--smoke',action='store_true'); run(p.parse_args())
if __name__=='__main__': main()
