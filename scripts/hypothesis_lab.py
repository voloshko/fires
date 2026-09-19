#!/usr/bin/env python3
"""SPEC-32..36 research CLI. Does not create or modify submissions."""
from __future__ import annotations
import argparse
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import (GRID,binary_counts,binary_scores,choose_background,digest,nested_folds,paired_interval,write_json)
ROOT=Path(__file__).resolve().parents[1]


def manifest(args):
    versions={}
    for package in ('numpy','scipy','scikit-learn','torch','rasterio','pandas','planetary-computer'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='unavailable'
    return dict(software=versions,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=sys.argv,
                python=platform.python_version(),numpy=np.__version__,config={k:v for k,v in vars(args).items() if not callable(v)},
                git_revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip() or ((ROOT/'.source-revision').read_text().strip() if (ROOT/'.source-revision').exists() else 'unknown'),
                source_sha256={str(p.relative_to(ROOT)):digest(p) for directory in ('src','scripts') for p in sorted((ROOT/directory).rglob('*.py'))})


def af_cache(root,out,ids):
    from src.comp.af import AfDataset,features,THRESHOLDS
    d=AfDataset(root); cache=out/'cache'; cache.mkdir(parents=True,exist_ok=True); hashes={}
    for i,c in enumerate(ids):
        paths=[Path(root)/'viirs'/f'{c}_VIIRS_I1-I5.tif',Path(root)/'aux'/f'{c}_AUX.tif',Path(root)/'masks'/f'{c}_MASK.tif']
        hashes[c]={str(p):digest(p) for p in paths}
        p=cache/f'{c}.npz'
        if not p.exists():
            chip=d.load(c); x=features(chip).reshape(16,-1).T
            eligible=chip.valid().ravel()&~np.isin(chip.aux[0].ravel(),THRESHOLDS['exclude_landcover'])
            np.savez(p,x=x,y=chip.mask.ravel()>0,valid=chip.valid().ravel(),eligible=eligible)
        if i%50==0: print('cache',i,len(ids),flush=True)
    return cache,hashes


def af_set(cache,ids,seed,miner=None):
    xs,ys=[],[]; rng=np.random.default_rng(seed)
    for c in ids:
        z=np.load(cache/f'{c}.npz'); x,y,ok=z['x'],z['y'],z['valid']
        fire=np.flatnonzero(y&ok); back=np.flatnonzero(~y&ok)
        scores=None if miner is None else miner.predict_proba(x)[:,1]
        take=np.r_[fire,choose_background(back,rng,scores)]
        xs.append(x[take]); ys.append(y[take].astype(np.int8))
    return np.concatenate(xs),np.concatenate(ys)


def af_evaluate(model,cache,ids,output=None):
    rows=[]
    if output: output.mkdir(parents=True,exist_ok=True)
    for c in ids:
        z=np.load(cache/f'{c}.npz'); p=model.predict_proba(z['x'])[:,1]; p[~z['eligible']]=0
        rows.append([binary_counts(z['y'],p>=cut) for cut in GRID])
        if output: np.savez_compressed(output/f'{c}.npz',p=p.astype(np.float32))
    return np.asarray(rows)


def af_hard(args):
    from src.comp.af import train
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise FileExistsError('use fresh output directory')
    sp=json.loads(Path(args.split).read_text()); ids=sorted(sp['train']+sp['val'])
    if set(ids)&set(sp['holdout']): raise ValueError('holdout overlap')
    m=manifest(args); m['split_sha256']=digest(args.split); write_json(out/'manifest.json',m)
    cache,hashes=af_cache(args.data,out,ids); write_json(out/'data_manifest.json',hashes)
    results=[]; started=time.time()
    for k,(fit,cal,test) in enumerate(nested_folds(ids)):
        print('fold',k,'fit/cal/test',len(fit),len(cal),len(test),flush=True)
        x,y=af_set(cache,fit,args.seed); base=train(x,y,args.seed)
        xh,yh=af_set(cache,fit,args.seed,base); hard=train(xh,yh,args.seed)
        row=dict(fold=k,fit=fit,calibration=cal,evaluation=test)
        for name,model in [('random',base),('hard',hard)]:
            calibration=af_evaluate(model,cache,cal)
            scores=[binary_scores(calibration[:,j])['f1'] for j in range(len(GRID))]
            ix=int(np.argmax(scores)); counts=af_evaluate(model,cache,test,out/f'{name}_probabilities')
            row[name]=dict(cutoff=float(GRID[ix]),calibration_f1=scores[ix],counts=counts[:,ix].tolist(),grid_counts=counts.tolist(),metrics=binary_scores(counts[:,ix]))
            print(k,name,row[name]['cutoff'],row[name]['metrics'],flush=True)
        write_json(out/f'fold-{k}.json',row); results.append(row)
    a=np.concatenate([np.asarray(r['random']['counts']) for r in results]); b=np.concatenate([np.asarray(r['hard']['counts']) for r in results])
    interval=paired_interval(a,b); delta=binary_scores(b)['f1']-binary_scores(a)['f1']
    summary=dict(random=binary_scores(a),hard=binary_scores(b),delta=delta,ci95_chip_bootstrap=interval,
                 accepted=bool(delta>=.005 and interval[0]>0),seconds=time.time()-started,grid=GRID.tolist(),
                 non_claims=['Chip folds do not guarantee event independence.','Not competition hidden-test quality.','OOF pooled threshold is diagnostic, not nested evaluation.'])
    for name in ('random','hard'):
        grid=np.concatenate([np.asarray(r[name]['grid_counts']) for r in results]); scores=[binary_scores(grid[:,j])['f1'] for j in range(len(GRID))]; ix=int(np.argmax(scores))
        summary[name+'_optimistic_oof']=dict(cutoff=float(GRID[ix]),f1=scores[ix])
    write_json(out/'summary.json',summary); print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='task',required=True)
    af=sub.add_parser('af-hard'); af.add_argument('--data',default='data/comp/train/af'); af.add_argument('--split',default='data/comp/split_af.json'); af.add_argument('--out',required=True); af.add_argument('--seed',type=int,default=20260918); af.set_defaults(func=af_hard)
    args=p.parse_args(); args.func(args)
if __name__=='__main__': main()
