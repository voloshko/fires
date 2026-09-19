"""Fixed two-seed and v13-reference comparisons; no ensemble weight search."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import bs_prediction,bs_scores,confusion,digest,write_json,verify_bs_probability_cache


def main():
    from scripts.train_unet import load_split
    p=argparse.ArgumentParser();p.add_argument('--root',default='research');p.add_argument('--out',required=True);args=p.parse_args();root=Path(args.root)
    d,fit,tune=load_split('data/comp/train/bs','data/comp/split_bs.json'); chips=[d.load(c) for c in tune]
    hashes=verify_bs_probability_cache(root/'bs-boost-v1',fit,tune)
    pb=np.load(root/'bs-boost-v1/probabilities.npy').astype(np.float32)
    tags=['d7opt','d7opt_s1','d7opt_s2','d7optjit','d7optjit_s1']; reference=[]
    for tag in tags:
        path=Path('baseline_models')/f'exp_{tag}.tune.npy'
        reference.append(np.load(path).astype(np.float32));hashes[str(path)]=digest(path)
    reference=np.mean(reference,axis=0)
    def score(prob):
        if prob.shape!=pb.shape or len(pb)!=len(chips): raise ValueError('probability cache shape mismatch')
        if not np.isfinite(prob).all() or not np.isfinite(pb).all(): raise ValueError('nonfinite probability')
        matrices=[]; clear=[]
        for c,boost,pred in zip(chips,pb,prob):
            out=bs_prediction(pred,boost,c);matrices.append(confusion(c.mask,out));clear.append(confusion(c.mask,out,valid=c.valid()))
        return dict(all_pixels=bs_scores(np.sum(matrices,axis=0)),clear_sky=bs_scores(np.sum(clear,axis=0)),per_chip=[m.tolist() for m in matrices])
    results={'v13_reference':score(reference)}
    for variant in ('optical','raw','siam','temporal','external'):
        runs=[root/f'bs-{variant}-{s}-v1' for s in (20260918,20260919)]
        if not all((r/'summary.json').exists() for r in runs):continue
        probs=[];singles=[]
        for run in runs:
            data=json.loads((run/'data_manifest.json').read_text())
            if data['fit']!=fit or data['evaluation']!=tune:raise ValueError('screening split mismatch')
            probs.append(np.load(run/'probabilities.npy').astype(np.float32));singles.append(json.loads((run/'summary.json').read_text())['mixed']['weighted'])
            hashes[str(run/'probabilities.npy')]=digest(run/'probabilities.npy')
        paired=np.mean(probs,axis=0)
        results[variant]=dict(mean_single_seed_weighted=float(np.mean(singles)),two_seed_ensemble=score(paired),v13_plus_two=score((5*reference+2*paired)/7))
    if 'optical' in results:
        for variant in ('raw','siam','temporal','external'):
            if variant in results:
                control='siam' if variant=='external' else 'optical'
                delta=results[variant]['mean_single_seed_weighted']-results[control]['mean_single_seed_weighted']
                results[variant].update(paired_control=control,delta_to_control=delta,screening_pass=delta>=.005)
    write_json(args.out,dict(results=results,probability_sha256=hashes,fit=fit,evaluation=tune,non_claims=['Historical 35-chip development screening, not an independent test.','v13 caches are historical 144-chip analogs, not scores of final224 deployed weights.','Fixed equal-weight seven-model reference blend; no weight search.']))
    for name,row in results.items(): print(name,{k:v for k,v in row.items() if k not in ('per_chip','two_seed_ensemble','v13_plus_two')})
if __name__=='__main__':main()
