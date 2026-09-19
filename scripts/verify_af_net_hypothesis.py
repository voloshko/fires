"""SPEC-27: cached prediction replay and explicitly optimistic pooled OOF diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import rasterio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import write_json


def counts_for_thresholds(truth,prob,grid):
    t=np.asarray(truth,dtype=bool).ravel(); p=np.asarray(prob).ravel()
    positive=np.sort(p[t]);negative=np.sort(p[~t])
    tp=len(positive)-np.searchsorted(positive,grid,side='left')
    fp=len(negative)-np.searchsorted(negative,grid,side='left')
    return np.stack([tp,fp,len(positive)-tp],axis=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();root=Path(args.run)
    summary=json.loads((root/'summary.json').read_text()); grid=np.asarray(summary['threshold_grid']); weights=[0.,.3,.4,.5,.6,.7,1.]
    total=np.zeros((len(weights),len(grid),3),np.int64);mismatches=[];seen=[]
    for path in sorted(root.glob('fold-*.json')):
        fold=json.loads(path.read_text())
        if set(fold['evaluation'])&(set(fold['fit'])|set(fold['calibration'])):raise ValueError('split overlap')
        for i,c in enumerate(fold['evaluation']):
            z=np.load(root/f'prob-{c}.npz')
            with rasterio.open(f'data/comp/train/af/masks/{c}_MASK.tif') as ds: truth=ds.read(1)>0
            counts=np.asarray([counts_for_thresholds(truth,w*z['network']+(1-w)*z['boost'],grid) for w in weights]);total+=counts;seen.append(c)
            for name in ('boost','network','mixture'):
                r=fold[name]; w=weights.index(r['weight']); g=int(np.argmin(np.abs(grid-r['cutoff'])))
                if not np.array_equal(counts[w,g],r['counts'][i]):mismatches.append(dict(chip=c,component=name,stored=r['counts'][i],replayed=counts[w,g].tolist()))
    if len(seen)!=336 or len(set(seen))!=336:raise ValueError('incomplete OOF coverage')
    scores=2*total[:,:,0]/np.maximum(1,2*total[:,:,0]+total[:,:,1]+total[:,:,2]);optimistic={}
    for name,indices in [('boost',[0]),('network',[6]),('mixture',list(range(1,6)))]:
        choices=[(float(scores[w,g]),w,g) for w in indices for g in range(len(grid))];score,w,g=max(choices,key=lambda a:a[0]);optimistic[name]=dict(f1=score,weight=weights[w],cutoff=float(grid[g]))
    result=dict(status='PASS' if not mismatches else 'FAIL',chips=len(seen),mismatches=mismatches,optimistic_pooled_oof=optimistic,non_claims=['Pooled OOF grid maxima fit the same labels and are not nested evaluation.','Stored probabilities are float32; decision-boundary rounding can affect replay.','Does not independently retrain models.'])
    write_json(root/'diagnostics.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
