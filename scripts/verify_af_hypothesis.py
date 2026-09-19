"""Replay stored AF predictions against original masks, independent count implementation."""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio


def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();root=Path(args.run); totals={name:np.zeros(3,np.int64) for name in ('random','hard')}; chips=[]
    for fold in sorted(root.glob('fold-*.json')):
        r=json.loads(fold.read_text()); assert not(set(r['fit'])&set(r['evaluation']) or set(r['calibration'])&set(r['evaluation']))
        for i,c in enumerate(r['evaluation']):
            with rasterio.open(f'data/comp/train/af/masks/{c}_MASK.tif') as ds: truth=ds.read(1).reshape(-1)>0
            for name in totals:
                prediction=np.load(root/f'{name}_probabilities'/f'{c}.npz')['p']>=r[name]['cutoff']
                tp=np.sum(prediction[truth]); fp=np.sum(prediction[~truth]); fn=np.sum(~prediction[truth]); counts=np.array([tp,fp,fn])
                if not np.array_equal(counts,r[name]['counts'][i]): raise AssertionError((fold.name,c,name,counts,r[name]['counts'][i]))
                totals[name]+=counts
            chips.append(c)
    assert len(chips)==len(set(chips))==336
    summary=json.loads((root/'summary.json').read_text())
    for name,c in totals.items():
        f1=2*c[0]/(2*c[0]+c[1]+c[2]); assert abs(f1-summary[name]['f1'])<1e-12
    print(json.dumps(dict(status='PASS',chips=len(chips),totals={k:v.tolist() for k,v in totals.items()},non_claim='Replays stored probabilities; does not independently retrain models.'),indent=2))
if __name__=='__main__':main()
