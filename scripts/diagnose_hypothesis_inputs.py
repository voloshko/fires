"""Read-only input diagnostics for interpreting SPEC-32/34/35 pilots."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import write_json


def main():
    from src.comp.chips import BsDataset,SCL_INVALID
    from src.comp.hypothesis_models import make_model
    prepared=json.loads(Path('research/external-train-v1/prepared-v2.json').read_text())
    valid=[]
    for row in prepared['patches']:
        z=np.load(row['file']); valid.append(float(np.mean(z['mask']!=255)))
    d=BsDataset('data/comp/train/bs'); temporal=[]
    qc=json.loads(Path('research/temporal-pre-v1/result.json').read_text())
    for row in qc['records']:
        if not row['accepted']: continue
        c=d.load(row['chip']); extra=np.load(f'research/temporal-pre-v1/{row["chip"]}.npz')['pre']
        ok=(c.mask>0)&~np.isin(c.pre[9],SCL_INVALID)&~np.isin(extra[9],SCL_INVALID)
        def nbr(a):
            x,y=a[6].astype(float),a[8].astype(float); return (x-y)/np.maximum(x+y,1)
        delta=np.abs(nbr(c.pre)[ok]-nbr(extra)[ok]); temporal.append(dict(chip=row['chip'],compared_burn_fraction=float(ok.sum()/max(1,np.count_nonzero(c.mask))),median=float(np.median(delta)),p90=float(np.quantile(delta,.9)),p95=float(np.quantile(delta,.95)),fraction_above_01=float(np.mean(delta>.1))))
    parameters={}
    for variant in ('optical','raw','siam'):
        net=make_model(variant);parameters[variant]=sum(p.numel() for p in net.parameters());del net
    result=dict(external_valid_fraction=dict(min=min(valid),p10=float(np.quantile(valid,.1)),median=float(np.median(valid)),max=max(valid)),temporal=temporal,parameters=parameters,non_claims=['NBR median stability is not pixelwise label invariance.','Masked padding still influences convolution and batch-normalization activations.'])
    write_json('research/input-diagnostics-v1.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
