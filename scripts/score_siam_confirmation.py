"""SPEC-36 development revalidation; paired event bootstrap, no hidden-test claim."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import bs_scores,bs_confirmation_split,write_json,digest


def main():
    from scripts.train_unet import load_split
    p=argparse.ArgumentParser();p.add_argument('--root',default='research');p.add_argument('--out',required=True);args=p.parse_args();root=Path(args.root)
    d,development,selection=load_split('data/comp/train/bs','data/comp/split_bs.json');mapping=d.meta.set_index('chip_id').fire_event_id.to_dict()
    counts={'optical':{},'siam':{}};folds=[];hashes={};reference_sources=None
    relevant=['scripts/hypothesis_bs.py','scripts/train_unet.py','src/comp/hypothesis_models.py','src/comp/features.py','src/comp/postproc.py','src/comp/hypothesis_lab.py']
    for fold in range(5):
        fit,evaluation=bs_confirmation_split(d.meta,development,selection,fold);row={'fold':fold,'evaluation':evaluation}
        for variant in counts:
            run=root/f'bs-confirm-{variant}-f{fold}-v1'
            m=json.loads((run/'manifest.json').read_text());dm=json.loads((run/'data_manifest.json').read_text());summary=json.loads((run/'summary.json').read_text())
            cfg=m['config']
            if cfg['seed']!=20260920+fold or cfg['fold']!=fold or cfg['precision']!='bf16' or cfg['variant']!=variant or summary['smoke']:raise ValueError('confirmation configuration drift')
            if dm['fit']!=fit or dm['evaluation']!=evaluation:raise ValueError('confirmation split drift')
            sources={k:m['source_sha256'][k] for k in relevant}
            if reference_sources is None:reference_sources=sources
            if sources!=reference_sources:raise ValueError('training source changed across confirmation runs')
            per=json.loads((run/'per_chip.json').read_text())
            if [r['chip'] for r in per]!=evaluation:raise ValueError('prediction order mismatch')
            for r in per:
                if r['chip'] in counts[variant]:raise ValueError('duplicate evaluated chip')
                cm=np.asarray(r['mixed'],dtype=np.int64)
                if cm.shape!=(4,4) or cm.sum()!=512*512 or (cm<0).any():raise ValueError('invalid confusion counts')
                counts[variant][r['chip']]=cm
            row[variant]=bs_scores(np.sum([counts[variant][c] for c in evaluation],axis=0))
            for filename in ('manifest.json','data_manifest.json','per_chip.json','summary.json'):hashes[str(run/filename)]=digest(run/filename)
        row['delta']=row['siam']['weighted']-row['optical']['weighted'];folds.append(row)
    if any(set(counts[v])!=set(development) for v in counts):raise ValueError('incomplete development coverage')
    groups=sorted({mapping[c] for c in development});grouped={v:np.asarray([np.sum([counts[v][c] for c in development if mapping[c]==g],axis=0) for g in groups]) for v in counts}
    metrics={v:bs_scores(grouped[v].sum(0)) for v in counts};delta=metrics['siam']['weighted']-metrics['optical']['weighted']
    rng=np.random.default_rng(20260918);boot=[]
    for _ in range(2000):
        sample=rng.integers(len(groups),size=len(groups));a=bs_scores(grouped['optical'][sample].sum(0))['weighted'];b=bs_scores(grouped['siam'][sample].sum(0))['weighted']
        if a is not None and b is not None:boot.append(b-a)
    if len(boot)<1900:raise ValueError('too many undefined bootstrap scores')
    lo,hi=map(float,np.quantile(boot,[.025,.975]))
    report=dict(metrics=metrics,delta=delta,ci95_event_bootstrap=[lo,hi],accepted=bool(delta>=.005 and lo>0),folds=folds,events=len(groups),chips=len(development),artifact_sha256=hashes,non_claims=['Development revalidation after adaptive model selection; not an untouched test.','Bootstrap is conditional on fitted models and does not refit the full selection/training procedure.','Provided event IDs do not prove geographic or temporal independence.','Does not establish CEMS or competition hidden-test superiority.'])
    write_json(args.out,report);print(json.dumps({k:v for k,v in report.items() if k not in ('folds','artifact_sha256')},indent=2))
if __name__=='__main__':main()
