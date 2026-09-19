"""SPEC-38: final420 hard-negative AF, preserving submitted v13 BS bytes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.hypothesis_lab import digest,write_json


def replace_af_rows(base,replacements,out):
    """Replace exactly class1 for known AF IDs; copy every other line verbatim."""
    csv.field_size_limit(10_000_000)
    lines=Path(base).read_bytes().splitlines(keepends=True)
    if next(csv.reader([lines[0].decode()]))!=['chip_id','class_id','rle']: raise ValueError('unexpected columns')
    seen=set(); output=[lines[0]]; before=[]; after=[]; changed=0
    for line in lines[1:]:
        row=next(csv.reader([line.decode()]))
        if len(row)!=3: raise ValueError('invalid CSV row')
        chip,cls,rle=row
        if chip in replacements:
            if int(cls)!=1 or chip in seen: raise ValueError('invalid or duplicate AF row')
            seen.add(chip); buf=io.StringIO();csv.writer(buf,lineterminator='\n').writerow([chip,1,replacements[chip]])
            output.append(buf.getvalue().encode());changed+=rle!=replacements[chip]
        else: output.append(line);before.append(line);after.append(output[-1])
    if seen!=set(replacements): raise ValueError('missing AF rows')
    target=Path(out)
    if target.exists(): raise FileExistsError(target)
    target.write_bytes(b''.join(output))
    a=hashlib.sha256(b''.join(before)).hexdigest(); b=hashlib.sha256(b''.join(after)).hexdigest()
    assert a==b
    return dict(changed_af_rows=int(changed),af_rows=len(seen),unchanged_other_rows=len(before),bs_sha256=a,bs_bytes_preserved=True)


def main():
    from src.comp.af import AfDataset,train,save,predict,predict_model
    from src.comp.rle import encode
    from src.comp.submission import validate
    from scripts.hypothesis_lab import manifest,af_cache,af_set
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--test',required=True);p.add_argument('--out',required=True);p.add_argument('--train',default='data/comp/train/af');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);run=manifest(args)
    run.update(seed=20260918,cutoff=.5,base_sha256=digest(args.base),template_sha256=digest(Path(args.test)/'sample_submission.csv'),purpose='final training, no holdout quality evaluation')
    write_json(out/'manifest.json',run)
    template=Path(args.test)/'sample_submission.csv'; faults=validate(args.base,template)
    if faults: raise ValueError(f'base validation failed: {faults}')
    training=AfDataset(args.train);ids=training.chip_ids()
    if len(ids)!=420: raise ValueError('expected 420 final training chips')
    start=time.time();cache,hashes=af_cache(args.train,out,ids);write_json(out/'training_data_manifest.json',hashes)
    x,y=af_set(cache,ids,20260918);base_model=train(x,y,20260918);initial_samples=len(y);del x,y
    x,y=af_set(cache,ids,20260918,base_model);model=train(x,y,20260918)
    if len(y)!=initial_samples: raise ValueError('sampling budget changed')
    write_json(out/'training_summary.json',dict(chips=len(ids),samples=len(y),positive_samples=int(y.sum()),seconds=time.time()-start,quality_evaluated=False))
    save(model,out/'model.pkl',cutoff=.5);del x,y,base_model
    testing=AfDataset(Path(args.test)/'af'); replacements={};pixels={};input_hashes={};reload_checked=False
    for c in testing.chip_ids():
        ch=testing.load(c);mask=predict_model(model,ch,cutoff=.5)
        if not reload_checked:
            assert np.array_equal(mask,predict(ch,out/'model.pkl'));reload_checked=True
        replacements[c]=encode(mask>0);pixels[c]=int(mask.sum())
        input_hashes[c]={str(f):digest(f) for folder in ('viirs','aux') for f in (Path(args.test)/'af'/folder).glob(c+'_*.tif')}
    output=out/'submission_candidate_af_hard.csv'; preservation=replace_af_rows(args.base,replacements,output)
    with output.open() as f: rows=list(csv.DictReader(f))
    shapes={r['chip_id']:((256,256) if r['chip_id'] in replacements else (512,512)) for r in rows}
    faults=validate(output,template,shapes)
    if len(rows)!=447 or faults: raise ValueError(f'candidate invalid: rows={len(rows)}, faults={faults}')
    write_json(out/'test_input_manifest.json',input_hashes)
    result=dict(status='PASS',rows=len(rows),validator_faults=faults,model_sha256=digest(out/'model.pkl'),submission_sha256=digest(output),base_sha256=run['base_sha256'],reload_checked=reload_checked,af_empty_chips=sum(n==0 for n in pixels.values()),af_fire_pixels=sum(pixels.values()),af_pixels_by_chip=pixels,seconds=time.time()-start,**preservation,non_claims=['No hidden-test labels accessed or quality measured.','Prediction counts are not accuracy metrics.','Historical holdout is used only for final fitting, not evaluation.','No submission sent to organizers.'])
    write_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='af_pixels_by_chip'},indent=2),flush=True)
if __name__=='__main__':main()
