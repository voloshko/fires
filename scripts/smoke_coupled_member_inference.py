"""SPEC-121: actual512 reference parity and coupled output mechanics, no quality claim."""
import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.score_coupled_member_inference import expanded_predictions
from scripts.score_russian_transfer import predict
from scripts.forest_pair_train import read_sample
from src.comp.accuracy_contract import sha256_file
from src.comp.accuracy_artifacts import execution_sources,freeze_recipe
from src.comp.five_seed_additions import audit_all
from src.comp.temporal_adapter_contract import read_seal,fit_population
from src.comp.coupled_member_inference import guard_output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','members-root','reference-root','prithvi-root','out'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.data_root
    if args.out.exists():raise ValueError('immutable smoke output exists')
    args.out.mkdir();args.runs=args.out
    base=json.load(open(root/'forest-pairs/manifest.json'))['records']
    russian=json.load(open(root/'russia/manifest.json'))['records']
    seal=read_seal(args.members_root/'recipe.json')
    paths,_,_=audit_all(Namespace(data_root=root,runs=args.members_root),fit_population(base,russian),{},seal)
    record=sorted((r for r in russian if r['partition']=='fit'),key=lambda r:r['sample_id'])[0]
    recipe=freeze_recipe(args.out/'recipe.json',execution_sources(Path(__file__),
        'scripts/score_coupled_member_inference.py','src/comp/coupled_member_inference.py',
        'scripts/score_russian_transfer.py','scripts/queues/smoke_coupled_member_inference.sh'),
        dict(sample_id=record['sample_id'],input_hashes=record['hashes'],members=[sha256_file(p)for ps in paths.values()for p in ps],smoke=True))
    reference_paths=[args.reference_root/f'hls-c1mm-final-s{s}/model.pt'for s in range(1,6)]
    guard_output(args.out,2*1024**2)
    anchor=predict(root,[record],reference_paths,'c1mm',args.out/'anchor.npy',args.prithvi_root)
    args.anchor_cache_sha=sha256_file(args.out/'anchor.npy')
    variants=[]
    for variant in ('pair','post'):
        prediction=expanded_predictions(args,[record],anchor,paths[variant],variant,args.out/(variant+'.packed.npz'))
        _,_,valid=read_sample(root,record,'post')
        variants.append(dict(variant=variant,shape=list(prediction.shape),reference_parity_passed=True,
            finite_coupled_probability=True,reference_positives_preserved=True,prediction_pixels=int(prediction.sum()),
            valid_pixels=int(valid.sum())))
    result=dict(spec_id='SPEC-121',recipe_id=recipe,variants=variants,
        non_claims=['Actual512 source/model/coupling/parity mechanics only; no accuracy acceptance.'])
    guard_output(args.out,65536)
    with (args.out/'result.json').open('x')as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
