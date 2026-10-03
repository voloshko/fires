"""SPEC-98: frozen equal paired severity on previously unused events."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file, validate_manifest
from src.comp.accuracy_artifacts import checkpoint_metadata, assert_matched_recipes, execution_sources
from src.comp.fresh_severity import assert_fresh, verify_frozen_models, verify_development_result, population
from scripts.prepare_fresh_paired_severity import validate_cached_record
from src.comp.paired_severity import (apply_fixed_boundary, grouped_severity_comparison,
    mtbs_evaluation_labels, severity_confusion, severity_metrics)
from src.comp.severity_layers import evaluation_fmask
from src.comp.terrain import water_ndwi
from scripts.forest_pair_train import read_sample
from scripts.score_forest_pair import infer as infer_boundary
from scripts.score_paired_severity import predict_severity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","manifest","runs","development-result","reference-root","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--profile",choices=("2024-west","retrospective-2016-17"),default="2024-west")
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable result already exists")
    payload=json.loads(args.manifest.read_text());records=payload["records"]
    spec_id,_,_=population(args.profile)
    if payload["spec_id"]!=spec_id or payload.get("profile","2024-west")!=args.profile:
        raise ValueError("evaluation manifest belongs to another population")
    if not payload["ready_for_evaluation"] or payload["qc_probe"]:raise ValueError("no full evaluation admission")
    counts=validate_manifest(records,require_pairs=True)
    if counts["groups"]<100:raise ValueError("fewer than100 independent fresh events")
    selection=args.manifest.parent/"selection.json"
    if sha256_file(selection)!=payload["selection_sha256"]:raise ValueError("selection changed")
    plan=json.loads(selection.read_text());assert_fresh(records,plan)
    if plan.get("profile","2024-west")!=args.profile:raise ValueError("selection profile differs")
    events={r["event_id"]:r for r in plan["selected"]}
    for record in records:
        validate_cached_record(args.data_root,record,events[record["group_id"].removeprefix("mtbs:")])
        for key,path in record["files"].items():
            if sha256_file(args.data_root/path)!=record["hashes"][key]:raise ValueError("evaluation input changed")
    verify_development_result(args.development_result)
    development=json.loads(args.development_result.read_text())
    if development["spec_id"]!="SPEC-94":raise ValueError("wrong archived development result")
    models=[args.runs/f"equal-s{s}"/"model.pt" for s in (1,2,3)]
    boundaries=[args.reference_root/f"hls-c1mm-final-s{s}"/"model.pt" for s in (1,2,3,4,5)]
    verify_frozen_models(models,development["model_hashes"]["equal"])
    verify_frozen_models(boundaries,development["boundary_hashes"])
    metadata=[checkpoint_metadata(path,export_root=Path("research/tensor-exports")) for path in models]
    recipe=assert_matched_recipes(metadata,("width","depth","epochs","batch","classes","evaluation_ids",
                                          "fit_class_counts","mean","std"))
    if recipe!=development["recipe_id"]:raise ValueError("candidate recipe differs from SPEC94")
    probabilities=predict_severity(args.data_root,records,models,"equal",development["data_manifest"],
                                  args.out.parent/(args.out.stem+"-equal.npy"))
    boundary_prob=infer_boundary(args.data_root,records,boundaries,"c1mm",sha256_file(args.manifest),
                                 args.out.parent/(args.out.stem+"-boundary.npy"))
    counts={v:[] for v in ("equal","hybrid")};conditional={v:[] for v in counts};rows=[]
    class_pixels=np.zeros(3,np.int64)
    for i,record in enumerate(records):
        image,_,valid=read_sample(args.data_root,record,"pair")
        with rasterio.open(args.data_root/record["files"]["severity"]) as source:raw=source.read(1)
        truth=mtbs_evaluation_labels(raw,valid);known=truth>=0
        class_pixels+=np.array([(truth==c).sum() for c in (1,2,3)])
        fm,fm_hash=evaluation_fmask(args.data_root,record);post=image[6:]
        boundary=(boundary_prob[i]>=.5)&~water_ndwi(fm,post[1],post[3])
        nbr=lambda x:(x[3]-x[5])/(x[3]+x[5]+1e-6)
        levels=np.digitize(nbr(image[:6])-nbr(post),[.27,.66])+1
        predictions=dict(equal=apply_fixed_boundary(probabilities[i],boundary,known),
                         hybrid=np.where(boundary&known,levels,0))
        row=dict(sample_id=record["sample_id"],group_id=record["group_id"],valid_pixels=int(known.sum()),fmask_sha256=fm_hash)
        for name,pred in predictions.items():
            cm=severity_confusion(truth,pred);counts[name].append(cm);row[name]=cm.tolist()
            grades=levels if name=="hybrid" else probabilities[i].argmax(0)+1
            conditional[name].append(severity_confusion(np.where(truth>0,truth,-1),grades))
        rows.append(row)
    if not (class_pixels>0).all():raise ValueError("missing severity class on evaluation")
    comparison=grouped_severity_comparison([r["group_id"] for r in records],counts["equal"],counts["hybrid"])
    delta=np.asarray(comparison["candidate"]["per_class"])-comparison["baseline"]["per_class"]
    accepted=comparison["delta"]>=.02 and comparison["ci95"][0]>0 and np.all(delta>=-.02)
    sources=execution_sources(Path(__file__),"scripts/score_paired_severity.py","src/comp/paired_severity.py",
                              "scripts/score_forest_pair.py","scripts/forest_pair_train.py",
                              "src/comp/severity_layers.py","src/comp/fresh_severity.py","src/comp/terrain.py")
    result=dict(spec_id=spec_id,profile=args.profile,accepted=bool(accepted),versus_hybrid=comparison,
                per_class_delta=delta.tolist(),class_pixels=class_pixels.tolist(),per_window=rows,
                on_true_burn={k:severity_metrics(np.sum(c,axis=0)) for k,c in conditional.items()},
                recipe_id=recipe,data_manifest=sha256_file(args.manifest),development_sha256=sha256_file(args.development_result),
                model_hashes=development["model_hashes"]["equal"],boundary_hashes=development["boundary_hashes"],
                source_hashes={str(s):sha256_file(s) for s in sources},
                non_claims=[f"{args.profile} published MTBS satellite-label agreement, not field truth.",
                            "Retrospective independent events do not establish prospective accuracy.",
                            "No Siberian severity validation or global leaderboard claim.",
                            "No superiority over models not compared on these pixels."])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="per_window"}),flush=True)


if __name__=="__main__":main()
