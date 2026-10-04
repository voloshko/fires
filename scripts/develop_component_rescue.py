"""SPEC-103: fixed GT-blind component rescue, CPU-only registered development."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file,paired_group_comparison
from src.comp.component_rescue import rescue,eligible
from src.comp.l2a_full_replay import REFERENCE_RESULT_SHA256
from src.comp.l2a_finetuning import l2a_water
from src.comp.severity_layers import evaluation_fmask
from src.comp.terrain import water_ndwi
from src.comp.russian_error_report import summarize
from scripts.forest_pair_train import read_sample
from scripts.report_russian_errors import completed_predictions
from scripts.score_russian_transfer import counts_for


def predictions(root,records,probabilities):
    masks=[];diagnostics=[]
    for i,record in enumerate(records):
        image,_,valid=read_sample(root,record,"post")
        if record.get("date_semantics")=="annual-interval-pair":
            with rasterio.open(root/record["files"]["post_scl"]) as source:scl=source.read(1)
            water=l2a_water(image,scl)
        else:
            fmask,_=evaluation_fmask(root,record);water=water_ndwi(fmask,image[1],image[3])
        mask,diagnostic=rescue(probabilities["c1mm"][i],probabilities["reserve"][i],valid,water)
        masks.append(mask);diagnostics.append(dict(sample_id=record["sample_id"],**diagnostic))
    return np.stack(masks),diagnostics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","parent","russian-manifest","base-manifest","runs","reference-root","prithvi-root","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable result exists")
    if sha256_file(args.parent)!=REFERENCE_RESULT_SHA256:raise ValueError("wrong parent result")
    parent=json.loads(args.parent.read_text())
    for key,path in (("russian_manifest",args.russian_manifest),("base_manifest",args.base_manifest)):
        if sha256_file(path)!=parent[key]:raise ValueError("source manifest differs")
    russian=json.loads(args.russian_manifest.read_text())["records"];base=json.loads(args.base_manifest.read_text())["records"]
    populations=dict(russia=[r for r in russian if r["partition"]=="evaluation"])
    populations.update({d:[r for r in base if r["partition"]=="evaluation" and r["dataset"]==d] for d in ("ems","floga")})
    comparisons={};summaries={};restored={}
    for dataset,records in populations.items():
        prob=completed_predictions(args,parent,records,dataset)
        mask,diagnostic=predictions(args.data_root,records,prob)
        prob["rescue"]=mask
        counts,rows=counts_for(args.data_root,records,prob)
        if dataset=="russia":
            if [{k:v for k,v in r.items() if k!="rescue"} for r in rows]!=parent["per_window"]:
                raise ValueError("parent counts differ")
        for row,r in zip(rows,records):row["total_pixels"]=r["grid"]["width"]*r["grid"]["height"]
        summary=summarize(rows,list(prob));summaries[dataset]=summary
        comparison=paired_group_comparison([r["group_id"] for r in records],counts["rescue"],counts["c1mm"])
        comparison.update(candidate_missed_groups=len(summary["models"]["rescue"]["missed_positive_groups"]),
                          baseline_missed_groups=len(summary["models"]["c1mm"]["missed_positive_groups"]))
        comparisons[dataset]=comparison;restored[dataset]=diagnostic
    allowed=eligible(comparisons["russia"],{d:comparisons[d] for d in ("ems","floga")})
    confirmation=[r for r in russian if r["partition"]=="calibration"]
    if len(confirmation)<50 or len({r["group_id"] for r in confirmation})<20:raise ValueError("confirmation population insufficient")
    if any(args.out.parent.glob("component-rescue-v1-confirmation*.npy")):
        raise ValueError("registered confirmation already consumed")
    result=dict(spec_id="SPEC-103",development_eligible=allowed,comparisons=comparisons,summaries=summaries,
                restored=restored,parent_result_sha256=REFERENCE_RESULT_SHA256,
                non_claims=["Development cohorts already evaluated; no independent superiority.",
                            "Connected components are not physical fire IDs."])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    plan=dict(spec_id="SPEC-103",eligible=allowed,recipe="mean+8connected-rejected-reference-components-size<=64",
        development_sha256=sha256_file(args.out),parent_result_sha256=REFERENCE_RESULT_SHA256,
        russian_manifest=parent["russian_manifest"],base_manifest=parent["base_manifest"],
        confirmation_ids=[r["sample_id"] for r in confirmation],original_partition="calibration",
        model_hashes=parent["model_hashes"],prithvi_weight_sha256=parent["competitor_weights_sha256"],
        threshold=.5,component_cap=64,connectivity=8,alpha=.5,tta=8,precision="bf16",
        source_hashes={p:sha256_file(p) for p in ("scripts/develop_component_rescue.py","src/comp/component_rescue.py",
            "src/comp/pair_fusion.py","scripts/report_russian_errors.py","scripts/score_russian_transfer.py")})
    with args.out.with_name("component-rescue-v1-plan.json").open("x") as stream:json.dump(plan,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("summaries","restored")}),flush=True)


if __name__=="__main__":main()
