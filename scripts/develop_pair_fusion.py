"""SPEC-102: CPU-only development, then freeze one unused confirmation plan."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file,paired_group_comparison
from src.comp.l2a_full_replay import REFERENCE_RESULT_SHA256
from src.comp.pair_fusion import RECIPES,fuse,select_recipe
from src.comp.russian_error_report import summarize
from scripts.report_russian_errors import completed_predictions
from scripts.score_russian_transfer import counts_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","parent","russian-manifest","base-manifest","runs","reference-root","prithvi-root","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable development result exists")
    if sha256_file(args.parent)!=REFERENCE_RESULT_SHA256:raise ValueError("wrong frozen SPEC92 result")
    parent=json.loads(args.parent.read_text())
    for key,path in (("russian_manifest",args.russian_manifest),("base_manifest",args.base_manifest)):
        if sha256_file(path)!=parent[key]:raise ValueError("frozen source manifest changed")
    russian=json.loads(args.russian_manifest.read_text())["records"]
    base=json.loads(args.base_manifest.read_text())["records"]
    populations={"russia":[r for r in russian if r["partition"]=="evaluation"]}
    populations.update({d:[r for r in base if r["partition"]=="evaluation" and r["dataset"]==d] for d in ("ems","floga")})
    reports={name:dict(mediterranean={}) for name in RECIPES};summaries={}
    for dataset,records in populations.items():
        probabilities=completed_predictions(args,parent,records,dataset)
        for name in RECIPES:probabilities[name]=fuse(probabilities["c1mm"],probabilities["reserve"],name)
        counts,rows=counts_for(args.data_root,records,probabilities)
        if dataset=="russia":
            original=[{k:v for k,v in r.items() if k not in RECIPES} for r in rows]
            if original!=parent["per_window"]:raise ValueError("development parent counts differ")
        for row,record in zip(rows,records):row["total_pixels"]=record["grid"]["width"]*record["grid"]["height"]
        summaries[dataset]=summarize(rows,list(probabilities))
        groups=[r["group_id"] for r in records]
        for name in RECIPES:
            comparison=paired_group_comparison(groups,counts[name],counts["c1mm"])
            if dataset=="russia":reports[name]["russian_vs_c1mm"]=comparison
            else:
                comparison.update(candidate_missed_groups=len(summaries[dataset]["models"][name]["missed_positive_groups"]),
                    baseline_missed_groups=len(summaries[dataset]["models"]["c1mm"]["missed_positive_groups"]))
                reports[name]["mediterranean"][dataset]=comparison
    chosen=select_recipe(reports)
    calibration=[r for r in russian if r["partition"]=="calibration"]
    if len(calibration)<50 or len({r["group_id"] for r in calibration})<20:raise ValueError("insufficient original unused calibration population")
    if any(args.out.parent.glob("pair-fusion-v1-confirmation*.npy")):
        raise ValueError("registered confirmation already has predictions; unused claim unavailable")
    plan=dict(spec_id="SPEC-102",chosen_recipe=chosen,development_sha256=None,
        confirmation_ids=[r["sample_id"] for r in calibration],original_partition="calibration",
        russian_manifest=parent["russian_manifest"],base_manifest=parent["base_manifest"],
        parent_result_sha256=REFERENCE_RESULT_SHA256,model_hashes=parent["model_hashes"],
        prithvi_weight_sha256=parent["competitor_weights_sha256"],
        threshold=.5,tta=8,precision="bf16",bootstrap=2000,bootstrap_seed=20261003,
        primary=dict(delta_to_both_competitors=.01,ci_lower_min_exclusive=0.,reserve_delta_min=-.01),
        source_hashes={str(p):sha256_file(p) for p in (Path(__file__),Path("src/comp/pair_fusion.py"),
            Path("scripts/report_russian_errors.py"),Path("scripts/score_russian_transfer.py"))},
        non_claims=["Choice uses already evaluated development cohorts, not independent superiority.",
                    "No prior registered calibration predictions; original split preserved, calibration consumed once as confirmation."])
    result=dict(spec_id="SPEC-102",reports=reports,summaries=summaries,selected_recipe=chosen,
                parent_result_sha256=REFERENCE_RESULT_SHA256,non_claims=plan["non_claims"])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    plan["development_sha256"]=sha256_file(args.out)
    with args.out.with_name("pair-fusion-v1-plan.json").open("x") as stream:json.dump(plan,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="summaries"}),flush=True)


if __name__=="__main__":main()
