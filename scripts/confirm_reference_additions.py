"""SPEC-104: one frozen recipe, unused Russian confirmation, all competitors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file,paired_group_comparison
from src.comp.accuracy_artifacts import checkpoint_metadata,execution_sources,freeze_recipe
from src.comp.l2a_full_replay import REFERENCE_RESULT_SHA256
from src.comp.pair_fusion import confirmation_acceptance
from src.comp.russian_error_report import summarize
from scripts.russian_transfer_train import load_russian
from scripts.score_russian_transfer import predict,counts_for
from scripts.develop_reference_additions import predictions
from scripts.report_russian_errors import completed_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","plan","development","parent","russian-manifest","runs","reference-root","prithvi-root","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable confirmation already exists")
    plan=json.loads(args.plan.read_text())
    if (plan["spec_id"]!="SPEC-104" or not plan["eligible"] or plan["addition_threshold"]!=.95
            or plan["reference_threshold"]!=.5 or plan["threshold"]!=.5
            or plan["original_partition"]!="calibration"):
        raise ValueError("not the frozen eligible recipe")
    if sha256_file(args.parent)!=REFERENCE_RESULT_SHA256 or plan["parent_result_sha256"]!=REFERENCE_RESULT_SHA256:
        raise ValueError("wrong completed parent")
    if sha256_file(args.development)!=plan["development_sha256"]:raise ValueError("development result changed")
    for path,digest in plan["source_hashes"].items():
        if sha256_file(path)!=digest:raise ValueError("development source changed")
    if sha256_file(args.russian_manifest)!=plan["russian_manifest"]:raise ValueError("Russian population changed")
    all_records=load_russian(args.data_root,args.russian_manifest)
    records=[r for r in all_records if r["partition"]=="calibration"]
    if [r["sample_id"] for r in records]!=plan["confirmation_ids"] or len(records)<50 or len({r["group_id"] for r in records})<20:
        raise ValueError("original confirmation population/order differs")
    models=dict(reserve=[args.runs/f"reserve-s{s}"/"model.pt" for s in (1,2,3)],
                c1mm=[args.reference_root/f"hls-c1mm-final-s{s}"/"model.pt" for s in (1,2,3,4,5)],prithvi=[])
    confirm_ids=set(plan["confirmation_ids"])
    for name in ("reserve","c1mm"):
        if [sha256_file(p) for p in models[name]]!=plan["model_hashes"][name]:raise ValueError("frozen model lineage changed")
    for path in models["reserve"]:
        metadata=checkpoint_metadata(path,export_root=Path("research/tensor-exports"))
        if confirm_ids.intersection(metadata["fit_ids"]):raise ValueError("confirmation was fitted/mined")
    if sha256_file(args.prithvi_root/"Prithvi_EO_V2_300M_BurnScars.pt")!=plan["prithvi_weight_sha256"]:
        raise ValueError("Prithvi reference changed")
    # Verify the original signed development caches again; this binds the
    # official Prithvi YAML as well as its weights before unused predictions.
    parent=json.loads(args.parent.read_text())
    completed_predictions(args,parent,[r for r in all_records if r["partition"]=="evaluation"],"russia")
    sources=execution_sources(Path(__file__),"src/comp/reference_additions.py","src/comp/component_rescue.py","src/comp/pair_fusion.py",
        "scripts/develop_reference_additions.py","scripts/score_russian_transfer.py","scripts/russian_transfer_train.py",
        "src/comp/l2a_finetuning.py","src/comp/terrain.py","src/comp/severity_layers.py",
        "scripts/queues/queue_reference_additions_confirmation.sh")
    recipe=freeze_recipe(args.out.with_suffix(".recipe.json"),sources,
        dict(plan_sha256=sha256_file(args.plan),input_manifest=sha256_file(args.russian_manifest),
             models=plan["model_hashes"],prithvi=plan["prithvi_weight_sha256"],
             prithvi_config=sha256_file(args.prithvi_root/"burn_scars_config.yaml"),reference_threshold=.5,addition_threshold=.95))
    prob={v:predict(args.data_root,records,paths,v,args.out.parent/(args.out.stem+"-"+v+".npy"),args.prithvi_root)
          for v,paths in models.items()}
    mask,rescued=predictions(args.data_root,records,prob);prob["fusion"]=mask
    counts,rows=counts_for(args.data_root,records,prob)
    for row,record in zip(rows,records):row["total_pixels"]=record["grid"]["width"]*record["grid"]["height"]
    report=summarize(rows,list(prob));groups=[r["group_id"] for r in records]
    comparisons={v:paired_group_comparison(groups,counts["fusion"],counts[v]) for v in models}
    missed={v:len(r["missed_positive_groups"]) for v,r in report["models"].items()}
    accepted=confirmation_acceptance(comparisons,missed)
    result=dict(spec_id="SPEC-104",accepted=accepted,comparisons=comparisons,missed_positive_clusters=missed,
        errors=report,per_window=rows,rescued=rescued,recipe_id=recipe,plan_sha256=sha256_file(args.plan),
        source_hashes={str(p):sha256_file(p) for p in sources},
        model_hashes=plan["model_hashes"],prithvi_weight_sha256=plan["prithvi_weight_sha256"],
        non_claims=["Previously reserved calibration is consumed once for this fixed confirmation, no tuning on it.",
                    "Spring2020 published interval perimeters; spatial clusters/components are not physical fire IDs.",
                    "No field truth, global leaderboard or fresh Mediterranean superiority claim."])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("errors","per_window","rescued","source_hashes")}),flush=True)


if __name__=="__main__":main()
