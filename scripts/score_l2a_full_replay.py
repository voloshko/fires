"""SPEC-101: frozen full-pool adaptation against matched and original controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file
from src.comp.accuracy_artifacts import execution_sources,checkpoint_metadata,assert_matched_recipes,load_checkpoint
from src.comp.l2a_finetuning import l2a_water
from src.comp.russian_error_report import summarize
from scripts.l2a_full_replay_io import load_manifest,verify_baseline
from scripts.l2a_finetune import read_twin
from scripts.score_l2a_finetuning import predict,l2a_acceptance
from scripts.score_forest_pair import binary_counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","manifest","runs","reference-root","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable result exists")
    manifest=load_manifest(args.data_root,args.manifest);digest=sha256_file(args.manifest)
    fit=[r for r in manifest["records"] if r["partition"]=="fit"]
    evaluation=[r for r in manifest["records"] if r["partition"]=="evaluation"]
    models={v:[args.runs/f"{v}-s{s}"/"model.pt" for s in (1,2,3)] for v in ("control","replay")}
    models["initial"]=[args.reference_root/f"hls-c1mm-final-s{s}"/"model.pt" for s in (1,2,3,4,5)]
    verify_baseline(models["initial"],manifest["baseline_model_hashes"])
    records={v:[checkpoint_metadata(p,export_root=Path("research/tensor-exports")) for p in models[v]] for v in ("control","replay")}
    recipe=assert_matched_recipes(records["control"]+records["replay"],
        ("w","depth","epochs","batch","classes","initializer_pool","source_schedule","normalization_mode"))
    sources=execution_sources("scripts/l2a_full_replay_train.py","scripts/l2a_full_replay_io.py",
        "src/comp/l2a_full_replay.py","src/comp/l2a_finetuning.py","scripts/queues/queue_l2a_full_replay.sh",tta=False)
    expected_sources={str(p):sha256_file(p) for p in sources}
    summaries={}
    for seed in (1,2,3):
        initializer=manifest["initializers"][seed-1]
        if sha256_file(models["initial"][seed-1])!=initializer["sha256"]:raise ValueError("initial reference differs")
        original=load_checkpoint(models["initial"][seed-1],initializer["sha256"],export_root=Path("research/tensor-exports"))
        norm={k:np.asarray(original[k],np.float32).tolist() for k in ("mean","std")};del original
        for variant in ("control","replay"):
            metadata=records[variant][seed-1]
            if (metadata["spec_id"]!="SPEC-101" or metadata["variant"]!=variant or metadata["seed"]!=seed
                    or metadata["smoke"] or metadata["epochs"]!=30 or metadata["batch"]!=8
                    or metadata["data_manifest"]!=digest or metadata["source_hashes"]!=expected_sources
                    or metadata["initialization_sha256"]!=initializer["sha256"]
                    or metadata["fit_ids"]!=[r["sample_id"] for r in fit]
                    or any(metadata[k]!=norm[k] for k in norm)):
                raise ValueError("continuation recipe/lineage/normalization differs")
            summary=json.loads(models[variant][seed-1].with_name("summary.json").read_text())
            weights=np.array([r["sampling_weight"] for r in fit])
            full=np.array([(15 if r["twin"] is not None else 30)*w for r,w in zip(fit,weights)])
            common=30*weights-full;actual_l2a=common if variant=="replay" else np.zeros(len(fit),int)
            if (not summary["completed"] or summary["fingerprint"]!=metadata["fingerprint"]
                    or summary["full_hls_exposures"]!=full.tolist() or summary["common_exposures"]!=common.tolist()
                    or summary["l2a_exposures"]!=actual_l2a.tolist()):
                raise ValueError("full original/source exposure evidence differs")
            summaries[(variant,seed)]=summary
        if summaries[("control",seed)]["order_sha256"]!=summaries[("replay",seed)]["order_sha256"]:
            raise ValueError("matched ordering/augmentation RNG drift")
    probabilities={v:predict(args.data_root,evaluation,paths,v,digest,
        args.out.parent/(args.out.stem+"-"+v+".npy")) for v,paths in models.items()}
    counts={v:[] for v in models};rows=[]
    for i,record in enumerate(evaluation):
        image,truth,valid=read_twin(args.data_root,record,"l2a")
        with rasterio.open(args.data_root/record["files"]["scl"]) as source:scl=source.read(1)
        water=l2a_water(image,scl)
        row=dict(sample_id=record["sample_id"],group_id=record["group_id"],dataset=record["dataset"],
                 valid_pixels=int(valid.sum()),total_pixels=int(valid.size))
        for name,prob in probabilities.items():
            cm=binary_counts(truth==1,(prob[i]>=.5)&~water,valid);counts[name].append(cm);row[name]=cm
        rows.append(row)
    result=l2a_acceptance(evaluation,counts["replay"],counts["control"],counts["initial"])
    result.update(spec_id="SPEC-101",recipe_id=recipe,data_manifest=digest,per_window=rows,
        errors=summarize(rows,list(models)),full_original_fit_count=len(fit),fit_l2a_twins=manifest["fit_twins"],
        source_exposure_per_seed={str(s):dict(full_hls=sum(summaries[("replay",s)]["full_hls_exposures"]),
            l2a=sum(summaries[("replay",s)]["l2a_exposures"])) for s in (1,2,3)},
        model_hashes={v:[sha256_file(p) for p in paths] for v,paths in models.items()},
        baseline_expected_hashes=manifest["baseline_model_hashes"],
        non_claims=["Mixed-source adaptation preserves original supervision; not a SPEC93/97 L2A-only event-coverage PASS.",
                    "Mediterranean development evaluation, not independent Siberian/global competitor superiority.",
                    "Published satellite interpretation, not field truth."])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_window","errors")}),flush=True)


if __name__=="__main__":main()
