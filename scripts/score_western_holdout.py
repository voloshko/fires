"""SPEC-105: one frozen geographic comparison, no source/weight selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file,validate_manifest,paired_group_comparison
from src.comp.accuracy_artifacts import checkpoint_metadata,execution_sources,freeze_recipe
from src.comp.l2a_finetuning import l2a_valid
from src.comp.l2a_full_replay import REFERENCE_RESULT_SHA256
from src.comp.russian_error_report import summarize
from src.comp.western_holdout import BUDGET_BYTES,verify_exclusions
from scripts.prepare_russian_windows import interval_pair_valid
from scripts.prepare_western_holdout import SOURCE_SHA
from scripts.score_russian_transfer import predict,counts_for
from scripts.report_russian_errors import completed_predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","manifest","runs","reference-root","prithvi-root","parent","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable geographic result exists")
    if sha256_file(args.parent)!=REFERENCE_RESULT_SHA256:raise ValueError("frozen parent changed")
    parent=json.loads(args.parent.read_text());payload=json.loads(args.manifest.read_text())
    if payload["spec_id"]!="SPEC-105" or not payload["ready_for_evaluation"] or payload["qc_probe"]:
        raise ValueError("no full geographic admission")
    records=payload["records"];counts=validate_manifest(records)
    if counts["samples"]<50 or counts["groups"]<20:raise ValueError("geographic population too small")
    if payload.get("selection_file")!="selection-v2.json":raise ValueError("geographic source completeness not audited")
    selection=args.manifest.parent/payload["selection_file"]
    if sha256_file(selection)!=payload["selection_sha256"]:raise ValueError("selection changed")
    plan=json.loads(selection.read_text());selected={r["sample_id"]:r for r in plan["records"]}
    if not plan["ready_for_collection"] or plan["bindings"]["source_zip"]!=SOURCE_SHA:
        raise ValueError("wrong geographic source selection")
    if sha256_file(args.data_root/"russia/burned_area_2020_topology.zip")!=SOURCE_SHA:
        raise ValueError("source ZIP changed")
    for name,digest in plan["bindings"].items():
        if name!="source_zip" and sha256_file(args.data_root/name)!=digest:raise ValueError("history/selection inputs changed")
        if name!="source_zip":
            audit=verify_exclusions(name,json.loads((args.data_root/name).read_text()),digest)
            if plan.get("exclusion_audit",{}).get(name)!=audit:raise ValueError("geographic source completeness differs")
    used=sum(p.stat().st_size for p in args.manifest.parent.rglob("*") if p.is_file())
    if used>BUDGET_BYTES:raise ValueError("geographic data budget exceeded")
    for record in records:
        frozen=selected.get(record["sample_id"])
        if frozen is None:raise ValueError("unselected geographic record")
        for key in ("grid","bbox_wgs84","group_id","partition","center"):
            if record[key]!=frozen[key]:raise ValueError("geographic record identity/geometry drift")
        interval_pair_valid(record);arrays={}
        for key,relative in record["files"].items():
            path=args.data_root/relative
            if sha256_file(path)!=record["hashes"][key]:raise ValueError("geographic pixel source changed")
            with rasterio.open(path) as source:
                if (str(source.crs)!=record["grid"]["crs"] or list(source.transform)[:6]!=record["grid"]["transform"]
                        or source.shape!=(512,512)):
                    raise ValueError("geographic raster grid differs")
                arrays[key]=source.read() if key in ("pre","post") else source.read(1)
        source=gpd.read_file("zip://"+str(args.data_root/"russia/burned_area_2020_topology.zip"),bbox=tuple(record["bbox_wgs84"]))
        if str(source.crs)!="EPSG:4326":raise ValueError("source annotation CRS differs")
        source=source[source.geometry.notna()&~source.geometry.is_empty&source.geometry.is_valid].to_crs(record["grid"]["crs"])
        truth=rasterize([(g,1) for g in source.geometry],out_shape=(512,512),
            transform=rasterio.Affine(*record["grid"]["transform"]),fill=0,dtype="int16")
        expected=l2a_valid(arrays["pre"],arrays["pre_scl"],truth)&l2a_valid(arrays["post"],arrays["post_scl"],truth)
        if not np.array_equal(expected,arrays["valid"].astype(bool)):raise ValueError("strict common observation mask differs")
        if not np.array_equal(np.where(expected,truth,-1),arrays["mask"]):raise ValueError("published supervision differs")
        if expected.mean()<.5 or ((arrays["mask"]==1)&expected).sum()<9:raise ValueError("geographic supervision below admission")
    models=dict(reserve=[args.runs/f"reserve-s{s}"/"model.pt" for s in (1,2,3)],
        c1mm=[args.reference_root/f"hls-c1mm-final-s{s}"/"model.pt" for s in (1,2,3,4,5)],prithvi=[])
    identities={r["sample_id"].split(":",1)[1] for r in records}
    for name in ("reserve","c1mm"):
        if [sha256_file(p) for p in models[name]]!=parent["model_hashes"][name]:raise ValueError("selected model weights changed")
    for path in models["reserve"]:
        metadata=checkpoint_metadata(path,export_root=Path("research/tensor-exports"))
        if identities.intersection(i.split(":",1)[-1] for i in metadata["fit_ids"]):raise ValueError("geographic labels were fitted")
    # Reconstruct signed original caches before new inference, binding config too.
    old=json.loads((args.data_root/"russia/manifest.json").read_text())["records"]
    completed_predictions(args,parent,[r for r in old if r["partition"]=="evaluation"],"russia")
    sources=execution_sources(Path(__file__),"scripts/prepare_western_holdout.py","src/comp/western_holdout.py",
        "scripts/prepare_russian_windows.py","scripts/score_russian_transfer.py","scripts/report_russian_errors.py",
        "src/comp/l2a_finetuning.py","src/comp/terrain.py","src/comp/severity_layers.py")
    recipe=freeze_recipe(args.out.with_suffix(".recipe.json"),sources,dict(manifest=sha256_file(args.manifest),
        selection=sha256_file(selection),models=parent["model_hashes"],prithvi=parent["competitor_weights_sha256"],
        config=sha256_file(args.prithvi_root/"burn_scars_config.yaml"),threshold=.5,tta=8,bootstrap=2000,seed=20261003))
    probabilities={name:predict(args.data_root,records,paths,name,args.out.parent/(args.out.stem+"-"+name+".npy"),args.prithvi_root)
                   for name,paths in models.items()}
    cm,rows=counts_for(args.data_root,records,probabilities)
    for row,r in zip(rows,records):row["total_pixels"]=r["grid"]["width"]*r["grid"]["height"]
    errors=summarize(rows,list(models));groups=[r["group_id"] for r in records]
    comparisons={name:paired_group_comparison(groups,cm["reserve"],cm[name]) for name in ("c1mm","prithvi")}
    missed={name:len(info["missed_positive_groups"]) for name,info in errors["models"].items()}
    accepted=all(c["delta"]>=.01 and c["ci95"][0]>0 for c in comparisons.values()) and missed["reserve"]<=missed["c1mm"]
    result=dict(spec_id="SPEC-105",accepted=bool(accepted),comparisons=comparisons,missed_positive_groups=missed,
        errors=errors,per_window=rows,recipe_id=recipe,manifest_sha256=sha256_file(args.manifest),
        selection_sha256=sha256_file(selection),model_hashes=parent["model_hashes"],
        non_claims=["Independent geography, shared PANGAEA annotation method and spring2020 interval.",
                    "No field truth/summerSiberian/global ranking/severity/L2A useful-gain proof.",
                    "Spatial groups are not physical fire IDs; regional deployment needs a separate spec."])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_window","errors")}),flush=True)


if __name__=="__main__":main()
