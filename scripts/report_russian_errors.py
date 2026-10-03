"""SPEC-92: CPU-only error/seed reports from completed signed prediction caches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import canonical_hash,sha256_file
from src.comp.l2a_finetuning import l2a_water
from src.comp.russian_error_report import summarize,diagnostic_subset,size_bin
from scripts.forest_pair_train import read_sample
from scripts.score_russian_transfer import predict,counts_for


def completed_predictions(args,parent,records,dataset):
    models={v:[args.runs/f"{v}-s{s}"/"model.pt" for s in (1,2,3)] for v in ("control","reserve")}
    models["c1mm"]=[args.reference_root/f"hls-c1mm-final-s{s}"/"model.pt" for s in (1,2,3,4,5)]
    for name,paths in models.items():
        if [sha256_file(p) for p in paths]!=parent["model_hashes"][name]:
            raise ValueError("diagnostic models differ from completed parent result")
    if sha256_file(args.prithvi_root/"Prithvi_EO_V2_300M_BurnScars.pt")!=parent["competitor_weights_sha256"]:
        raise ValueError("Prithvi weights differ")
    if dataset=="russia":models["prithvi"]=[]
    probabilities={}
    for name,paths in models.items():
        stem=args.parent.stem+("-"+dataset if dataset!="russia" else "")
        cache=args.parent.parent/(stem+"-"+name+".npy")
        if not cache.exists() or not cache.with_name(cache.name+".binding.json").exists():
            raise ValueError("required completed cache is missing; no inference permitted")
        probabilities[name]=predict(args.data_root,records,paths,name,cache,args.prithvi_root)
    return probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","russian-manifest","base-manifest","parent","runs","reference-root","prithvi-root","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("immutable error report already exists")
    parent=json.loads(args.parent.read_text())
    if parent["spec_id"]!="SPEC-92":raise ValueError("wrong result")
    for name,path in (("russian_manifest",args.russian_manifest),("base_manifest",args.base_manifest)):
        if sha256_file(path)!=parent[name]:raise ValueError("source manifest changed")
    russian=json.loads(args.russian_manifest.read_text())["records"]
    base=json.loads(args.base_manifest.read_text())["records"]
    selected=[r for r in russian if r["partition"]=="evaluation"]
    archive=args.data_root/"russia/burned_area_2020_topology.zip"
    source=json.loads((archive.parent/"source_manifest.json").read_text())
    if sha256_file(archive)!=source["sha256"]:raise ValueError("source contours changed")
    contours=gpd.read_file("zip://"+str(archive)).to_crs(4326)
    contours=contours[contours.geometry.notna()&~contours.geometry.is_empty&contours.geometry.is_valid]
    contours=contours[contours.intersects(box(80,50,115,65))]
    geometries={"russia:"+canonical_hash(g.wkb_hex):g for g in contours.geometry}
    datasets={};seeds=[]
    for dataset in ("russia","ems","floga"):
        records=selected if dataset=="russia" else [r for r in base if r["partition"]=="evaluation" and r["dataset"]==dataset]
        probabilities=completed_predictions(args,parent,records,dataset)
        _,rows=counts_for(args.data_root,records,probabilities)
        if dataset=="russia" and rows!=parent["per_window"]:
            raise ValueError("reconstructed counts differ from parent result")
        for row,record in zip(rows,records):
            row["total_pixels"]=record["grid"]["width"]*record["grid"]["height"]
        datasets[dataset]=dict(all=summarize(rows,list(probabilities)),
                              reference_top25pct_fp=dict(sample_ids=[r["sample_id"] for r in diagnostic_subset(rows)],
                                                        report=summarize(diagnostic_subset(rows),list(probabilities))))
        if dataset!="russia":continue
        for i,record in enumerate(records):
            polygon=gpd.GeoSeries([geometries[record["sample_id"]]],crs=4326).to_crs(record["grid"]["crs"]).iloc[0]
            image,label,valid=read_sample(args.data_root,record,"post")
            mask=rasterize([(polygon,1)],out_shape=valid.shape,transform=rasterio.Affine(*record["grid"]["transform"]),dtype="uint8").astype(bool)
            observed=mask&valid&(label==1)
            with rasterio.open(args.data_root/record["files"]["post_scl"]) as raster:scl=raster.read(1)
            water=l2a_water(image,scl)
            row=dict(sample_id=record["sample_id"],group_id=record["group_id"],area_ha=record["area_ha"],
                     size_bin=size_bin(record["area_ha"]),observed_seed_pixels=int(observed.sum()),
                     masked_fraction=float(1-valid.mean()))
            row["tp"]={name:int(((probs[i]>=.5)&~water&observed).sum()) for name,probs in probabilities.items()}
            row["status"]="observable" if row["observed_seed_pixels"] else "unobservable"
            seeds.append(row)
    summary={}
    for size in ("all","1-25ha","25-250ha","250ha+"):
        rows=seeds if size=="all" else [r for r in seeds if r["size_bin"]==size]
        observed=[r for r in rows if r["status"]=="observable"]
        summary[size]=dict(selected_contours=len(rows),observable_contours=len(observed),
                           unobservable_contours=len(rows)-len(observed),
                           missed_contours={m:sum(r["tp"][m]==0 for r in observed) for m in ("control","reserve","c1mm","prithvi")})
    result=dict(spec_id="SPEC-92",parent_result_sha256=sha256_file(args.parent),source_archive_sha256=sha256_file(archive),
                datasets=datasets,seed_summary=summary,per_seed=seeds,
                source_hashes={p:sha256_file(p) for p in ("scripts/report_russian_errors.py","src/comp/russian_error_report.py",
                    "scripts/score_russian_transfer.py","src/comp/l2a_finetuning.py","src/comp/terrain.py","src/comp/severity_layers.py")},
                non_claims=["Russian groups are spatial clusters, not physical fire IDs.",
                            "Missed source contours/windows do not prove a count of physical fires.",
                            "Reference-defined hard subset is diagnostic; it cannot select a new model/threshold.",
                            "Satellite interpretation and fixed common observation mask, not field truth."])
    with args.out.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("per_seed","source_hashes")}),flush=True)


if __name__=="__main__":main()
