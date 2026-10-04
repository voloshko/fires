"""SPEC-101: preserve every original fit record and freeze optional PC twins."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file,validate_manifest
from src.comp.fresh_severity import overlaps
from src.comp.l2a_finetuning import EXPECTED_BANDS,l2a_valid
from src.comp.l2a_full_replay import original_valid,REFERENCE_RESULT_SHA256
from scripts.prepare_l2a_finetuning import source_jobs


def verify_twin(root,record,job,image,labels,valid,grid):
    if record["provider"]!="Planetary Computer" or record["collection"]!="sentinel-2-l2a":
        raise ValueError("twin is not from the authorized PC bank")
    if record["sample_id"]!=job["sample_id"] or record["partition"]!=job["partition"] or record["grid"]!=grid:
        raise ValueError("twin source identity/grid/partition differs")
    if record["files"]["hls"]!=job["source"]:
        raise ValueError("twin points at another original source")
    for key,path in record["files"].items():
        if sha256_file(root/path)!=record["hashes"][key]:raise ValueError("twin input changed")
    match=re.search(r"HLS\.S30\.(T[0-9A-Z]+)\.(\d{7})",Path(job["source"]).name)
    if match is None:raise ValueError("missing original acquisition metadata")
    day=datetime.strptime(match[2],"%Y%j").date()
    if datetime.fromisoformat(record["acquisition"].replace("Z","+00:00")).date()!=day:
        raise ValueError("twin date differs")
    if "_"+match[1]+"_" not in record["l2a_item"]:raise ValueError("twin MGRS differs")
    if set(record["encodings"])!=set(EXPECTED_BANDS):raise ValueError("physical band inventory differs")
    for key,expected in EXPECTED_BANDS.items():
        encoding=record["encodings"][key]
        if encoding["band"]!=expected or not np.isfinite([encoding["scale"],encoding["offset"]]).all() or encoding["scale"]<=0:
            raise ValueError("invalid physical band encoding")
        if not encoding["encoding_basis"].startswith(("ESA-L2A-processing-baseline:","STAC-raster:bands")):
            raise ValueError("unknown L2A encoding basis")
    arrays={}
    for key in ("l2a","scl","valid","mask"):
        with rasterio.open(root/record["files"][key]) as source:
            if (str(source.crs)!=grid["crs"] or list(source.transform)[:6]!=grid["transform"]
                    or source.shape!=(grid["height"],grid["width"])):
                raise ValueError("twin raster grid differs")
            arrays[key]=source.read() if key=="l2a" else source.read(1)
    if not np.array_equal(arrays["mask"],labels):raise ValueError("twin annotation differs")
    expected=l2a_valid(arrays["l2a"],arrays["scl"],labels)&valid
    if not np.array_equal(expected,arrays["valid"].astype(bool)):raise ValueError("cached common mask differs from strict SCL")
    return dict(record,common_pixels=int(expected.sum()),common_positive_pixels=int(((labels==1)&expected).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","event-registry","twins","reference-root","reference-result","out"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args();root=args.data_root.resolve();out=args.out.resolve()
    if not out.is_relative_to(root):raise ValueError("outputs escape own data root")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():raise ValueError("immutable manifest already exists")
    if sha256_file(args.reference_result)!=REFERENCE_RESULT_SHA256:raise ValueError("C1-MM reference result differs")
    reference=json.loads(args.reference_result.read_text())
    initializers=[]
    for seed,expected in enumerate(reference["model_hashes"]["c1mm"][:3],1):
        path=args.reference_root/f"hls-c1mm-final-s{seed}"/"model.pt"
        if sha256_file(path)!=expected:raise ValueError("C1-MM initializer changed")
        initializers.append(dict(seed=seed,path=str(path),sha256=expected))
    bank=json.loads(args.twins.read_text())
    if bank["qc_probe"] or bank["spec_id"]!="SPEC-93" or bank["event_registry_sha256"]!=sha256_file(args.event_registry):
        raise ValueError("wrong or provisional twin bank")
    lookup={r["sample_id"]:r for r in bank["records"]}
    if len(lookup)!=len(bank["records"]):raise ValueError("duplicate twin bank IDs")
    jobs=source_jobs(root,args.event_registry)
    fitjobs=[j for j in jobs if j["partition"]=="fit"]
    expected=dict(hls=540,mtbs=300,ems=49,floga=40)
    if dict(Counter(j["dataset"] for j in fitjobs))!=expected:raise ValueError("full original population changed")
    records=[]
    for job in fitjobs:
        image_path=root/job["source"];label_path=Path(str(image_path).replace("_merged.tif",".mask.tif"))
        with rasterio.open(image_path) as source,rasterio.open(label_path) as annotation:
            if source.count!=6 or source.shape!=(512,512) or source.crs!=annotation.crs or source.transform!=annotation.transform:
                raise ValueError("original input/annotation grid differs")
            image=source.read();labels=annotation.read(1);valid=original_valid(image,labels)
            grid=dict(crs=str(source.crs),transform=list(source.transform)[:6],width=source.width,height=source.height)
            bounds=list(transform_bounds(source.crs,4326,*source.bounds))
            mask_path=out/(job["sample_id"].replace(":","_")+".valid.tif")
            if mask_path.exists():
                with rasterio.open(mask_path) as stored:
                    if not np.array_equal(stored.read(1),valid):raise ValueError("original valid mask changed")
            else:
                with rasterio.open(mask_path,"w",**dict(source.profile,count=1,dtype="uint8",nodata=0,compress="deflate")) as target:
                    target.write(valid.astype(np.uint8)[None])
        files=dict(hls=job["source"],mask=str(label_path.relative_to(root)),original_valid=str(mask_path.relative_to(root)))
        record=dict(job,grid=grid,bbox_wgs84=bounds,files=files,hashes={k:sha256_file(root/v) for k,v in files.items()},
                    original_pixels=int(valid.sum()),original_positive_pixels=int(((labels==1)&valid).sum()),twin=None)
        if job["sample_id"] in lookup:
            record["twin"]=verify_twin(root,lookup[job["sample_id"]],job,image,labels,valid,grid)
        records.append(record)
    fit=sorted(records,key=lambda r:r["sample_id"]);evaluation=[];excluded=[]
    fitgroups={r["group_id"] for r in fit}
    for job in jobs:
        if job["partition"]!="evaluation":continue
        twin=lookup.get(job["sample_id"])
        if twin is None:
            excluded.append(dict(sample_id=job["sample_id"],reason="missing or QC-rejected PC twin"));continue
        if twin["group_id"] in fitgroups or any(overlaps(twin["bbox_wgs84"],r["bbox_wgs84"]) for r in fit):
            excluded.append(dict(sample_id=job["sample_id"],reason="intersects full original fit"));continue
        with rasterio.open(root/twin["files"]["hls"]) as source,rasterio.open(root/twin["files"]["mask"]) as mask:
            image,labels=source.read(),mask.read(1)
            valid=original_valid(image,labels)
        verify_twin(root,twin,job,image,labels,valid,twin["grid"])
        evaluation.append(twin)
    evaluation.sort(key=lambda r:r["sample_id"])
    validate_manifest(fit+evaluation)
    twinned=[r for r in fit if r["twin"] is not None]
    groups={ds:len({r["group_id"] for r in evaluation if r["dataset"]==ds}) for ds in ("ems","floga")}
    ready=len(twinned)>=300 and len(evaluation)>=50 and sum(groups.values())>=20 and all(n>=10 for n in groups.values())
    payload=dict(spec_id="SPEC-101",records=fit+evaluation,full_original_fit_count=len(fit),fit_twins=len(twinned),
                 fallback_ids=[r["sample_id"] for r in fit if r["twin"] is None],evaluation_exclusions=excluded,
                 evaluation_count=len(evaluation),evaluation_groups=groups,initializers=initializers,
                 twin_bank_sha256=sha256_file(args.twins),event_registry_sha256=sha256_file(args.event_registry),
                 reference_result_sha256=sha256_file(args.reference_result),ready_for_replay=ready,
                 original_supervised_pixels=sum(r["original_pixels"] for r in fit),
                 common_supervised_pixels=sum(r["twin"]["common_pixels"] for r in twinned),
                 source_hashes={str(p):sha256_file(p) for p in (Path(__file__),Path("src/comp/l2a_full_replay.py"),Path("scripts/prepare_l2a_finetuning.py"),Path("src/comp/l2a_finetuning.py"))},
                 non_claims=["Original samples/supervision retained through HLS replay, not all events have L2A exposure.",
                             "Does not turn SPEC93/97 event-retention gate into PASS."])
    with (out/"manifest.json").open("x") as stream:json.dump(payload,stream,indent=2)
    print("RESULT",json.dumps({k:v for k,v in payload.items() if k not in ("records","fallback_ids","evaluation_exclusions")}),flush=True)
    if not ready:raise SystemExit(2)


if __name__=="__main__":main()
