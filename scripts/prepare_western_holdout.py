"""SPEC-105: freeze a whole-group-purged geographic holdout before imagery."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import shutil

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
import planetary_computer
import pystac_client
from pyproj import Transformer
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import canonical_hash,sha256_file,validate_manifest
from src.comp.western_holdout import balanced_selection,exclude_groups,StorageBudget,BUDGET_BYTES,storage_guard,verify_exclusions
from scripts.build_russian_reserve import spatial_clusters
from scripts.prepare_russian_windows import choose,interval_pair_valid

SOURCE_SHA="5f1cfb843d5579d800a1e8359a017f8cd33c5658f6e0c0a6eda7ac534a1c6b3c"


def write_once(path,payload):
    if path.exists():
        if json.loads(path.read_text())!=payload:raise ValueError("immutable geographic artifact changed")
    else:
        with path.open("x") as stream:json.dump(payload,stream,indent=2)


def frame(record,geometry):
    point=geometry.representative_point();epsg=32600+int((point.x+180)//6)+1
    x,y=Transformer.from_crs(4326,epsg,always_xy=True).transform(point.x,point.y)
    left,top=np.floor((x-7680)/30)*30,np.ceil((y+7680)/30)*30
    transform=from_origin(left,top,30,30)
    return dict(record,partition="evaluation",dataset="russia",sampling_weight=1,
        label_provenance="participatory_perimeter",center=[point.x,point.y],
        grid=dict(crs=f"EPSG:{epsg}",transform=list(transform)[:6],width=512,height=512),
        bbox_wgs84=list(transform_bounds(epsg,4326,left,top-15360,left+15360,top)))


def make_plan(root,out):
    archive=root/"russia/burned_area_2020_topology.zip"
    if sha256_file(archive)!=SOURCE_SHA:raise ValueError("source ZIP changed")
    paths=[root/"prior-footprints.json",root/"russia/manifest.json",root/"severity-covered-v1/manifest.json"]
    bindings={str(p.relative_to(root)):sha256_file(p) for p in paths}
    bindings["source_zip"]=SOURCE_SHA
    histories={};audit={}
    for p in paths:
        name=str(p.relative_to(root));payload=json.loads(p.read_text())
        audit[name]=verify_exclusions(name,payload,bindings[name]);histories[name]=payload
    contours=gpd.read_file("zip://"+str(archive),bbox=(65,54,80,65))
    if str(contours.crs)!="EPSG:4326":raise ValueError("source CRS differs")
    contours=contours[contours.geometry.notna()&~contours.geometry.is_empty&contours.geometry.is_valid]
    contours=contours[contours.intersects(box(65,54,80,65))].copy()
    geometries={"western:"+canonical_hash(g.wkb_hex):g for g in contours.geometry}
    if len(geometries)!=len(contours):raise ValueError("duplicate source geometries")
    metric=contours.to_crs(3576);groups=spatial_clusters(list(metric.geometry))
    footprints=[]
    for p in paths:
        footprints.extend(r["bbox_wgs84"] for r in histories[str(p.relative_to(root))]["records"])
    projected=gpd.GeoSeries([box(*b) for b in footprints],crs=4326).to_crs(3576)
    tree=STRtree(list(projected));records=[];bad=set()
    for geometry,projected_geometry,group in zip(contours.geometry,metric.geometry,groups):
        identity="western:"+canonical_hash(geometry.wkb_hex)
        record=dict(sample_id=identity,group_id="western:"+group,cluster_id=group,area_ha=projected_geometry.area/10000)
        records.append(record)
        if len(tree.query(projected_geometry.buffer(20000),predicate="intersects")):bad.add(identity)
    eligible,rejected=exclude_groups(records,bad)
    # Check proposed chip extents too, and purge their whole groups before cap.
    provisional=balanced_selection(eligible,cap=len(eligible)*4)
    frames=[frame(r,geometries[r["sample_id"]]) for r in provisional];bad_frames=set()
    for r in frames:
        projected_frame=gpd.GeoSeries([box(*r["bbox_wgs84"])],crs=4326).to_crs(3576).iloc[0]
        if len(tree.query(projected_frame.buffer(20000),predicate="intersects")):bad_frames.add(r["sample_id"])
    eligible,frame_rejected=exclude_groups(eligible,bad_frames)
    chosen=[frame(r,geometries[r["sample_id"]]) for r in balanced_selection(eligible)]
    count_groups=len({r["group_id"] for r in chosen})
    plan=dict(spec_id="SPEC-105",bindings=bindings,records=chosen,source_contours=len(records),
        exclusion_audit=audit,
        original_groups=len(set(groups)),excluded_groups=sorted(rejected|frame_rejected),
        eligible_groups=len({r["group_id"] for r in eligible}),selected_groups=count_groups,
        ready_for_collection=len(chosen)>=50 and count_groups>=20,budget_bytes=BUDGET_BYTES,
        policy="EPSG3576/20km/whole-group purge/96max/4pergroup/metadata-only order")
    write_once(out/"selection-v2.json",plan)
    return plan,contours


def collect(root,out,record,contours,budget):
    metadata=out/(record["sample_id"].replace(":","_")+".json")
    if metadata.exists():
        old=json.loads(metadata.read_text());interval_pair_valid(old)
        for key in ("sample_id","group_id","grid","bbox_wgs84","partition","center"):
            if old[key]!=record[key]:raise ValueError("cached window differs from frozen selection")
        for key,path in old["files"].items():
            if sha256_file(root/path)!=old["hashes"][key]:raise ValueError("old geographic window changed")
        return old
    budget.reserve()
    try:
        catalog=pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
        post,xp,sp,vp,ep=choose(catalog,record,2020)
        tile=post.properties["s2:mgrs_tile"]
        pre,xb,sb,vb,eb=choose(catalog,record,2019,tile=tile,target_date=post.datetime.replace(year=2019))
        valid=vp&vb
        if valid.mean()<.5:raise ValueError("less than half commonly valid")
        # Chips can extend beyond the selection bbox. All published polygons in
        # their actual frame must label them, including those outside65–80/54–65.
        nearby=gpd.read_file("zip://"+str(root/"russia/burned_area_2020_topology.zip"),bbox=tuple(record["bbox_wgs84"]))
        if str(nearby.crs)!="EPSG:4326":raise ValueError("annotation CRS differs")
        nearby=nearby[nearby.geometry.notna()&~nearby.geometry.is_empty&nearby.geometry.is_valid].to_crs(record["grid"]["crs"])
        tr=rasterio.Affine(*record["grid"]["transform"])
        truth=rasterize([(g,1) for g in nearby.geometry],out_shape=(512,512),transform=tr,fill=0,dtype="int16")
        if ((truth==1)&valid).sum()<9:raise ValueError("fewer than nine labelled fire pixels")
        values=dict(pre=(np.where(vb[None],xb,-9999),"float32",-9999),post=(np.where(vp[None],xp,-9999),"float32",-9999),
            pre_scl=(sb[None],"uint8",0),post_scl=(sp[None],"uint8",0),valid=(valid.astype(np.uint8)[None],"uint8",0),
            mask=(np.where(valid,truth,-1)[None],"int16",-1))
        files={}
        for key,(array,dtype,nodata) in values.items():
            path=out/(record["sample_id"].replace(":","_")+"."+key+".tif")
            if path.exists():raise ValueError("partial original file preserved; cannot overwrite")
            with budget.lock:
                storage_guard(budget.used(),budget.reserved,shutil.disk_usage(out).free)
                with rasterio.open(path,"w",driver="GTiff",height=512,width=512,count=len(array),dtype=dtype,
                    crs=record["grid"]["crs"],transform=tr,compress="deflate",nodata=nodata) as target:
                    target.write(array.astype(dtype))
            files[key]=str(path.relative_to(root))
        result=dict(record,files=files,hashes={k:sha256_file(root/p) for k,p in files.items()},
            ignition_interval=["2020-01-01T00:00:00Z","2020-05-16T00:00:00Z"],pre_at=pre.datetime.isoformat(),post_at=post.datetime.isoformat(),
            pre_item=pre.id,post_item=post.id,mgrs=tile,pre_mgrs=pre.properties["s2:mgrs_tile"],pre_grid=record["grid"],
            encodings=dict(pre=eb,post=ep),date_semantics="annual-interval-pair",source_provenance="participatory_perimeter",
            valid_fraction=float(valid.mean()),fire_pixels=int(((truth==1)&valid).sum()))
        interval_pair_valid(result)
        with budget.lock:
            storage_guard(budget.used(),budget.reserved,shutil.disk_usage(out).free)
            write_once(metadata,result)
        print("prepared",result["sample_id"],result["fire_pixels"],flush=True);return result
    finally:budget.release()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--plan-only",action="store_true");p.add_argument("--limit",type=int,help="QC only")
    args=p.parse_args();root=args.data_root.resolve();out=args.out.resolve()
    if not out.is_relative_to(root):raise ValueError("output escapes data root")
    out.mkdir(parents=True,exist_ok=True)
    plan,contours=make_plan(root,out)
    print("PLAN",json.dumps({k:v for k,v in plan.items() if k not in ("records","excluded_groups")}),flush=True)
    if args.plan_only:
        if not plan["ready_for_collection"]:raise SystemExit(2)
        return
    destination=out/("qc-manifest.json" if args.limit else "manifest.json")
    if destination.exists():raise ValueError("immutable collector result exists")
    if not plan["ready_for_collection"]:raise ValueError("geographic selection lacks50windows/20groups")
    jobs=plan["records"][:args.limit] if args.limit else plan["records"];budget=StorageBudget(out)
    def run(record):
        try:return collect(root,out,record,contours,budget),None
        except Exception as error:return None,dict(sample_id=record["sample_id"],reason=type(error).__name__+": "+str(error)[:200])
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(run,jobs))
    records=sorted([r for r,e in results if r],key=lambda r:r["sample_id"])
    counts=validate_manifest(records) if records else dict(samples=0,groups=0)
    ready=not args.limit and counts["samples"]>=50 and counts["groups"]>=20
    payload=dict(spec_id="SPEC-105",records=records,counts=counts,failures=[e for r,e in results if e],
        ready_for_evaluation=ready,qc_probe=bool(args.limit),selection_file="selection-v2.json",
        selection_sha256=sha256_file(out/"selection-v2.json"),storage_bytes=budget.used())
    write_once(destination,payload);print("RESULT",json.dumps({k:v for k,v in payload.items() if k not in ("records","failures")}),flush=True)
    if not ready and not args.limit:raise SystemExit(2)


if __name__=="__main__":main()
