"""SPEC-98: collect a bounded, frozen, previously unused MTBS2024 paired test."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
import requests
from shapely import wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file, validate_manifest
from src.comp.fresh_severity import isolate, assert_fresh, population
from src.comp.paired_severity import mtbs_targets
from scripts.prepare_forest_pairs import prepare, read_grid, hls_bad, BANDS

SEVERITY_URL = "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_MTBS_CONUS/ImageServer/exportImage"
STATES = {"CA", "OR", "WA", "ID", "MT", "CO", "WY", "UT", "NV", "AZ", "NM"}


def immutable(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("immutable collection artifact differs")
    else:
        with path.open("x") as stream: json.dump(payload, stream, indent=2)


def validate_severity(path, grid, identity):
    metadata=path.with_suffix(".source.json")
    if not metadata.exists():raise ValueError("existing severity lacks source binding")
    stored=json.loads(metadata.read_text())
    if stored!={"identity":identity,"sha256":sha256_file(path)}:
        raise ValueError("severity source/hash binding differs")
    with rasterio.open(path) as source:
        if source.count!=1 or source.shape!=(512,512) or str(source.crs)!=grid["crs"] or source.transform!=grid["transform"]:
            raise ValueError("MTBS severity grid differs")
        raw=source.read(1)
    if not np.isin(raw,[0,1,2,3,4,5,6]).all():raise ValueError("unknown MTBS severity encoding")
    return raw


def severity_identity(grid, year=2024):
    if year not in (2016,2017,2020,2021,2022,2023,2024):raise ValueError("year outside frozen severity populations")
    left,bottom,right,top=rasterio.transform.array_bounds(512,512,grid["transform"])
    epsg=rasterio.crs.CRS.from_string(grid["crs"]).to_epsg()
    query=dict(bbox=f"{left},{bottom},{right},{top}",bboxSR=epsg,imageSR=epsg,
               size="512,512",format="tiff",pixelType="U8",interpolation="RSP_NearestNeighbor",
               mosaicRule=json.dumps({"mosaicMethod":"esriMosaicNone","where":f"name='mtbs_CONUS_{year}'"}),f="image")
    return dict(url=SEVERITY_URL,query=query)


def validate_cached_record(root, record, event):
    if (record["group_id"]!="mtbs:"+event["event_id"] or record["grid"]!=event["grid"]
            or record["partition"]!="evaluation" or record.get("severity_source")!=SEVERITY_URL):
        raise ValueError("cached record differs from frozen event/source")
    if "fmask" not in record["files"]:raise ValueError("cached record lacks frozen water Fmask")
    for key,relative in record["files"].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256_file(path)!=record["hashes"][key]:
            raise ValueError("cached record input changed or escaped own data root")
    grid=dict(event["grid"],transform=rasterio.Affine(*event["grid"]["transform"]))
    raw=validate_severity(root/record["files"]["severity"],grid,severity_identity(grid,int(event.get("ig_date","2024")[:4])))
    with rasterio.open(root/record["files"]["valid"]) as source:
        if source.shape!=(512,512) or str(source.crs)!=grid["crs"] or source.transform!=grid["transform"]:
            raise ValueError("cached common-valid mask grid differs")
        valid=source.read(1).astype(bool)
    targets=mtbs_targets(raw,valid)
    counts=np.bincount(targets[targets>=0],minlength=3).tolist()
    if counts!=record["severity_counts"]:raise ValueError("cached class population differs from raster")


def freeze_plan(source, prior, out, profile="2024-west"):
    spec_id,years,prefix=population(profile)
    path = out / "selection.json"
    vector = source / "external/mtbs/mtbs_perims_DD.shp"
    hashes = {p.name: sha256_file(p) for p in sorted(vector.parent.glob("mtbs_perims_DD.*"))}
    used, manifests = set(), {}
    for name in ("hls_fresh", "hls_fresh2", "hls_fresh3", "hls_fresh4"):
        p = source / "external" / name / "manifest.json"
        manifests[name] = sha256_file(p)
        used.update(w["event_id"] for w in json.loads(p.read_text())["windows"])
    bindings = dict(vector_hashes=hashes, old_manifests=manifests, prior_sha256=sha256_file(prior))
    if profile=="pc-covered-2020-24":
        coverage=out/"source-coverage.json"
        if not coverage.exists():
            url="https://planetarycomputer.microsoft.com/api/stac/v1/collections/hls2-s30"
            response=requests.get(url,timeout=30);response.raise_for_status()
            immutable(coverage,dict(url=url,temporal=response.json()["extent"]["temporal"]["interval"]))
        observed=json.loads(coverage.read_text())
        if not observed["temporal"] or observed["temporal"][0][0]>"2020-01-01T00:00:00Z":
            raise ValueError("HLS source does not cover frozen2020 start")
        bindings["hls_coverage_sha256"]=sha256_file(coverage)
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["bindings"] != bindings or plan.get("profile","2024-west")!=profile:
            raise ValueError("selection inputs/population changed")
        return plan
    frame = gpd.read_file(vector)
    dates = pd.to_datetime(frame.ig_date)
    regional=frame.event_id.str[:2].isin(STATES) if profile=="2024-west" else ~frame.event_id.str[:2].isin({"AK","HI","PR"})
    frame = frame[dates.dt.year.isin(years) & regional].copy()
    projected = frame.to_crs(5070).bounds
    frame = frame[((projected.maxx-projected.minx) <= 14000) & ((projected.maxy-projected.miny) <= 14000)]
    frame["order"] = frame.event_id.map(lambda e: hashlib.sha256((prefix+e).encode()).hexdigest())
    candidates = []
    for row in frame.sort_values("order").itertuples():
        geom = gpd.GeoSeries([row.geometry], crs=frame.crs)
        lon=geom.to_crs(4326).iloc[0].centroid.x
        crs = rasterio.crs.CRS.from_epsg(32600+int((lon+180)//6)+1)
        polygon = geom.to_crs(crs).iloc[0]; c = polygon.centroid
        left = np.floor(c.x/30)*30-256*30; top = np.floor(c.y/30)*30+256*30
        transform = from_origin(left, top, 30, 30)
        grid = dict(crs=str(crs), transform=list(transform)[:6], width=512, height=512)
        bounds = list(transform_bounds(crs, 4326, left, top-512*30, left+512*30, top))
        candidates.append(dict(event_id=row.event_id, ig_date=str(pd.Timestamp(row.ig_date).date()),
                               grid=grid, bbox_wgs84=bounds, polygon_wkt=polygon.wkt))
    cap = 240 if profile == "pc-covered-2020-24" else 180
    selected, excluded = isolate(candidates, used, json.loads(prior.read_text())["records"],limit=cap)
    plan = dict(spec_id=spec_id,profile=profile, bindings=bindings, used_events=sorted(used), selected=selected,
                excluded=excluded, eligible=len(candidates), cap=cap)
    immutable(path, plan)
    return plan


def collect(root, out, event):
    if shutil.disk_usage(root).free < 8*1024**3:
        raise RuntimeError("less than8GiB free; preserve data and stop collection")
    inputs = root / "external" / ("hls_"+out.name)
    inputs.mkdir(parents=True, exist_ok=True)
    if not inputs.resolve().is_relative_to(root): raise ValueError("outputs escape own data root")
    grid = dict(event["grid"], transform=rasterio.Affine(*event["grid"]["transform"]))
    start = pd.Timestamp(event["ig_date"], tz="UTC")
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    items = list(catalog.search(collections=["hls2-s30"], bbox=event["bbox_wgs84"],
                 datetime=f"{(start+pd.Timedelta(days=30)).date()}/{(start+pd.Timedelta(days=150)).date()}").items())
    ranked = []
    for item in sorted(items, key=lambda i: i.id):
        fm = read_grid(item.assets["Fmask"].href, grid, mask=True)
        bad = hls_bad(fm)
        if bad.mean() < .10: ranked.append((float(bad.mean()), item.datetime, item.id, item, fm))
    if not ranked: raise ValueError("no clean post window")
    _, dt, _, item, fm = min(ranked, key=lambda x:x[:3])
    name = f"fresh_{event['event_id']}_{item.id}"
    window = dict(name=name, event_id=event["event_id"], ig_date=event["ig_date"], item=item.id,
                  item_date=str(dt.date()))
    image = np.stack([read_grid(item.assets[b].href, grid) for b in BANDS]).astype(np.float32)
    bad = hls_bad(fm) | (image == -9999).any(0) | ~np.isfinite(image).all(0)
    if bad.mean() >= .10: raise ValueError("post reflectance nodata fails QC")
    labels = rasterize([(wkt.loads(event["polygon_wkt"]),1)], out_shape=(512,512), transform=grid["transform"], dtype="int16")
    labels[bad] = -1
    profile = dict(driver="GTiff", crs=grid["crs"], transform=grid["transform"], width=512, height=512, compress="deflate")
    arrays = {"_merged.tif": (np.where(bad[None],-9999,image*.0001).astype(np.float32), -9999),
              ".mask.tif": (labels[None], -1), ".fmask.tif": (fm[None].astype(np.uint8), 255)}
    for suffix,(array,nodata) in arrays.items():
        path = inputs/(name+suffix)
        if path.exists():
            with rasterio.open(path) as old:
                if not np.array_equal(old.read(),array): raise ValueError("existing post differs")
        else:
            with rasterio.open(path,"w",**profile,count=array.shape[0],dtype=array.dtype,nodata=nodata) as target:
                target.write(array)
    record = prepare(root,out,"mtbs","evaluation",window,source_directory=inputs.name)
    severity = inputs/(name+".sev.tif")
    identity=severity_identity(grid,int(event["ig_date"][:4]))
    if not severity.exists():
        response = requests.get(SEVERITY_URL,params=identity["query"],timeout=120);response.raise_for_status()
        with rasterio.open(io.BytesIO(response.content)) as source:
            if source.shape!=(512,512) or str(source.crs)!=grid["crs"] or source.transform!=grid["transform"]:
                raise ValueError("MTBS response grid differs")
            raw = source.read(1)
        if not np.isin(raw,[0,1,2,3,4,5,6]).all():raise ValueError("unknown MTBS severity encoding")
        with rasterio.open(severity,"w",**profile,count=1,dtype="uint8") as target:target.write(raw[None].astype(np.uint8))
        immutable(severity.with_suffix(".source.json"),dict(identity=identity,sha256=sha256_file(severity)))
    raw=validate_severity(severity,grid,identity)
    with rasterio.open(root/record["files"]["valid"]) as source: valid = source.read(1).astype(bool)
    targets = mtbs_targets(raw,valid);counts=np.bincount(targets[targets>=0],minlength=3)
    if not counts.sum():raise ValueError("no published severity on common-valid fire")
    fmask_path=inputs/(name+".fmask.tif")
    return dict(record, label_provenance="mtbs_severity", severity_counts=counts.tolist(),
                severity_source=SEVERITY_URL, files=dict(record["files"],severity=str(severity.relative_to(root)),
                                                       fmask=str(fmask_path.relative_to(root))),
                hashes=dict(record["hashes"],severity=sha256_file(severity),fmask=sha256_file(fmask_path)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data-root","source-root","prior","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--limit",type=int,help="QC only")
    p.add_argument("--profile",choices=("2024-west","retrospective-2016-17","pc-covered-2020-24"),default="2024-west")
    args=p.parse_args();root=args.data_root.resolve();out=args.out.resolve()
    if not out.is_relative_to(root):raise ValueError("outputs escape data root")
    out.mkdir(parents=True,exist_ok=True)
    destination=out/("qc-manifest.json" if args.limit else "manifest.json")
    if destination.exists():raise ValueError("immutable collection already exists")
    spec_id,_,_=population(args.profile)
    plan=freeze_plan(args.source_root.resolve(),args.prior,out,args.profile);events=plan["selected"]
    if len(events)<100 and not args.limit:
        payload=dict(spec_id=spec_id,profile=args.profile,records=[],counts=dict(samples=0,groups=0),
                     failures=[dict(reason="insufficient independent candidate population",selected=len(events),minimum=100)],
                     ready_for_evaluation=False,qc_probe=False,selection_sha256=sha256_file(out/"selection.json"))
        immutable(destination,payload);print("RESULT",json.dumps(payload),flush=True);raise SystemExit(2)
    if args.limit:events=events[:args.limit]
    def run(event):
        record_path=out/(event["event_id"]+".record.json")
        for attempt in range(3):
            try:
                if record_path.exists():
                    record=json.loads(record_path.read_text());validate_cached_record(root,record,event)
                    return record,None
                record=collect(root,out,event);immutable(record_path,record)
                print("prepared",event["event_id"],flush=True);return record,None
            except (ValueError,RuntimeError) as error:return None,dict(event_id=event["event_id"],reason=str(error))
            except Exception as error:
                if attempt==2:return None,dict(event_id=event["event_id"],reason=type(error).__name__)
                time.sleep(5*(attempt+1))
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(run,events))
    records=sorted([r for r,e in results if r],key=lambda r:r["sample_id"])
    assert_fresh(records,plan)
    counts=validate_manifest(records,require_pairs=True) if records else dict(samples=0,groups=0)
    classes=np.sum([r["severity_counts"] for r in records],axis=0).tolist() if records else [0,0,0]
    ready=not args.limit and counts["groups"]>=100 and all(c>0 for c in classes)
    payload=dict(spec_id=spec_id,profile=args.profile,records=records,counts=counts,class_counts=classes,
                 failures=[e for r,e in results if e],ready_for_evaluation=ready,qc_probe=bool(args.limit),
                 selection_sha256=sha256_file(out/"selection.json"))
    immutable(destination,payload);print("RESULT",json.dumps({k:v for k,v in payload.items() if k not in ("records","failures")}),flush=True)
    if not ready and not args.limit:raise SystemExit(2)


if __name__=="__main__":main()
