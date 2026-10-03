"""SPEC-91: prepare source-bound six-band pre/post pairs on the data host."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
import planetary_computer
import pystac_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import event_group, sha256_file, validate_manifest, validate_pair
from src.comp.fire_dates import start_of

BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def hls_bad(fmask):
    # Cirrus, cloud, adjacent cloud, shadow, snow; water is handled separately.
    return (fmask == 255) | ((fmask & 31) != 0)


def read_grid(href, grid, mask=False):
    with rasterio.open(href) as source, WarpedVRT(
        source, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
        height=grid["height"], resampling=Resampling.nearest if mask else Resampling.average,
        src_nodata=255 if mask else -9999, nodata=255 if mask else -9999,
    ) as warped:
        return warped.read(1)


def prepare(data, out, dataset, partition, window, source_directory=None):
    source = data / "external" / (source_directory or ("hls_fresh" if dataset == "mtbs" else dataset + "_hls"))
    name = window["name"]; destination = out / dataset / name
    meta_path = Path(str(destination) + ".pair.json")
    if meta_path.exists():
        record = json.loads(meta_path.read_text())
        validate_pair(record)
        for key in ("pre", "post", "mask", "valid"):
            path = data / record["files"][key]
            if sha256_file(path) != record["hashes"][key]: raise ValueError("existing pair input changed")
        return record
    start = start_of(window)
    with rasterio.open(source / (name + "_merged.tif")) as raster:
        grid = dict(crs=raster.crs, transform=raster.transform, width=raster.width, height=raster.height)
        profile = raster.profile
        bbox = list(transform_bounds(raster.crs, 4326, *raster.bounds))
        post_image = raster.read().astype(np.float32)
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                       modifier=planetary_computer.sign_inplace)
    collection = catalog.get_collection("hls2-s30")
    post = collection.get_item(window["item"])
    if post is None: raise ValueError("post item unavailable")
    planetary_computer.sign_inplace(post)
    if post.datetime < start: raise ValueError("post predates ignition")
    tile = post.id.split(".")[2]
    candidates = [it for it in catalog.search(collections=["hls2-s30"], bbox=bbox,
                      datetime=f"{start - timedelta(days=90):%Y-%m-%dT%H:%M:%SZ}/{start:%Y-%m-%dT%H:%M:%SZ}").items()
                  if it.id.split(".")[2] == tile and it.datetime < start]
    pre, pre_image, pre_bad = None, None, None
    for item in sorted(candidates, key=lambda it: it.datetime, reverse=True):
        fm = read_grid(item.assets["Fmask"].href, grid, mask=True)
        bad = hls_bad(fm)
        if bad.mean() > .10: continue
        image = np.stack([read_grid(item.assets[b].href, grid) for b in BANDS]).astype(np.float32)
        bad |= (image == -9999).any(0) | ~np.isfinite(image).all(0)
        if bad.mean() > .10: continue
        pre, pre_image, pre_bad = item, image * .0001, bad
        break
    if pre is None: raise ValueError("no clean pre in same MGRS within 90 days")
    post_bad = hls_bad(read_grid(post.assets["Fmask"].href, grid, mask=True))
    post_bad |= (post_image == -9999).any(0) | ~np.isfinite(post_image).all(0)
    with rasterio.open(source / (name + ".mask.tif")) as raster:
        label = raster.read(1)
        if raster.crs != grid["crs"] or raster.transform != grid["transform"]:
            raise ValueError("annotation grid differs")
    valid = ~(pre_bad | post_bad) & (label >= 0)
    if not valid.any() or not ((label == 1) & valid).any():
        raise ValueError("no labelled fire remains on common valid pixels")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pre_path = Path(str(destination) + ".pre.tif")
    valid_path = Path(str(destination) + ".valid.tif")
    with rasterio.open(pre_path, "w", **dict(profile, dtype="float32", count=6, nodata=-9999)) as raster:
        raster.write(np.where(pre_bad[None], -9999, pre_image).astype(np.float32))
    with rasterio.open(valid_path, "w", **dict(profile, dtype="uint8", count=1, nodata=0)) as raster:
        raster.write(valid.astype(np.uint8)[None])
    serial_grid = dict(crs=str(grid["crs"]), transform=list(grid["transform"])[:6],
                       width=grid["width"], height=grid["height"])
    files = {"pre": str(pre_path.relative_to(data)), "valid": str(valid_path.relative_to(data)),
             "post": str((source / (name + "_merged.tif")).relative_to(data)),
             "mask": str((source / (name + ".mask.tif")).relative_to(data))}
    record = dict(sample_id=dataset + ":" + name, group_id=event_group(dataset, window),
                  partition=partition, label_provenance="expert_perimeter", bbox_wgs84=bbox,
                  grid=serial_grid, pre_grid=serial_grid, mgrs=tile, pre_mgrs=tile,
                  pre_at=pre.datetime.isoformat(), start_at=start.isoformat(),
                  post_at=post.datetime.isoformat(), pre_item=pre.id, post_item=post.id,
                  files=files, hashes={k: sha256_file(data / v) for k, v in files.items()},
                  dataset=dataset, sampling_weight=2 if dataset in ("ems", "floga") else 1,
                  valid_fraction=float(valid.mean()), burned_pixels=int(((label == 1) & valid).sum()))
    validate_pair(record)
    with meta_path.open("x") as stream: json.dump(record, stream, indent=2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="QC probe only; cannot authorize training")
    args = parser.parse_args(); data, out = args.data_root.resolve(), args.out.resolve()
    if not out.is_relative_to(data): raise ValueError("prepared outputs must live under data root")
    out.mkdir(parents=True, exist_ok=True); os.chdir(data)
    jobs = []
    for dataset, directory, manifest, part in [
        ("mtbs", "hls_fresh", "manifest.json", "fit"),
        ("ems", "ems_hls", "manifest_train.json", "fit"),
        ("floga", "floga_hls", "manifest_train.json", "fit"),
        ("ems", "ems_hls", "manifest_test.json", "evaluation"),
        ("floga", "floga_hls", "manifest_test.json", "evaluation"),
    ]:
        windows = json.loads((data / "external" / directory / manifest).read_text())["windows"]
        jobs.extend((dataset, part, window) for window in windows)
    if args.limit: jobs = jobs[:args.limit]
    failures = []
    def run(job):
        for attempt in range(3):
            try:
                result = prepare(data, out, *job)
                print("prepared", result["sample_id"], flush=True); return result
            except ValueError as error:
                failures.append({"sample": job[2]["name"], "reason": str(error)}); return None
            except Exception as error:
                if attempt == 2:
                    failures.append({"sample": job[2]["name"], "reason": repr(error)[:300]}); return None
                time.sleep(5 * (attempt + 1))
    with ThreadPoolExecutor(4) as pool: records = [r for r in pool.map(run, jobs) if r]
    records.sort(key=lambda r: r["sample_id"])
    counts = validate_manifest(records, require_pairs=True)
    fit = [r for r in records if r["partition"] == "fit"]
    ready = (args.limit is None and sum(r["dataset"] == "mtbs" for r in fit) >= 200
             and sum(r["dataset"] in ("ems", "floga") for r in fit) >= 50
             and any(r["dataset"] == "ems" and r["partition"] == "evaluation" for r in records)
             and any(r["dataset"] == "floga" and r["partition"] == "evaluation" for r in records))
    result = dict(spec_id="SPEC-91", records=records, counts=counts, failures=failures,
                  ready_for_training=ready, qc_probe=args.limit is not None)
    target = out / ("qc-manifest.json" if args.limit else "manifest.json")
    with target.open("x") as stream: json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("records", "failures")}), flush=True)
    if not ready and args.limit is None: raise SystemExit(2)


if __name__ == "__main__": main()
