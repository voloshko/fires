"""SPEC-92: frozen interval-labelled Russian windows, PC L2A on both dates."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
import planetary_computer
import pystac_client
from pyproj import Transformer
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import canonical_hash, sha256_file, utc_time, validate_manifest
from src.comp.l2a_finetuning import EXPECTED_BANDS, SCL_INVALID, pc_asset_encoding, asset_encoding


def round_robin(rows, limit):
    bins = [deque(sorted((r for r in rows if (0 if r["area_ha"] < 25 else 1 if r["area_ha"] < 250 else 2) == i),
                         key=lambda r: canonical_hash("russia-windows:" + r["sample_id"]))) for i in range(3)]
    chosen = []
    while len(chosen) < limit and any(bins):
        for bucket in bins:
            if bucket and len(chosen) < limit: chosen.append(bucket.popleft())
    return chosen


def interval_pair_valid(record):
    lower, upper = map(utc_time, record["ignition_interval"])
    if not utc_time(record["pre_at"]) < lower < upper <= utc_time(record["post_at"]):
        raise ValueError("images do not bracket the labelled interval")
    if record["mgrs"] != record["pre_mgrs"] or record["grid"] != record["pre_grid"]:
        raise ValueError("interval pair needs one MGRS and one grid")


def select_frames(root, prior_path):
    russia = root / "russia"; plan_path = russia / "reserve_plan.json"
    plan = json.loads(plan_path.read_text()); prior = json.loads(prior_path.read_text())
    # GeoPackage may reorder polygon rings; raw-source WKB defines the frozen IDs.
    source_archive = russia / "burned_area_2020_topology.zip"
    contours = gpd.read_file("zip://" + str(source_archive)).to_crs(4326)
    contours = contours[contours.geometry.notna() & ~contours.geometry.is_empty & contours.geometry.is_valid]
    contours = contours[contours.intersects(box(80, 50, 115, 65))].copy()
    geometries = {"russia:" + canonical_hash(g.wkb_hex): g for g in contours.geometry}
    groups = defaultdict(list)
    for record in plan["records"]:
        if record["area_ha"] >= 1: groups[(record["partition"], record["group_id"])].append(record)
    history = STRtree([box(*r["bbox_wgs84"]) for r in prior["records"]])
    selected, excluded, previous = [], [], []
    for partition in ("evaluation", "calibration", "fit"):
        limit = {"fit": 8, "calibration": 4, "evaluation": 16}[partition]
        for (_, group), rows in sorted((k, v) for k, v in groups.items() if k[0] == partition):
            for record in round_robin(rows, limit):
                point = geometries[record["sample_id"]].representative_point()
                zone = min(60, max(1, int((point.x + 180) // 6) + 1)); epsg = 32600 + zone
                x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(point.x, point.y)
                left, top = np.floor((x - 7680) / 30) * 30, np.ceil((y + 7680) / 30) * 30
                transform = from_origin(left, top, 30, 30)
                bounds = list(transform_bounds(epsg, 4326, left, top - 15360, left + 15360, top))
                footprint = box(*bounds)
                reason = None
                if len(history.query(footprint, predicate="intersects")): reason = "overlaps prior data"
                elif any(part != partition and footprint.intersects(other) for part, other in previous):
                    reason = "overlaps another partition"
                if reason:
                    excluded.append(dict(sample_id=record["sample_id"], reason=reason)); continue
                selected.append(dict(record, bbox_wgs84=bounds, center=[point.x, point.y],
                                     grid=dict(crs=f"EPSG:{epsg}", transform=list(transform)[:6], width=512, height=512)))
                previous.append((partition, footprint))
    validate_manifest(selected)
    payload = dict(spec_id="SPEC-92", records=selected, exclusions=excluded,
                   reserve_plan_sha256=sha256_file(plan_path), prior_sha256=sha256_file(prior_path),
                   contour_file_sha256=sha256_file(russia / "target_contours.gpkg"),
                   source_policy="1ha; round-robin size bins; caps fit8/cal4/eval16; prior and cross-partition purge")
    frozen = russia / "frame_selection.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != payload: raise ValueError("frozen Russian frames changed")
    else:
        with frozen.open("x") as stream: json.dump(payload, stream, indent=2)
    return selected, contours, frozen


def read_asset(item, asset_key, record, mask=False):
    grid = record["grid"]
    with rasterio.open(item.assets[asset_key].href) as source, WarpedVRT(
        source, crs=grid["crs"], transform=rasterio.Affine(*grid["transform"]),
        width=512, height=512, resampling=Resampling.nearest if mask else Resampling.average,
        src_nodata=0, nodata=0 if mask else np.nan, dtype="uint8" if mask else "float32",
    ) as warped: return warped.read(1)


def choose(catalog, record, year, tile=None, target_date=None):
    items = list(catalog.search(collections=["sentinel-2-l2a"],
                 intersects={"type": "Point", "coordinates": record["center"]},
                 datetime=f"{year}-05-16T00:00:00Z/{year}-07-15T23:59:59Z").items())
    if tile is not None: items = [i for i in items if i.properties.get("s2:mgrs_tile") == tile]
    if target_date is None: items.sort(key=lambda i: i.datetime)
    else: items.sort(key=lambda i: (abs((i.datetime - target_date).total_seconds()), i.id))
    for item in items:
        if item.properties.get("eo:cloud_cover", 100) > 80: continue
        scl = read_asset(item, "SCL", record, mask=True)
        if np.isin(scl, SCL_INVALID).mean() > .3: continue
        image, encodings = [], {}
        for key, band in EXPECTED_BANDS.items():
            asset, basis = pc_asset_encoding(item.assets[band].to_dict(), key,
                                             item.properties.get("s2:processing_baseline"))
            scale, offset, _ = asset_encoding(asset, key)
            raw = read_asset(item, band, record)
            image.append(np.where(np.isfinite(raw), raw * scale + offset, -9999).astype(np.float32))
            encodings[key] = dict(scale=scale, offset=offset, basis=basis, band=band)
        image = np.stack(image)
        good = ~np.isin(scl, SCL_INVALID) & np.isfinite(image).all(0) & (image != -9999).all(0)
        if good.mean() >= .7: return item, image, scl, good, encodings
    raise ValueError("no clean seasonal image in frozen interval")


def build(root, record, contours):
    out = root / "russia/windows" / record["sample_id"].replace(":", "_")
    metadata = out.with_suffix(".json")
    if metadata.exists():
        old = json.loads(metadata.read_text()); interval_pair_valid(old)
        for key, path in old["files"].items():
            if sha256_file(root / path) != old["hashes"][key]: raise ValueError("Russian window input changed")
        return old
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                       modifier=planetary_computer.sign_inplace)
    post, xp, sp, vp, ep = choose(catalog, record, 2020)
    tile = post.properties["s2:mgrs_tile"]
    seasonal = post.datetime.replace(year=2019)
    pre, xb, sb, vb, eb = choose(catalog, record, 2019, tile=tile, target_date=seasonal)
    valid = vp & vb
    if valid.mean() < .5: raise ValueError("less than half the pair is commonly valid")
    nearby = contours.iloc[contours.sindex.query(box(*record["bbox_wgs84"]), predicate="intersects")]
    projected = nearby.to_crs(record["grid"]["crs"])
    transform = rasterio.Affine(*record["grid"]["transform"])
    truth = rasterize([(g, 1) for g in projected.geometry], out_shape=(512, 512),
                      transform=transform, fill=0, dtype="int16")
    if ((truth == 1) & valid).sum() < 9: raise ValueError("fewer than nine valid labelled fire pixels")
    out.parent.mkdir(parents=True, exist_ok=True)
    values = {"pre": (np.where(vb[None], xb, -9999), "float32", -9999),
              "post": (np.where(vp[None], xp, -9999), "float32", -9999),
              "pre_scl": (sb[None], "uint8", 0), "post_scl": (sp[None], "uint8", 0),
              "valid": (valid.astype(np.uint8)[None], "uint8", 0),
              "mask": (np.where(valid, truth, -1)[None], "int16", -1)}
    files = {}
    for key, (array, dtype, nodata) in values.items():
        path = out.with_suffix("." + key + ".tif")
        with rasterio.open(path, "w", driver="GTiff", height=512, width=512, count=len(array),
                           dtype=dtype, crs=record["grid"]["crs"], transform=transform,
                           compress="deflate", nodata=nodata) as raster: raster.write(array.astype(dtype))
        files[key] = str(path.relative_to(root))
    result = dict(record, files=files, hashes={k: sha256_file(root / p) for k, p in files.items()},
                  ignition_interval=["2020-01-01T00:00:00Z", "2020-05-16T00:00:00Z"],
                  pre_at=pre.datetime.isoformat(), post_at=post.datetime.isoformat(),
                  pre_item=pre.id, post_item=post.id, mgrs=tile, pre_mgrs=pre.properties["s2:mgrs_tile"],
                  pre_grid=record["grid"], encodings=dict(pre=eb, post=ep), date_semantics="annual-interval-pair",
                  source_provenance="participatory_perimeter", valid_fraction=float(valid.mean()),
                  fire_pixels=int(((truth == 1) & valid).sum()), sampling_weight=1)
    interval_pair_valid(result)
    with metadata.open("x") as stream: json.dump(result, stream, indent=2)
    print("prepared", record["partition"], record["sample_id"], result["fire_pixels"], flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--prior-footprints", type=Path, required=True)
    p.add_argument("--limit", type=int, help="QC only")
    args = p.parse_args(); root = args.data_root.resolve()
    selected, contours, frozen = select_frames(root, args.prior_footprints)
    if args.limit: selected = selected[:args.limit]
    failures = []
    def run(record):
        for attempt in range(3):
            try: return build(root, record, contours)
            except ValueError as error:
                failures.append(dict(sample_id=record["sample_id"], reason=str(error))); return None
            except Exception as error:
                if attempt == 2:
                    failures.append(dict(sample_id=record["sample_id"], reason=repr(error)[:300])); return None
                time.sleep(5 * (attempt + 1))
    with ThreadPoolExecutor(4) as pool: records = [r for r in pool.map(run, selected) if r]
    records.sort(key=lambda r: r["sample_id"])
    counts = validate_manifest(records)
    eval_groups = {r["group_id"] for r in records if r["partition"] == "evaluation"}
    ready = (args.limit is None and counts["partitions"]["fit"] >= 150
             and counts["partitions"]["evaluation"] >= 50 and len(eval_groups) >= 20)
    result = dict(spec_id="SPEC-92", records=records, counts=counts, failures=failures,
                  selection_sha256=sha256_file(frozen), ready_for_training=ready,
                  qc_probe=args.limit is not None, evaluation_groups=len(eval_groups),
                  non_claims=["Interval labels, not individual ignition dates.",
                              "Spring2020 participatory perimeters, not field truth or summer taiga validation."])
    target = root / "russia" / ("qc-manifest.json" if args.limit else "manifest.json")
    with target.open("x") as stream: json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("records", "failures")}), flush=True)
    if not ready and args.limit is None: raise SystemExit(2)


if __name__ == "__main__": main()
