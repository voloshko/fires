"""SPEC-93: fetch source-matched L2A twins for the full C1-MM fit pool."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
import pystac_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import event_group, sha256_file, validate_manifest
from src.comp.l2a_finetuning import EXPECTED_BANDS, asset_encoding, l2a_valid


def source_jobs(root):
    jobs = []
    for image in sorted((root / "external/hls_burn_scars/training").glob("*_merged.tif")):
        tile = re.search(r"HLS\.S30\.(T[0-9A-Z]+)\.", image.name).group(1)
        jobs.append(dict(sample_id="hls:" + image.name, group_id="hls:" + tile,
                         dataset="hls", partition="fit", label_provenance="expert_perimeter",
                         source=str(image.relative_to(root)), sampling_weight=1))
    for dataset, directory, manifest, part in [
        ("mtbs", "hls_fresh", "manifest.json", "fit"),
        ("ems", "ems_hls", "manifest_train.json", "fit"),
        ("floga", "floga_hls", "manifest_train.json", "fit"),
        ("ems", "ems_hls", "manifest_test.json", "evaluation"),
        ("floga", "floga_hls", "manifest_test.json", "evaluation"),
    ]:
        path = root / "external" / directory / manifest
        for record in json.loads(path.read_text())["windows"]:
            jobs.append(dict(sample_id=dataset + ":" + record["name"], group_id=event_group(dataset, record),
                             dataset=dataset, partition=part, label_provenance="expert_perimeter",
                             source=str(path.parent.relative_to(root) / (record["name"] + "_merged.tif")),
                             sampling_weight=2 if dataset in ("ems", "floga") else 1))
    return jobs


def twin(root, out, job):
    image_path = root / job["source"]
    name = image_path.name.replace("_merged.tif", "")
    destination = out / job["dataset"] / name
    metadata = Path(str(destination) + ".json")
    if metadata.exists():
        record = json.loads(metadata.read_text())
        for key, relative in record["files"].items():
            if sha256_file(root / relative) != record["hashes"][key]: raise ValueError("twin inputs changed")
        return record
    match = re.search(r"HLS\.S30\.(T[0-9A-Z]+)\.(\d{7})", name)
    if match is None: raise ValueError("cannot identify source MGRS and acquisition day")
    tile, day = match.group(1), datetime.strptime(match.group(2), "%Y%j")
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    items = list(catalog.search(collections=["sentinel-2-c1-l2a"],
                 query={"grid:code": {"eq": "MGRS-" + tile[1:]}},
                 datetime=f"{day:%Y-%m-%dT00:00:00Z}/{day + timedelta(days=1):%Y-%m-%dT00:00:00Z}").items())
    if not items: raise ValueError("no same-day same-MGRS L2A")
    item = min(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))
    if item.datetime.date() != day.date(): raise ValueError("twin day differs")
    with rasterio.open(image_path) as source:
        grid = dict(crs=source.crs, transform=source.transform, width=source.width, height=source.height)
        profile = source.profile; source_image = source.read()
        bbox = list(transform_bounds(source.crs, 4326, *source.bounds))
    bands, encodings = [], {}
    for key in EXPECTED_BANDS:
        asset = item.assets[key].to_dict()
        scale, offset, nodata = asset_encoding(asset, key)
        with rasterio.open(item.assets[key].href) as source, WarpedVRT(
            source, **grid, resampling=Resampling.average, src_nodata=nodata, nodata=np.nan,
            dtype="float32",
        ) as warped: raw = warped.read(1)
        bands.append(np.where(np.isfinite(raw), raw * scale + offset, -9999).astype(np.float32))
        encodings[key] = dict(band=EXPECTED_BANDS[key], scale=scale, offset=offset, nodata=nodata)
    image = np.stack(bands)
    with rasterio.open(item.assets["scl"].href) as source, WarpedVRT(
        source, **grid, resampling=Resampling.nearest, src_nodata=0, nodata=0,
    ) as warped: scl = warped.read(1).astype(np.uint8)
    label_path = Path(str(image_path).replace("_merged.tif", ".mask.tif"))
    with rasterio.open(label_path) as source:
        labels = source.read(1)
        if source.crs != grid["crs"] or source.transform != grid["transform"]:
            raise ValueError("source annotation grid differs")
    valid = l2a_valid(image, scl, labels)
    valid &= (source_image != -9999).all(0) & np.isfinite(source_image).all(0)
    if valid.mean() < .5 or not ((labels == 1) & valid).any():
        raise ValueError("not enough common valid pixels / no annotated fire")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(str(destination) + "." + key + ".tif") for key in ("l2a", "scl", "valid")}
    for key, array, dtype, nodata in [("l2a", image, "float32", -9999),
                                     ("scl", scl[None], "uint8", 0),
                                     ("valid", valid.astype(np.uint8)[None], "uint8", 0)]:
        with rasterio.open(paths[key], "w", **dict(profile, count=len(array), dtype=dtype, nodata=nodata)) as raster:
            raster.write(array)
    files = {key: str(path.relative_to(root)) for key, path in paths.items()}
    files.update(hls=job["source"], mask=str(label_path.relative_to(root)))
    record = dict(job, bbox_wgs84=bbox, grid=dict(crs=str(grid["crs"]),
                  transform=list(grid["transform"])[:6], width=grid["width"], height=grid["height"]),
                  files=files, hashes={key: sha256_file(root / path) for key, path in files.items()},
                  l2a_item=item.id, acquisition=item.datetime.isoformat(), encodings=encodings,
                  valid_fraction=float(valid.mean()))
    with metadata.open("x") as stream: json.dump(record, stream, indent=2)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="QC only, cannot authorize full training")
    args = parser.parse_args(); root, out = args.data_root.resolve(), args.out.resolve()
    if not out.is_relative_to(root): raise ValueError("outputs must be inside data root")
    out.mkdir(parents=True, exist_ok=True)
    jobs = source_jobs(root); fit_groups = {j["group_id"] for j in jobs if j["partition"] == "fit"}
    if args.limit: jobs = jobs[:args.limit]
    failures = []
    def run(job):
        for attempt in range(3):
            try:
                record = twin(root, out, job); print("prepared", record["sample_id"], flush=True); return record
            except ValueError as error:
                failures.append(dict(sample=job["sample_id"], reason=str(error))); return None
            except Exception as error:
                if attempt == 2:
                    failures.append(dict(sample=job["sample_id"], reason=repr(error)[:300])); return None
                time.sleep(5 * (attempt + 1))
    with ThreadPoolExecutor(4) as pool: records = [r for r in pool.map(run, jobs) if r]
    records.sort(key=lambda r: r["sample_id"]); counts = validate_manifest(records)
    fit = [r for r in records if r["partition"] == "fit"]
    fraction = len({r["group_id"] for r in fit}) / len(fit_groups)
    ready = args.limit is None and len(fit) >= 300 and fraction >= .7
    result = dict(spec_id="SPEC-93", records=records, counts=counts, failures=failures,
                  original_fit_groups=len(fit_groups), retained_fit_group_fraction=fraction,
                  ready_for_training=ready, qc_probe=args.limit is not None)
    with (out / ("qc-manifest.json" if args.limit else "manifest.json")).open("x") as stream:
        json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("records", "failures")}), flush=True)
    if not ready and args.limit is None: raise SystemExit(2)


if __name__ == "__main__": main()
