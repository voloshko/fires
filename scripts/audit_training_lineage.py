"""SPEC-96: audited event associations and complete known prior footprints."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import canonical_hash, sha256_file


def unique_event(scores):
    """An ambiguous association is unknown, never a tile-derived synthetic event."""
    ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
    if not ranked or ranked[0][1] < .95 or (len(ranked) > 1 and ranked[1][1] > .05): return None
    return ranked[0][0]


def write_once(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload: raise ValueError("frozen lineage audit changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream: json.dump(payload, stream, indent=2)


def events(root, output):
    vector = root / "external/mtbs/mtbs_perims_DD.shp"
    vector_hashes = {p.name: sha256_file(p) for p in sorted(vector.parent.glob("mtbs_perims_DD.*")) if p.is_file()}
    frame = gpd.read_file(vector).to_crs(4326)
    frame["date"] = pd.to_datetime(frame["ig_date"], utc=True, errors="coerce")
    index = frame.sindex
    records = []
    for image in sorted((root / "external/hls_burn_scars/training").glob("*_merged.tif")):
        match = re.search(r"HLS\.S30\.(T[0-9A-Z]+)\.(\d{7})", image.name)
        if match is None: raise ValueError("HLS scene has no acquisition day")
        day = pd.Timestamp(datetime.strptime(match[2], "%Y%j"), tz="UTC")
        mask = Path(str(image).replace("_merged.tif", ".mask.tif"))
        with rasterio.open(image) as raster, rasterio.open(mask) as labels:
            if raster.crs is None or raster.crs != labels.crs or raster.transform != labels.transform:
                raise ValueError("HLS label has no matching georeference")
            image_array, target = raster.read(), labels.read(1)
            valid = (image_array != -9999).all(0) & np.isfinite(image_array).all(0) & (target >= 0)
            positive = (target == 1) & valid
            bounds = list(transform_bounds(raster.crs, 4326, *raster.bounds))
            candidate_idx = index.query(box(*bounds), predicate="intersects")
            candidates = frame.iloc[candidate_idx]
            candidates = candidates[(candidates.date <= day) & (candidates.date >= day - pd.Timedelta(days=450))]
            scores = {}
            if positive.any() and not candidates.empty:
                # Multiple mapping products for one official event form one association.
                projected = candidates.to_crs(raster.crs).dissolve(by="event_id")
                for event_id, row in projected.iterrows():
                    pixels = rasterize([(row.geometry, 1)], out_shape=raster.shape,
                                       transform=raster.transform, fill=0, dtype="uint8").astype(bool)
                    scores[str(event_id)] = float((pixels & positive).sum() / positive.sum())
        event = unique_event(scores)
        record = dict(sample_id="hls:" + image.name, source=str(image.relative_to(root)),
                      source_sha256=sha256_file(image), label_sha256=sha256_file(mask),
                      acquisition_day=str(day.date()), bbox_wgs84=bounds,
                      event_id=event, decision="matched" if event else "unknown",
                      candidates=sorted(scores.items(), key=lambda p: (-p[1], p[0])))
        records.append(record)
        print(record["sample_id"], record["decision"], event, flush=True)
    if not records: raise ValueError("missing HLS fit pool")
    payload = dict(spec_id="SPEC-96", vector_hashes=vector_hashes, records=records,
                   matched=sum(r["decision"] == "matched" for r in records),
                   unknown=sum(r["decision"] == "unknown" for r in records),
                   criteria=dict(positive_pixel_coverage=.95, second_candidate_max=.05, max_age_days=450),
                   non_claims=["Geometric/date association, not a publisher-provided event lookup.",
                               "Unknown associations block full event-retention claims."])
    payload["audit_sha256"] = canonical_hash(payload); write_once(output, payload)
    print("RESULT", payload["matched"], "matched", payload["unknown"], "unknown", flush=True)


def raster_footprint(path, root):
    with rasterio.open(path) as source:
        if source.crs is None: raise ValueError(f"prior raster has no CRS: {path}")
        bounds = list(transform_bounds(source.crs, 4326, *source.bounds))
        if not np.isfinite(bounds).all() or not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
            raise ValueError("invalid prior footprint")
        return dict(sample_id=str(path.relative_to(root)), bbox_wgs84=bounds,
                    source_sha256=sha256_file(path), source_crs=str(source.crs))


def footprints(root, output):
    records, source_manifests = [], {}
    for split in ("training", "validation"):
        directory = root / "external/hls_burn_scars" / split
        images = sorted(directory.glob("*_merged.tif"))
        if not images: raise ValueError("missing prior HLS split")
        records.extend(raster_footprint(image, root) for image in images)
    for directory in ("hls_fresh", "hls_fresh2", "hls_fresh3", "hls_fresh4", "ems_hls", "floga_hls"):
        manifest = root / "external" / directory / "manifest.json"
        source_manifests[str(manifest.relative_to(root))] = sha256_file(manifest)
        for window in json.loads(manifest.read_text())["windows"]:
            records.append(raster_footprint(manifest.parent / (window["name"] + "_merged.tif"), root))
    case = root / "data/comp/train/bs/meta.csv"; meta = pd.read_csv(case)
    source_manifests[str(case.relative_to(root))] = sha256_file(case)
    for _, row in meta.iterrows():
        values = [row[k] for k in ("x_min", "y_min", "x_max", "y_max")]
        if not np.isfinite(values).all() or not np.isfinite(row.epsg): raise ValueError("case chip lacks coordinates")
        bounds = list(transform_bounds(f"EPSG:{int(row.epsg)}", 4326, *values))
        if not np.isfinite(bounds).all(): raise ValueError("case footprint transform failed")
        records.append(dict(sample_id="case:" + row.chip_id, bbox_wgs84=bounds,
                            source_sha256=source_manifests[str(case.relative_to(root))],
                            source_crs=f"EPSG:{int(row.epsg)}"))
    payload = dict(spec_id="SPEC-96", records=records, source_manifests=source_manifests,
                   scope="Provided historical HLS/MTBS/EMS/FLOGA/competition-train data; includes prior evaluations.",
                   non_claims=["Not all possible observations used in global third-party pretraining."])
    payload["audit_sha256"] = canonical_hash(payload); write_once(output, payload)
    print("RESULT", len(records), "prior footprints", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("events", "footprints"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (events if args.mode == "events" else footprints)(args.source_root.resolve(), args.out.resolve())


if __name__ == "__main__": main()
