"""SPEC-112: freeze unused geographic identities; no catalogue, imagery or inference."""
import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_russian_reserve import spatial_clusters
from scripts.prepare_western_holdout import SOURCE_SHA, frame
from src.comp.accuracy_contract import canonical_hash, sha256_file, validate_manifest
from src.comp.eastern_confirmation_plan import REGISTRIES, balanced_frames, verify_histories
from src.comp.western_holdout import exclude_groups

BBOX = (115, 50, 150, 65)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root, out = args.data_root.resolve(), args.out.resolve()
    if not out.is_relative_to(root) or out.exists():
        raise ValueError("fresh owned plan path required")
    archive = root/"russia/burned_area_2020_topology.zip"
    if sha256_file(archive) != SOURCE_SHA:
        raise ValueError("frozen annotation ZIP changed")
    histories = {name: (sha256_file(root/name), json.loads((root/name).read_text())) for name in REGISTRIES}
    audit = verify_histories(histories)
    polygons = gpd.read_file("zip://"+str(archive), bbox=BBOX)
    if str(polygons.crs) != "EPSG:4326":
        raise ValueError("annotation CRS differs")
    polygons = polygons[polygons.geometry.notna() & ~polygons.geometry.is_empty & polygons.geometry.is_valid]
    polygons = polygons[polygons.intersects(box(*BBOX))].copy()
    geometries = {"eastern:"+canonical_hash(g.wkb_hex): g for g in polygons.geometry}
    if len(geometries) != len(polygons):
        raise ValueError("duplicate raw geometries")
    metric = polygons.to_crs(3576)
    groups = spatial_clusters(list(metric.geometry))
    previous = [r["bbox_wgs84"] for _, h in histories.values() for r in h["records"]]
    projected = gpd.GeoSeries([box(*b) for b in previous], crs=4326).to_crs(3576)
    tree = STRtree(list(projected))
    rows, bad = [], set()
    for geometry, projected_geometry, group in zip(polygons.geometry, metric.geometry, groups):
        identity = "eastern:"+canonical_hash(geometry.wkb_hex)
        rows.append(dict(sample_id=identity, group_id="eastern:"+group, cluster_id=group,
            area_ha=float(projected_geometry.area/10000)))
        if len(tree.query(projected_geometry.buffer(20000), predicate="intersects")):
            bad.add(identity)
    eligible, purged = exclude_groups(rows, bad)
    provisional = [frame(r, geometries[r["sample_id"]]) for r in balanced_frames(eligible, cap=len(eligible)*4)]
    bad_frames = set()
    for r in provisional:
        footprint = gpd.GeoSeries([box(*r["bbox_wgs84"])], crs=4326).to_crs(3576).iloc[0]
        if len(tree.query(footprint.buffer(20000), predicate="intersects")):
            bad_frames.add(r["sample_id"])
    eligible, frame_purged = exclude_groups(eligible, bad_frames)
    records = [frame(r, geometries[r["sample_id"]]) for r in balanced_frames(eligible)]
    counts = validate_manifest(records) if records else dict(samples=0, groups=0)
    ready = counts["samples"] >= 50 and counts["groups"] >= 20
    sources = (str(Path(__file__)), "src/comp/eastern_confirmation_plan.py", "src/comp/western_holdout.py",
        "scripts/build_russian_reserve.py", "scripts/prepare_western_holdout.py", "src/comp/accuracy_contract.py")
    plan = dict(spec_id="SPEC-112", bbox=list(BBOX), source_zip_sha256=SOURCE_SHA,
        exclusion_audit=audit, source_contours=len(rows), original_groups=len(set(groups)),
        excluded_groups=sorted(purged|frame_purged), eligible_groups=len({r["group_id"] for r in eligible}),
        counts=counts, records=records, ready_metadata_population=bool(ready),
        policy="EPSG3576/20km/whole-group purge/72max/4pergroup/spec112 hash/area bins",
        source_hashes={name: sha256_file(name) for name in sources},
        imagery_downloaded=False, prediction_read=False,
        future_collection_constraints=dict(max_bytes=768*1024**2, free_floor_bytes=3*1024**3,
            minimum_windows=50, minimum_groups=20, requires_separate_spec_and_eligible_candidate=True),
        non_claims=["Metadata-only unused geography; shared publisher/spring2020 and spatial groups, not dated fires.",
            "No data admission, model comparison, imagery quality, field truth, promotion or goal completion."])
    content = json.dumps(plan, indent=2, allow_nan=False)
    if len(content.encode()) > 512*1024:
        raise ValueError("metadata plan size exceeds512KiB")
    out.mkdir()
    with (out/"selection.json").open("x") as stream:
        stream.write(content+"\n")
    print(json.dumps({k: v for k, v in plan.items() if k not in ("records", "excluded_groups", "source_hashes")}), flush=True)


if __name__ == "__main__":
    main()
