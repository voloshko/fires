"""SPEC-92: fetch/audit independent Russian contours and freeze spatial groups.

Run on the remote data host. This does not manufacture event dates or negatives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import canonical_hash, sha256_file

DOI = "https://doi.pangaea.de/10.1594/PANGAEA.939429"
CITATION = "Drozdovskaya et al. (2024), Spring fires 2020 in Russia, PANGAEA.939429"


def open_source(url, timeout=60):
    for attempt in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "fires-research/1.0 (public scientific data)"}), timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2: raise
            time.sleep(10 * (attempt + 1))


def source_url():
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self): super().__init__(); self.urls = []
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                href = dict(attrs).get("href", "")
                if href.endswith("burned_area_2020_topology.zip"):
                    self.urls.append(href)
    with open_source(DOI + "?format=html") as response:
        page = response.read().decode("utf-8")
    parser = Links(); parser.feed(page)
    urls = sorted(set(parser.urls))
    if len(urls) != 1 or not urls[0].startswith("https://download.pangaea.de/"):
        raise ValueError("publisher does not expose one expected contour archive")
    return urls[0]


def fetch(out):
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "burned_area_2020_topology.zip"
    manifest = out / "source_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if sha256_file(archive) != old["sha256"]:
            raise ValueError("source archive differs from frozen source manifest")
        return archive
    url = source_url()
    if not archive.exists():
        temp = archive.with_suffix(".part")
        with open_source(url, timeout=120) as response, temp.open("wb") as stream:
            import shutil
            shutil.copyfileobj(response, stream)
        if not zipfile.is_zipfile(temp):
            raise ValueError("download is not a ZIP archive")
        temp.replace(archive)
    if not zipfile.is_zipfile(archive): raise ValueError("invalid cached archive")
    payload = dict(doi=DOI, citation=CITATION, license="CC-BY-4.0", url=url,
                   sha256=sha256_file(archive), bytes=archive.stat().st_size,
                   downloaded_at=datetime.now(timezone.utc).isoformat(),
                   provenance="participatory_perimeter",
                   interval=["2020-01-01", "2020-05-15"],
                   limitations=["Satellite visual interpretation, not field truth.",
                                "Publisher notes archive was not proof-read by author.",
                                "Individual fire dates must be audited, never inferred from polygon absence."])
    with manifest.open("x") as stream: json.dump(payload, stream, indent=2)
    return archive


def spatial_clusters(geometries, distance=20000):
    """Union all polygons within distance in a metric CRS, including chains."""
    from shapely.strtree import STRtree
    tree = STRtree(geometries); parent = list(range(len(geometries)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, geom in enumerate(geometries):
        for j in tree.query(geom.buffer(distance)):
            j = int(j)
            if j > i and geom.distance(geometries[j]) <= distance:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    members = {}
    for i in range(len(geometries)): members.setdefault(root(i), []).append(i)
    ids = {}
    for idx in members.values():
        key = canonical_hash(sorted(geometries[i].wkb_hex for i in idx))
        for i in idx: ids[i] = key
    return [ids[i] for i in range(len(geometries))]


def partition(cluster):
    import hashlib
    digit = int(hashlib.sha256(("russia-20261003:" + cluster).encode()).hexdigest(), 16) % 10
    return "fit" if digit < 6 else "calibration" if digit < 8 else "evaluation"


def prior_overlap_audit(out, prior):
    from shapely.geometry import box
    from shapely.strtree import STRtree
    plan_path = out / "reserve_plan.json"
    plan = json.loads(plan_path.read_text()); history = json.loads(prior.read_text())
    frames = [box(*r["bbox_wgs84"]) for r in history["records"]]
    if not frames: raise ValueError("prior footprint registry is empty")
    tree = STRtree(frames); overlaps = []
    for record in plan["records"]:
        if record["partition"] not in ("evaluation", "calibration"): continue
        hits = tree.query(box(*record["bbox_wgs84"]), predicate="intersects")
        if len(hits): overlaps.append(dict(sample_id=record["sample_id"], partition=record["partition"],
                                          prior_samples=[history["records"][int(i)]["sample_id"] for i in hits]))
    payload = dict(spec_id="SPEC-92", reserve_plan_sha256=sha256_file(plan_path),
                   prior_sha256=sha256_file(prior), prior_count=len(frames), overlaps=overlaps,
                   scope="Contour bounding boxes only; actual selected chip frames need a second check.",
                   ready_for_frame_selection=not overlaps)
    path = out / "prior_overlap.json"
    if path.exists():
        if json.loads(path.read_text()) != payload: raise ValueError("prior overlap audit changed")
    else:
        with path.open("x") as stream: json.dump(payload, stream, indent=2)
    print("PRIOR", len(frames), "footprints", len(overlaps), "contour overlaps", flush=True)
    if overlaps: raise ValueError("held-out Russian contours intersect previously observed frames")


def audit(out, archive, prior):
    plan_path = out / "reserve_plan.json"
    if plan_path.exists():
        frozen = json.loads(plan_path.read_text())
        digest = frozen.pop("manifest_sha256")
        if frozen["source_sha256"] != sha256_file(archive) or digest != canonical_hash(frozen):
            raise ValueError("frozen Russian split provenance changed")
        prior_overlap_audit(out, prior)
        return
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.read_file("zip://" + str(archive.resolve()))
    if g.crs is None: raise ValueError("contours have no CRS")
    fields = {str(k): str(v) for k, v in g.dtypes.items()}
    total = len(g); g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    invalid = int((~g.geometry.is_valid).sum())
    g = g[g.geometry.is_valid].to_crs(4326)
    # The target bounding region is frozen before model predictions.
    selected = g[g.intersects(box(80, 50, 115, 65))].copy()
    metric = selected.to_crs(3576)
    groups = spatial_clusters(list(metric.geometry))
    selected["cluster_id"] = groups
    selected["partition"] = [partition(k) for k in groups]
    selected["label_provenance"] = "participatory_perimeter"
    records = []
    for (_, row), geom in zip(selected.iterrows(), metric.geometry):
        records.append(dict(sample_id="russia:" + canonical_hash(row.geometry.wkb_hex),
                            group_id="russia:" + row.cluster_id, cluster_id=row.cluster_id,
                            partition=row.partition, label_provenance=row.label_provenance,
                            bbox_wgs84=list(row.geometry.bounds), area_ha=geom.area / 1e4,
                            interval=["2020-01-01", "2020-05-15"]))
    summary = dict(source_sha256=sha256_file(archive), total_contours=total,
                   invalid_contours=invalid, target_contours=len(records), fields=fields,
                   target_bbox=[80, 50, 115, 65], cluster_distance_m=20000,
                   split_seed="russia-20261003", records=records,
                   groups={p: len({r["group_id"] for r in records if r["partition"] == p})
                           for p in ("fit", "calibration", "evaluation")},
                   date_status="interval-only-until-source-fields-reviewed",
                   negative_status="no-negative-labels-created")
    summary["manifest_sha256"] = canonical_hash(summary)
    target = out / "reserve_plan.json"
    if target.exists():
        if json.loads(target.read_text()) != summary: raise ValueError("frozen plan differs")
    else:
        with target.open("x") as stream: json.dump(summary, stream, indent=2)
    if not (out / "target_contours.gpkg").exists():
        selected.to_file(out / "target_contours.gpkg", driver="GPKG")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
    prior_overlap_audit(out, prior)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--prior-footprints", type=Path)
    args = parser.parse_args()
    archive = fetch(args.out)
    print("archive", archive, "sha256", sha256_file(archive), flush=True)
    if args.audit:
        if args.prior_footprints is None: raise ValueError("prior footprint registry is required for split audit")
        audit(args.out, archive, args.prior_footprints)


if __name__ == "__main__": main()
