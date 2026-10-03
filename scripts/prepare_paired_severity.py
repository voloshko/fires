"""SPEC-94: prepare MTBS severity pairs after purging evaluation footprints."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file, validate_manifest
from src.comp.paired_severity import mtbs_targets
from scripts.prepare_forest_pairs import prepare


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def select_jobs(root):
    jobs, excluded = [], []
    for directory, part in [("hls_fresh2", "evaluation"), ("hls_fresh", "fit"),
                            ("hls_fresh3", "fit"), ("hls_fresh4", "fit")]:
        source = root / "external" / directory
        for window in json.loads((source / "manifest.json").read_text())["windows"]:
            severity = source / (window["name"] + ".sev.tif")
            if not severity.exists():
                excluded.append(dict(name=window["name"], reason="no published severity")); continue
            with rasterio.open(source / (window["name"] + "_merged.tif")) as raster:
                bounds = list(transform_bounds(raster.crs, 4326, *raster.bounds))
            jobs.append((directory, part, window, bounds))
    evaluation = [j for j in jobs if j[1] == "evaluation"]
    event_ids = {j[2]["event_id"] for j in evaluation}
    keep = []
    for job in jobs:
        if job[1] == "fit" and (job[2]["event_id"] in event_ids or
                                any(overlaps(job[3], e[3]) for e in evaluation)):
            excluded.append(dict(name=job[2]["name"], reason="fit intersects frozen evaluation"))
        else: keep.append(job)
    return keep, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="QC only")
    args = parser.parse_args(); root, out = args.data_root.resolve(), args.out.resolve()
    if not out.is_relative_to(root): raise ValueError("outputs must be in data root")
    out.mkdir(parents=True, exist_ok=True)
    jobs, failures = select_jobs(root)
    selection = dict(jobs=[dict(directory=j[0], partition=j[1], name=j[2]["name"], bbox=j[3]) for j in jobs],
                     exclusions=failures.copy())
    frozen = out / "selection.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != selection: raise ValueError("selection changed")
    else:
        with frozen.open("x") as stream: json.dump(selection, stream, indent=2)
    if args.limit: jobs = jobs[:args.limit]
    def run(job):
        directory, part, window, _ = job
        for attempt in range(3):
            try:
                source = root / "external" / directory
                # Reuse already prepared SPEC-91 fresh1 pairs without rebuilding them.
                previous = root / "forest-pairs/mtbs" / (window["name"] + ".pair.json")
                if directory == "hls_fresh" and previous.exists():
                    record = json.loads(previous.read_text()); record["partition"] = part
                else: record = prepare(root, out, "mtbs", part, window, source_directory=directory)
                severity = source / (window["name"] + ".sev.tif")
                with rasterio.open(severity) as raster:
                    if str(raster.crs) != record["grid"]["crs"] or list(raster.transform)[:6] != record["grid"]["transform"]:
                        raise ValueError("severity grid differs from pair")
                    labels = raster.read(1)
                with rasterio.open(root / record["files"]["valid"]) as raster: valid = raster.read(1).astype(bool)
                counts = np.bincount(mtbs_targets(labels, valid)[mtbs_targets(labels, valid) >= 0], minlength=3)
                if counts.sum() == 0: raise ValueError("no common valid severity supervision")
                record = dict(record, files=dict(record["files"], severity=str(severity.relative_to(root))),
                              hashes=dict(record["hashes"], severity=sha256_file(severity)),
                              severity_counts=counts.tolist(), label_provenance="mtbs_severity")
                print("prepared", part, record["sample_id"], "classes", counts.tolist(), flush=True)
                return record
            except ValueError as error:
                failures.append(dict(name=window["name"], reason=str(error))); return None
            except Exception as error:
                if attempt == 2:
                    failures.append(dict(name=window["name"], reason=repr(error)[:300])); return None
                time.sleep(5 * (attempt + 1))
    with ThreadPoolExecutor(4) as pool: records = [r for r in pool.map(run, jobs) if r]
    records.sort(key=lambda r: r["sample_id"]); counts = validate_manifest(records, require_pairs=True)
    fit = [r for r in records if r["partition"] == "fit"]
    class_counts = np.array([r["severity_counts"] for r in fit]).sum(0).tolist() if fit else [0, 0, 0]
    ready = (args.limit is None and counts["partitions"]["evaluation"] >= 100
             and len(fit) >= 200 and all(c > 0 for c in class_counts))
    result = dict(spec_id="SPEC-94", records=records, counts=counts, failures=failures,
                  fit_class_counts=class_counts, ready_for_training=ready, qc_probe=args.limit is not None,
                  selection_sha256=sha256_file(frozen))
    with (out / ("qc-manifest.json" if args.limit else "manifest.json")).open("x") as stream:
        json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("records", "failures")}), flush=True)
    if not ready and args.limit is None: raise SystemExit(2)


if __name__ == "__main__": main()
