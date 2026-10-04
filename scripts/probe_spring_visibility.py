"""SPEC-108: small CPU-only, in-period source visibility probe."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import planetary_computer
import pystac_client
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file
from src.comp.l2a_finetuning import EXPECTED_BANDS, SCL_INVALID, asset_encoding, pc_asset_encoding
from src.comp.spring_visibility import BINS, candidates, compare_view, decision, storage_guard
from scripts.prepare_russian_windows import read_asset

PARENT_SHA = "749699f3f6e01f0a0a6576afb496c891c4bc33c05ad1cca4087585ed5628cb2c"


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def item_metadata(item):
    return dict(item_id=item.id, date=item.datetime.isoformat() if item.datetime is not None else None,
        mgrs=item.properties.get("s2:mgrs_tile"), cloud=item.properties.get("eo:cloud_cover"),
        baseline=item.properties.get("s2:processing_baseline"))


def check_space(out):
    files = list(out.rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("visibility output contains symlink")
    storage_guard(sum(p.stat().st_size for p in files if p.is_file()), shutil.disk_usage(out).free)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "manifest", "parent", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root.resolve(), args.out.resolve()
    if not out.is_relative_to(root):
        raise ValueError("visibility outputs escape own data root")
    if out.exists():
        raise ValueError("immutable visibility experiment already exists")
    if sha256_file(args.parent) != PARENT_SHA:
        raise ValueError("wrong geographic parent")
    parent = json.loads(args.parent.read_text())
    if sha256_file(args.manifest) != parent["manifest_sha256"]:
        raise ValueError("original source manifest changed")
    ids = sorted(parent["errors"]["models"]["reserve"]["missed_positive_windows"])
    if len(ids) != 4:
        raise ValueError("wrong missed-window population")
    records = [r for r in json.loads(args.manifest.read_text())["records"] if r["sample_id"] in ids]
    records.sort(key=lambda r: r["sample_id"])
    if [r["sample_id"] for r in records] != ids:
        raise ValueError("missing original identity")
    for record in records:
        for name, relative in record["files"].items():
            if sha256_file(root/relative) != record["hashes"][name]:
                raise ValueError("original pixels changed")
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace)
    views, item_lookup = [], {}
    for record in records:
        items = list(catalog.search(collections=["sentinel-2-l2a"],
            intersects={"type": "Point", "coordinates": record["center"]},
            datetime="2020-03-01T00:00:00Z/2020-05-15T23:59:59Z").items())
        item_lookup.update({i.id: i for i in items})
        metadata = [item_metadata(i) for i in items]
        for lower, upper in BINS:
            views.append(dict(sample_id=record["sample_id"], interval=[lower, upper],
                candidates=candidates(metadata, record["mgrs"], lower, upper)))
    sources = {p: sha256_file(p) for p in (str(Path(__file__)), "src/comp/spring_visibility.py",
        "scripts/prepare_russian_windows.py", "src/comp/l2a_finetuning.py",
        "src/comp/geographic_failure_report.py", "src/comp/accuracy_contract.py")}
    plan = dict(spec_id="SPEC-108", parent_sha256=PARENT_SHA,
        manifest_sha256=sha256_file(args.manifest), views=views, source_hashes=sources)
    out.mkdir()
    check_space(out)
    write_json(out/"plan.json", plan)
    by_id = {r["sample_id"]: r for r in records}
    observations = []
    for view in views:
        result = dict(sample_id=view["sample_id"], interval=view["interval"], usable=False,
            attempts=[], reason="no observable candidate")
        record = by_id[view["sample_id"]]
        stage = "observation admission"
        try:
            check_space(out)
            selected = None
            for metadata in view["candidates"]:
                item = item_lookup[metadata["item_id"]]
                if item_metadata(item) != metadata:
                    raise ValueError("frozen item metadata changed")
                stage = "SCL read"
                scl = read_asset(item, "SCL", record, mask=True)
                valid_fraction = float((~np.isin(scl, SCL_INVALID)).mean())
                result["attempts"].append(dict(item_id=item.id, scl_valid_fraction=valid_fraction))
                if valid_fraction >= .5:
                    selected = item
                    break
            if selected is None:
                observations.append(result)
                print("NO_OBSERVATION", view["sample_id"], view["interval"], flush=True)
                continue
            image, encodings = [], {}
            for key, band in EXPECTED_BANDS.items():
                stage = "reflectance read " + band
                asset, basis = pc_asset_encoding(selected.assets[band].to_dict(), key,
                    selected.properties.get("s2:processing_baseline"))
                scale, offset, _ = asset_encoding(asset, key)
                raw = read_asset(selected, band, record)
                image.append(np.where(np.isfinite(raw), raw*scale+offset, -9999).astype(np.float32))
                encodings[key] = dict(scale=scale, offset=offset, basis=basis, band=band)
            image = np.stack(image)
            valid = (~np.isin(scl, SCL_INVALID) & np.isfinite(image).all(0) & (image != -9999).all(0))
            prefix = view["sample_id"].replace(":", "_")+"-"+view["interval"][0]
            files = {}
            result["files"] = files
            for name, array, dtype, nodata in (("image", image, "float32", -9999),
                    ("scl", scl[None], "uint8", 0)):
                stage = "retained pixel write " + name
                check_space(out)
                path = out/(prefix+"-"+name+".tif")
                if path.exists():
                    raise ValueError("immutable view pixel file exists")
                with rasterio.open(path, "w", driver="GTiff", width=512, height=512,
                    count=len(array), dtype=dtype, nodata=nodata, compress="deflate",
                    crs=record["grid"]["crs"], transform=rasterio.Affine(*record["grid"]["transform"])) as target:
                    target.write(array.astype(dtype))
                files[name] = dict(path=str(path.relative_to(root)), sha256=sha256_file(path))
            arrays = {}
            stage = "same-pixel comparison"
            for key in ("post", "mask", "valid"):
                with rasterio.open(root/record["files"][key]) as source:
                    arrays[key] = source.read() if key == "post" else source.read(1)
            metrics = compare_view(arrays["post"], image, arrays["mask"], arrays["valid"].astype(bool), valid)
            result.update(metrics, date=selected.datetime.isoformat(), item_id=selected.id,
                mgrs=selected.properties["s2:mgrs_tile"], encodings=encodings, files=files,
                reason=None if metrics["usable"] else "below common observation admission")
            print("OBSERVATION", view["sample_id"], result["date"], result["usable"], flush=True)
        except Exception as error:
            # Preserve partial pixel files; absence never becomes a negative label.
            # Raster/network exceptions can embed signed URLs. Never persist
            # exception messages or source hrefs in this public evidence.
            result["reason"] = type(error).__name__+" during "+stage
            print("UNAVAILABLE", view["sample_id"], type(error).__name__, flush=True)
        observations.append(result)
    outcome = decision(observations, ids)
    payload = dict(spec_id="SPEC-108", plan_sha256=sha256_file(out/"plan.json"),
        observations=observations, **outcome,
        storage_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        non_claims=["Selected consumed examples, not independent accuracy or a new model.",
            "Published background is not reviewed fire absence; contrasts do not determine ignition dates.",
            "No causal regrowth/misannotation, complete temporal coverage, field truth or goal completion."])
    check_space(out)
    write_json(out/"result.json", payload)
    print("RESULT", json.dumps(outcome), flush=True)


if __name__ == "__main__":
    with rasterio.Env(GDAL_HTTP_TIMEOUT=30, GDAL_HTTP_MAX_RETRY=0, GDAL_CACHEMAX=64):
        main()
