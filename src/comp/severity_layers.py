"""SPEC-94: source-bound Fmask for severity evaluation without writing legacy data."""
import json
from pathlib import Path

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
import planetary_computer
import pystac_client

from src.comp.accuracy_contract import canonical_hash, sha256_file


def evaluation_fmask(root, record):
    legacy = root / record["files"]["post"].replace("_merged.tif", ".fmask.tif")
    if legacy.exists():
        with rasterio.open(legacy) as source:
            if (str(source.crs) != record["grid"]["crs"] or list(source.transform)[:6] != record["grid"]["transform"]
                    or source.shape != (record["grid"]["height"], record["grid"]["width"])):
                raise ValueError("legacy Fmask grid differs")
            return source.read(1), sha256_file(legacy)
    identity = {"item": record["post_item"], "grid": record["grid"]}
    destination = Path(root) / "severity-fmask" / (canonical_hash(identity) + ".tif")
    metadata = destination.with_suffix(".json")
    if destination.exists() and metadata.exists():
        stored = json.loads(metadata.read_text())
        if stored["identity"] != identity or stored["sha256"] != sha256_file(destination):
            raise ValueError("cached Fmask provenance differs")
        with rasterio.open(destination) as source: return source.read(1), stored["sha256"]
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    item = catalog.get_collection("hls2-s30").get_item(record["post_item"])
    if item is None: raise ValueError("Fmask source item unavailable")
    planetary_computer.sign_inplace(item)
    grid = record["grid"]
    transform = rasterio.Affine(*grid["transform"])
    with rasterio.open(item.assets["Fmask"].href) as source, WarpedVRT(
        source, crs=grid["crs"], transform=transform, width=grid["width"], height=grid["height"],
        resampling=Resampling.nearest, src_nodata=255, nodata=255,
    ) as warped: array = warped.read(1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", driver="GTiff", count=1, width=grid["width"], height=grid["height"],
                       dtype="uint8", crs=grid["crs"], transform=transform, nodata=255, compress="deflate") as raster:
        raster.write(array[None])
    checksum = sha256_file(destination)
    with metadata.open("x") as stream: json.dump(dict(identity=identity, sha256=checksum), stream, indent=2)
    return array, checksum
