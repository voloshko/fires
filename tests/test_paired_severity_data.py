import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts.prepare_paired_severity import select_jobs
from src.comp.accuracy_contract import canonical_hash, sha256_file
from src.comp.severity_layers import evaluation_fmask


def test_fit_footprints_are_purged_before_fetch_or_training(tmp_path):
    for directory in ("hls_fresh", "hls_fresh2", "hls_fresh3", "hls_fresh4"):
        d = tmp_path / "external" / directory; d.mkdir(parents=True)
        windows = []
        if directory in ("hls_fresh", "hls_fresh2"):
            name = directory + "_window"; event = directory + "_event"
            profile = dict(driver="GTiff", width=8, height=8, count=1, dtype="uint8",
                           crs="EPSG:32636", transform=from_origin(500000, 4000000, 30, 30))
            for suffix in ("_merged.tif", ".sev.tif"):
                with rasterio.open(d / (name + suffix), "w", **profile) as raster:
                    raster.write(np.ones((1, 8, 8), np.uint8))
            windows.append(dict(name=name, event_id=event))
        (d / "manifest.json").write_text(json.dumps(dict(windows=windows)))
    selected, excluded = select_jobs(tmp_path)
    assert len(selected) == 1 and selected[0][1] == "evaluation"
    assert len(excluded) == 1 and "intersects" in excluded[0]["reason"]


def test_existing_fmask_cache_requires_identity_and_checksum(tmp_path):
    grid = dict(crs="EPSG:32636", width=8, height=8,
                transform=list(from_origin(500000, 4000000, 30, 30))[:6])
    record = dict(post_item="HLS.S30.fixture", grid=grid, files=dict(post="external/missing_merged.tif"))
    identity = dict(item=record["post_item"], grid=grid)
    path = tmp_path / "severity-fmask" / (canonical_hash(identity) + ".tif")
    path.parent.mkdir()
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype="uint8", crs=grid["crs"],
                       transform=from_origin(500000, 4000000, 30, 30), width=8, height=8) as raster:
        raster.write(np.full((1, 8, 8), 32, np.uint8))
    path.with_suffix(".json").write_text(json.dumps(dict(identity=identity, sha256=sha256_file(path))))
    array, _ = evaluation_fmask(tmp_path, record)
    assert np.all(array == 32)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="provenance"): evaluation_fmask(tmp_path, record)
