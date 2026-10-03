import json
from pathlib import Path
import zipfile

import pytest
from shapely.geometry import box

from scripts.build_russian_reserve import fetch, partition, spatial_clusters
from src.comp.accuracy_contract import sha256_file


def test_spatial_clusters_include_transitive_neighbours():
    geoms = [box(x, 0, x + 100, 100) for x in (0, 19000, 38000, 90000)]
    groups = spatial_clusters(geoms)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[2]
    assert partition(groups[0]) == partition(groups[2])


def test_partition_is_order_independent():
    geoms = [box(0, 0, 10, 10), box(100000, 0, 100010, 10)]
    assert spatial_clusters(geoms) == list(reversed(spatial_clusters(list(reversed(geoms)))))


def test_frozen_archive_change_fails(tmp_path):
    archive = tmp_path / "burned_area_2020_topology.zip"
    with zipfile.ZipFile(archive, "w") as z: z.writestr("data.txt", "fixture")
    (tmp_path / "source_manifest.json").write_text(json.dumps({"sha256": sha256_file(archive)}))
    assert fetch(tmp_path) == archive
    archive.write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"): fetch(tmp_path)
