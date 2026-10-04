"""SPEC-112: metadata completeness and fixed whole-group sample allocation."""
from collections import defaultdict, deque
import math

from .accuracy_contract import canonical_hash
from .western_holdout import EXCLUSION_SOURCES, verify_exclusions

REGISTRIES = dict(EXCLUSION_SOURCES, **{
    "western-holdout-v1/manifest.json": (96, "SPEC-105", "a5a01d9e3ca34c87914717e6c65a0f3e04bc2398c8b17db77403fc2c956f4bc3")})


def verify_histories(sources):
    if set(sources) != set(REGISTRIES):
        raise ValueError("all four exclusion registries required")
    audit = {}
    for name, (count, spec, digest) in REGISTRIES.items():
        actual, payload = sources[name]
        if name in EXCLUSION_SOURCES:
            audit[name] = verify_exclusions(name, payload, actual)
            continue
        if (actual != digest or payload.get("spec_id") != spec or len(payload.get("records", [])) != count
                or not payload.get("ready_for_evaluation") or payload.get("qc_probe")
                or payload.get("selection_file") != "selection-v2.json"
                or payload.get("selection_sha256") != "89bba735a19cc2aad34666f6e5e3e92720113a82b5f5e96b98d571d6be1cd116"):
            raise ValueError("consumed Western registry identity/population changed")
        ids = set()
        for row in payload["records"]:
            if not row.get("sample_id") or row["sample_id"] in ids:
                raise ValueError("Western registry identities missing/duplicate")
            ids.add(row["sample_id"])
            b = row.get("bbox_wgs84", [])
            if (len(b) != 4 or not all(math.isfinite(v) for v in b)
                    or not -180 <= b[0] < b[2] <= 180 or not -90 <= b[1] < b[3] <= 90):
                raise ValueError("Western registry footprint invalid")
            hashes = list(row.get("hashes", {}).values())
            if not hashes or any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
                raise ValueError("Western registry input bindings missing")
        audit[name] = dict(spec_id=spec, records=count, sha256=digest)
    return audit


def balanced_frames(records, cap=72):
    groups = defaultdict(list)
    for r in records:
        if r["area_ha"] >= 1:
            groups[r["group_id"]].append(r)
    chosen = {}
    for group, rows in groups.items():
        bins = [deque(sorted((r for r in rows if (0 if r["area_ha"] < 25 else 1 if r["area_ha"] < 250 else 2) == i),
            key=lambda r: canonical_hash("spec112:"+r["sample_id"]))) for i in range(3)]
        chosen[group] = []
        while any(bins) and len(chosen[group]) < 4:
            for bucket in bins:
                if bucket and len(chosen[group]) < 4:
                    chosen[group].append(bucket.popleft())
    order = sorted(chosen, key=lambda g: canonical_hash("spec112:"+g))
    return [chosen[g][i] for i in range(4) for g in order if i < len(chosen[g])][:cap]
