"""SPEC-90: event isolation, source-bound caches and paired grouped evidence.

This module uses only the standard library so data audits need no GPU runtime.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


EVALUATION_LABELS = {
    "expert_perimeter", "participatory_perimeter", "mtbs_severity", "reviewed_negative"
}
PARTITIONS = {"fit", "calibration", "evaluation"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def utc_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp needs an explicit timezone")
    return parsed.astimezone(timezone.utc)


def event_group(dataset, record):
    """EMS AOIs share an activation; a distinct AOI is not a distinct event."""
    key = {"ems": "code", "floga": "id", "mtbs": "event_id",
           "russia": "cluster_id"}.get(dataset)
    if key is None or record.get(key) in (None, ""):
        raise ValueError(f"missing event group for {dataset}")
    return f"{dataset}:{record[key]}"


def validate_pair(record, as_of=None):
    pre, start, post = (utc_time(record[key]) for key in
                        ("pre_at", "start_at", "post_at"))
    if not pre < start <= post:
        raise ValueError("pair must satisfy pre < start <= post")
    if not record.get("mgrs") or record.get("pre_mgrs") != record["mgrs"]:
        raise ValueError("pair must use one MGRS tile")
    if not record.get("grid") or record.get("pre_grid") != record["grid"]:
        raise ValueError("pair grids differ")
    grid = record["grid"]
    if not grid.get("crs") or any(type(grid.get(k)) is not int or grid[k] <= 0
                                  for k in ("width", "height")):
        raise ValueError("invalid pair grid dimensions or CRS")
    transform = grid.get("transform", [])
    if len(transform) != 6 or not all(math.isfinite(v) for v in transform):
        raise ValueError("invalid pair affine transform")
    if transform[0] * transform[4] - transform[1] * transform[3] == 0:
        raise ValueError("singular pair grid")
    if as_of is not None:
        cutoff = utc_time(as_of)
        for acquisition, availability in ((pre, "pre_available_at"),
                                          (post, "post_available_at")):
            available = utc_time(record[availability])
            if acquisition > available or available > cutoff:
                raise ValueError("input was unavailable at the claimed map time")


def _bbox(record):
    bounds = record["bbox_wgs84"]
    if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
        raise ValueError("invalid bbox")
    west, south, east, north = bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("invalid bbox")
    return bounds


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def validate_manifest(records, require_pairs=False, as_of=None):
    if not records:
        raise ValueError("empty manifest")
    samples, groups, previous = set(), {}, []
    for record in records:
        sample = record.get("sample_id")
        group, part = record.get("group_id"), record.get("partition")
        if not sample or sample in samples:
            raise ValueError("missing or duplicate sample_id")
        if not group or part not in PARTITIONS:
            raise ValueError("missing group_id or invalid partition")
        if group in groups and groups[group] != part:
            raise ValueError("event crosses partitions")
        provenance = record.get("label_provenance")
        if provenance not in EVALUATION_LABELS | {"weak_candidate"}:
            raise ValueError("unknown label provenance")
        if part != "fit" and provenance not in EVALUATION_LABELS:
            raise ValueError("weak labels cannot decide evaluation or calibration")
        if provenance == "reviewed_negative" and not record.get("review_reference"):
            raise ValueError("negative supervision requires a review reference")
        bounds = _bbox(record)
        for other, other_part in previous:
            if part != other_part and _overlaps(bounds, other):
                raise ValueError("spatial footprints cross partitions")
        if require_pairs:
            validate_pair(record, as_of)
        samples.add(sample)
        groups[group] = part
        previous.append((bounds, part))
    return {"samples": len(samples), "groups": len(groups),
            "partitions": {p: sum(r["partition"] == p for r in records)
                           for p in sorted(PARTITIONS)}}


def experiment_fingerprint(records, input_files, model_files, source_files, preprocessing):
    """Lists retain sample/model order because they determine prediction order."""
    files = lambda paths: [{"path": str(p), "sha256": sha256_file(p)} for p in paths]
    return canonical_hash({"records": records, "inputs": files(input_files),
                           "models": files(model_files), "sources": files(source_files),
                           "preprocessing": preprocessing})


def bind_cache(cache_path, fingerprint):
    cache = Path(cache_path)
    meta = cache.with_name(cache.name + ".binding.json")
    if meta.exists():
        verify_cache(cache, fingerprint)
        return
    payload = {"fingerprint": fingerprint, "sha256": sha256_file(cache)}
    with meta.open("x") as stream:
        json.dump(payload, stream, sort_keys=True)


def verify_cache(cache_path, fingerprint):
    cache = Path(cache_path)
    with cache.with_name(cache.name + ".binding.json").open() as stream:
        meta = json.load(stream)
    if meta != {"fingerprint": fingerprint, "sha256": sha256_file(cache)}:
        raise ValueError("cache provenance mismatch")


def _checked_counts(counts):
    if len(counts) != 3 or any(not math.isfinite(v) or v < 0 or int(v) != v for v in counts):
        raise ValueError("counts must be three finite nonnegative integers")
    return tuple(int(v) for v in counts)


def _sum_counts(values):
    return tuple(sum(v[i] for v in values) for i in range(3))


def pooled_iou(values):
    tp, fp, fn = _sum_counts([_checked_counts(v) for v in values])
    if tp + fp + fn == 0:
        raise ValueError("IoU is undefined for an empty union")
    return tp / (tp + fp + fn)


def _percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def paired_group_comparison(groups, candidate, control, n_boot=2000, seed=20261003):
    if not groups or len(groups) != len(candidate) or len(groups) != len(control):
        raise ValueError("unaligned paired inputs")
    if n_boot < 2 or any(not group for group in groups):
        raise ValueError("invalid bootstrap configuration")
    a, b = defaultdict(list), defaultdict(list)
    for group, ca, cb in zip(groups, candidate, control):
        a[group].append(_checked_counts(ca))
        b[group].append(_checked_counts(cb))
    keys = sorted(a)
    if len(keys) < 2:
        raise ValueError("at least two independent groups required")
    ga, gb = [_sum_counts(a[k]) for k in keys], [_sum_counts(b[k]) for k in keys]
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choices(range(len(keys)), k=len(keys))
        deltas.append(pooled_iou([ga[i] for i in idx]) - pooled_iou([gb[i] for i in idx]))
    ai, bi = pooled_iou(ga), pooled_iou(gb)
    lo, hi = _percentile(deltas, .025), _percentile(deltas, .975)
    return {"samples": len(groups), "groups": len(keys), "candidate_iou": ai,
            "control_iou": bi, "delta": ai - bi, "ci95": [lo, hi],
            "n_boot": n_boot, "seed": seed,
            "verdict": "better" if lo > 0 else "worse" if hi < 0 else "inconclusive"}
