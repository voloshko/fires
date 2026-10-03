"""SPEC-98: fresh-event selection and immutable model checks (stdlib only)."""
from .accuracy_contract import sha256_file

DEVELOPMENT_SHA256 = "5f6b2839b89c807692dc4c931b2f4e713ae03294f869cd5233faf89a5617999e"


def verify_development_result(path):
    if sha256_file(path) != DEVELOPMENT_SHA256:
        raise ValueError("development result differs from committed SPEC94 evidence")


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def isolate(candidates, used_events, prior_frames, limit=180):
    selected, excluded = [], []
    for candidate in candidates:
        event, bounds = candidate["event_id"], candidate["bbox_wgs84"]
        reason = None
        if event in used_events:
            reason = "previously used event"
        elif any(overlaps(bounds, r["bbox_wgs84"]) for r in prior_frames):
            reason = "intersects previous frame"
        elif any(event == r["event_id"] or overlaps(bounds, r["bbox_wgs84"]) for r in selected):
            reason = "intersects selected fresh event"
        if reason:
            excluded.append(dict(event_id=event, reason=reason))
        elif len(selected) < limit:
            selected.append(candidate)
    return selected, excluded


def verify_frozen_models(paths, expected_hashes):
    if len(paths) != len(expected_hashes) or not paths:
        raise ValueError("frozen model population differs")
    for path, expected in zip(paths, expected_hashes):
        if sha256_file(path) != expected:
            raise ValueError("model differs from frozen development result")


def assert_fresh(records, plan):
    selected = {r["event_id"]: r for r in plan["selected"]}
    used = set(plan["used_events"])
    seen = set()
    for record in records:
        event = record["group_id"].removeprefix("mtbs:")
        if event in used or event in seen or event not in selected:
            raise ValueError("evaluation event is not a distinct frozen fresh event")
        if record["grid"] != selected[event]["grid"]:
            raise ValueError("evaluation frame differs from frozen selection")
        if record["partition"] != "evaluation":
            raise ValueError("fresh confirmation cannot contain fit records")
        seen.add(event)
