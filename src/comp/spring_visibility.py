"""SPEC-108: fixed metadata ordering and same-pixel temporal diagnosis."""
import math

import numpy as np

from .accuracy_contract import utc_time
from .geographic_failure_report import normalized_ratio, spectral_summary

BINS = (("2020-03-01", "2020-03-15"), ("2020-03-16", "2020-03-31"),
        ("2020-04-01", "2020-04-30"), ("2020-05-01", "2020-05-15"))
BUDGET_BYTES = 128*1024**2
FREE_FLOOR_BYTES = 3*1024**3
RESERVATION_BYTES = 8*1024**2


def candidates(rows, tile, lower, upper):
    start, end = utc_time(lower+"T00:00:00Z"), utc_time(upper+"T23:59:59.999999Z")
    selected = []
    for row in rows:
        cloud = row.get("cloud")
        if (isinstance(cloud, bool) or not isinstance(cloud, (int, float))
                or not math.isfinite(cloud) or not 0 <= cloud <= 80 or row.get("mgrs") != tile):
            continue
        try:
            acquired = utc_time(row["date"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= acquired <= end:
            selected.append(row)
    return sorted(selected, key=lambda r: (r["cloud"], utc_time(r["date"]), r["item_id"]))[:3]


def storage_guard(used, free):
    if used < 0 or free < 0 or used+RESERVATION_BYTES > BUDGET_BYTES:
        raise ValueError("visibility storage budget unavailable")
    if free-RESERVATION_BYTES < FREE_FLOOR_BYTES:
        raise ValueError("visibility free-space floor unavailable")


def compare_view(old, current, labels, old_valid, current_valid):
    if old.shape != current.shape or old.shape != (6,)+labels.shape:
        raise ValueError("six-band observation grids differ")
    if old_valid.shape != labels.shape or current_valid.shape != labels.shape:
        raise ValueError("validity grid differs")
    common = old_valid & current_valid & (labels >= 0)
    true, background = common & (labels == 1), common & (labels == 0)
    report = dict(current_valid_fraction=float(current_valid.mean()),
        common_valid_fraction=float(common.mean()), masked_fraction=1-float(common.mean()),
        truth_pixels=int(true.sum()), background_pixels=int(background.sum()))
    report["usable"] = (report["current_valid_fraction"] >= .5 and
        report["common_valid_fraction"] >= .5 and true.sum() >= 9 and background.sum() >= 9)
    indices = {}
    for name, image in (("old", old), ("current", current)):
        indices[name+"_nbr"] = normalized_ratio(image[3], image[5])
        indices[name+"_ndvi"] = normalized_ratio(image[3], image[2])
    report["labelled_burn"] = spectral_summary(indices, true)
    report["published_background"] = spectral_summary(indices, background)
    contrasts = {}
    for name in ("old", "current"):
        a = report["labelled_burn"]["indices"][name+"_nbr"]["q10_median_q90"]
        b = report["published_background"]["indices"][name+"_nbr"]["q10_median_q90"]
        contrasts[name] = a[1]-b[1] if a is not None and b is not None else None
    report["contrast"] = contrasts
    report["extra_separation"] = bool(report["usable"] and all(v is not None for v in contrasts.values())
        and contrasts["current"] <= -.1 and contrasts["old"]-contrasts["current"] >= .1)
    return report


def decision(observations, sample_ids):
    if len(sample_ids) != 4 or len(set(sample_ids)) != 4:
        raise ValueError("visibility probe requires four fixed missed windows")
    frames = []
    for sample in sample_ids:
        rows = [r for r in observations if r["sample_id"] == sample and r.get("usable")]
        dates = [utc_time(r["date"]) for r in rows]
        span = (max(dates)-min(dates)).total_seconds()/86400 if dates else None
        assessable = len(set(dates)) >= 2 and span >= 14
        frames.append(dict(sample_id=sample, usable_dates=len(set(dates)), span_days=span,
            assessable=bool(assessable), extra_separation=bool(assessable and any(r["extra_separation"] for r in rows))))
    assessed = sum(r["assessable"] for r in frames)
    supported = sum(r["extra_separation"] for r in frames)
    status = "INSUFFICIENT_EVIDENCE" if assessed < 2 else "PASS" if supported >= 2 else "FAIL"
    return dict(status=status, assessable_frames=assessed, supported_frames=supported,
        supports_multidate_followup=status == "PASS", frames=frames)
