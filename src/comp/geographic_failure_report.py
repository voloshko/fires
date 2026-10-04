"""SPEC-106: descriptive evidence from consumed, fixed predictions."""
from collections import defaultdict
import math

import numpy as np

from .accuracy_contract import canonical_hash, pooled_iou
from .russian_error_report import summarize


def verify_recipe(recipe, expected):
    content = {k: v for k, v in recipe.items() if k != "recipe_id"}
    if recipe.get("recipe_id") != expected or canonical_hash(content) != expected:
        raise ValueError("parent recipe content differs")


def verify_rows(records, rows, reconstructed):
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids) or ids != [r["sample_id"] for r in rows]:
        raise ValueError("diagnostic identity/order mismatch")
    if [r["group_id"] for r in records] != [r["group_id"] for r in rows]:
        raise ValueError("diagnostic group mismatch")
    expected = [{k: v for k, v in r.items() if k != "total_pixels"} for r in rows]
    if expected != reconstructed:
        raise ValueError("diagnostic counts differ from completed parent")


def group_influence(rows, models=("reserve", "c1mm", "prithvi")):
    summarize(rows, models)  # Validate shared truth/observation and identities.
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(groups) < 2:
        raise ValueError("group influence needs at least two groups")
    totals = {m: [sum(r[m][i] for r in rows) for i in range(3)] for m in models}
    baseline = pooled_iou([totals["reserve"]]) - pooled_iou([totals["c1mm"]])
    report = []
    for group, members in sorted(groups.items()):
        cm = {m: [sum(r[m][i] for r in members) for i in range(3)] for m in models}
        remainder = {m: [a-b for a, b in zip(totals[m], cm[m])] for m in models}
        delta = pooled_iou([remainder["reserve"]]) - pooled_iou([remainder["c1mm"]])
        report.append(dict(group_id=group, windows=len(members), counts=cm,
            iou={m: pooled_iou([cm[m]]) for m in models},
            true_pixels=cm["reserve"][0]+cm["reserve"][2],
            delta_without_group=delta, delta_change=delta-baseline))
    return report


def normalized_ratio(a, b):
    denominator = a+b
    result = np.full(a.shape, np.nan, np.float32)
    np.divide(a-b, denominator, out=result, where=np.abs(denominator) > 1e-8)
    return result


def spectral_indices(pre, post):
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[0] != 6:
        raise ValueError("six-band paired grid required")
    pnbr = normalized_ratio(pre[3], pre[5])
    anbr = normalized_ratio(post[3], post[5])
    return dict(pre_nbr=pnbr, post_nbr=anbr, dnbr=pnbr-anbr,
        pre_ndvi=normalized_ratio(pre[3], pre[2]),
        post_ndvi=normalized_ratio(post[3], post[2]),
        post_ndwi=normalized_ratio(post[1], post[3]))


def spectral_summary(indices, mask):
    output = dict(pixels=int(mask.sum()), indices={})
    for name, values in indices.items():
        if values.shape != mask.shape:
            raise ValueError("spectral mask grid mismatch")
        selected = values[mask]
        finite = selected[np.isfinite(selected)]
        quantiles = np.quantile(finite, [.1, .5, .9]).tolist() if len(finite) else None
        output["indices"][name] = dict(finite_pixels=len(finite),
            undefined_pixels=len(selected)-len(finite), q10_median_q90=quantiles)
    return output


def diagnostic_masks(labels, valid, water, reserve, reference):
    shape = labels.shape
    if any(x.shape != shape for x in (valid, water, reserve, reference)):
        raise ValueError("diagnostic grids differ")
    if any(not np.isfinite(p[valid]).all() for p in (reserve, reference)):
        raise ValueError("nonfinite observed probability")
    observed = valid & (labels >= 0)
    true = observed & (labels == 1)
    a, b = (reserve >= .5) & ~water, (reference >= .5) & ~water
    return dict(labelled_burn=true,
        reserve_only_fp=observed & ~true & a & ~b,
        c1mm_only_fp=observed & ~true & b & ~a)


def probability_max(probability, mask):
    if not mask.any():
        return None
    value = float(probability[mask].max())
    if not math.isfinite(value):
        raise ValueError("nonfinite truth probability")
    return value
