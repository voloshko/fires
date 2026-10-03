"""SPEC-92: explicit scene/cluster diagnostics, never fabricated fire counts."""
import math


def summarize(rows, models):
    if not rows or not models:
        raise ValueError("empty error-report population")
    seen, true_pixels, valid_pixels, total_pixels = set(), 0, 0, 0
    totals = {m: [0, 0, 0] for m in models}
    groups = {m: {} for m in models}
    missed = {m: [] for m in models}
    for row in rows:
        sample = row["sample_id"]
        if sample in seen: raise ValueError("duplicate diagnostic sample")
        seen.add(sample)
        valid, pixels = row["valid_pixels"], row["total_pixels"]
        if not 0 <= valid <= pixels or pixels <= 0:
            raise ValueError("invalid observation population")
        truth = None
        for model in models:
            counts = row[model]
            if len(counts) != 3 or any(not math.isfinite(v) or v < 0 or int(v) != v for v in counts):
                raise ValueError("invalid confusion counts")
            tp, fp, fn = counts
            if tp+fp+fn > valid: raise ValueError("counts exceed observed population")
            if truth is None: truth = tp+fn
            if tp+fn != truth: raise ValueError("models do not share ground-truth pixels")
            totals[model] = [a+b for a,b in zip(totals[model], counts)]
            group = groups[model].setdefault(row["group_id"], [0, 0])
            group[0] += tp; group[1] += tp+fn
            if tp == 0 and tp+fn > 0: missed[model].append(sample)
        true_pixels += truth; valid_pixels += valid; total_pixels += pixels
    negative = valid_pixels-true_pixels
    report = {}
    for model, (tp,fp,fn) in totals.items():
        denominator = tp+fp+fn
        report[model] = dict(tp=tp,fp=fp,fn=fn,iou=tp/denominator if denominator else None,
                            fp_fraction_of_known_background=fp/negative if negative else None,
                            missed_positive_windows=missed[model],
                            missed_positive_groups=[g for g,(hits,true) in groups[model].items() if hits==0 and true>0])
    return dict(windows=len(rows),groups=len({r["group_id"] for r in rows}),
                valid_pixels=valid_pixels,true_pixels=true_pixels,known_background_pixels=negative,
                masked_fraction=1-valid_pixels/total_pixels,models=report)


def diagnostic_subset(rows, reference="c1mm"):
    """Frozen top25% reference FP ordering; descriptive, never a tuning set."""
    ordered = sorted(rows,key=lambda r:(-r[reference][1],r["sample_id"]))
    return ordered[:math.ceil(len(rows)*.25)]


def size_bin(area):
    if not math.isfinite(area) or area < 1: raise ValueError("invalid seed area")
    return "1-25ha" if area < 25 else "25-250ha" if area < 250 else "250ha+"
