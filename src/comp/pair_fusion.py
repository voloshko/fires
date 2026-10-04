"""SPEC-102: two fixed fusions and a selection rule frozen before confirmation."""
import numpy as np

RECIPES=("mean","maximum")


def fuse(reference,paired,recipe):
    reference,paired=np.asarray(reference),np.asarray(paired)
    if reference.shape!=paired.shape or not reference.size or reference.ndim<2:
        raise ValueError("fusion inputs have different grids/populations")
    for values in (reference,paired):
        if not np.isfinite(values).all() or values.min() < -1e-6 or values.max()>1+1e-6:
            raise ValueError("invalid probability input")
    if recipe=="mean":return (reference+paired)*.5
    if recipe=="maximum":return np.maximum(reference,paired)
    raise ValueError("fusion recipe was not preregistered")


def select_recipe(reports):
    if set(reports)!=set(RECIPES):raise ValueError("all preregistered candidates must be reported")
    eligible=[]
    for name,report in reports.items():
        allowed=report["russian_vs_c1mm"]["delta"]>=.01
        allowed &= all(r["delta"]>=-.01 and r["candidate_missed_groups"]<=r["baseline_missed_groups"]
                       for r in report["mediterranean"].values())
        if allowed:eligible.append(name)
    return sorted(eligible,key=lambda n:(-reports[n]["russian_vs_c1mm"]["candidate_iou"],n))[0] if eligible else None


def confirmation_acceptance(comparisons,missed):
    if set(comparisons)!={"c1mm","prithvi","reserve"}:
        raise ValueError("both competitors and paired control are required")
    accepted=all(comparisons[k]["delta"]>=.01 and comparisons[k]["ci95"][0]>0 for k in ("c1mm","prithvi"))
    accepted &= comparisons["reserve"]["delta"]>=-.01
    accepted &= missed["fusion"]<=missed["c1mm"]
    return bool(accepted)
