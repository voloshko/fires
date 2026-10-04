"""SPEC-103: one fixed, GT-blind small-component safeguard."""
import numpy as np
from scipy.ndimage import label
from .pair_fusion import fuse


def rescue(reference,paired,valid,water):
    mean=fuse(reference,paired,"mean")
    valid,water=np.asarray(valid,bool),np.asarray(water,bool)
    if mean.ndim!=2 or valid.shape!=mean.shape or water.shape!=mean.shape:
        raise ValueError("component masks/grids differ")
    observed=valid&~water
    baseline=(reference>=.5)&observed;prediction=(mean>=.5)&observed
    labels,n=label(baseline,structure=np.ones((3,3),bool))
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    selected=(sizes>=1)&(sizes<=64);selected[0]=False
    selected[np.unique(labels[prediction&baseline])]=False
    restored=selected[labels]
    return prediction|restored,dict(restored_components=int(selected.sum()),restored_pixels=int(restored.sum()))


def eligible(russia,mediterranean):
    if set(mediterranean)!={"ems","floga"}:raise ValueError("missing stability cohort")
    return bool(russia["delta"]>=.01 and all(c["delta"]>=-.01 and c["candidate_missed_groups"]<=c["baseline_missed_groups"]
                                            for c in mediterranean.values()))
