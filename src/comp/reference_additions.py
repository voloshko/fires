"""SPEC-104: retain reference positives, add only fixed .95 paired positives."""
import numpy as np
from .pair_fusion import fuse


def additions(reference,paired,valid,water):
    fuse(reference,paired,"mean")  # Shared strict probability/grid validation.
    valid,water=np.asarray(valid,bool),np.asarray(water,bool)
    if reference.ndim!=2 or valid.shape!=reference.shape or water.shape!=reference.shape:
        raise ValueError("addition masks/grids differ")
    observed=valid&~water
    baseline=(reference>=.5)&observed
    added=(paired>=.95)&observed&~baseline
    return baseline|added,dict(added_pixels=int(added.sum()),reference_pixels=int(baseline.sum()))
