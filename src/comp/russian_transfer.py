"""SPEC-92: split-safe background mining from published interval perimeters."""
import math
import numpy as np


def hard_background_weights(records, false_positive_counts):
    if any(r["partition"] != "fit" for r in records):
        raise ValueError("hard-negative mining may only see fit records")
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(false_positive_counts):
        raise ValueError("mining counts do not match the fit population")
    if any(not np.isfinite(v) or v < 0 or int(v) != v for v in false_positive_counts.values()):
        raise ValueError("invalid false-positive counts")
    ranked = sorted(ids, key=lambda sid: (-false_positive_counts[sid], sid))
    hard = set(ranked[:math.ceil(len(ranked) * .25)])
    return {sid: 2 if sid in hard else 1 for sid in ids}


def transfer_acceptance(russian_comparison, stability, competitor):
    accepted = russian_comparison["delta"] >= .01 and russian_comparison["ci95"][0] > 0
    accepted &= all(c["delta"] >= -.01 for c in stability.values())
    competitor_won = competitor["delta"] > 0 and competitor["ci95"][0] > 0
    return dict(accepted=bool(accepted), competitor_superiority=bool(competitor_won))
