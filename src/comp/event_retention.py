"""SPEC-97: conservative event retention without invented unknown IDs."""
import numpy as np


def pixel_survival(proof, valid):
    proof, valid = np.asarray(proof), np.asarray(valid, dtype=bool)
    if proof.shape != valid.shape or not np.isin(proof, [0, 1, 2]).all():
        raise ValueError("invalid retention proof")
    return {"known_event_survives": bool(((proof == 1) & valid).any()),
            "protected_pixels": int((proof == 2).sum()),
            "lost_protected_pixels": int(((proof == 2) & ~valid).sum())}


def conservative_bound(original, observed):
    """Original population is immutable; missing windows count as unobserved."""
    known, retained, lost = set(), set(), []
    for record in original:
        sample = record["sample_id"]; event = record.get("event_key")
        if event: known.add(event)
        result = observed.get(sample)
        if result is None:
            if record["protected_pixels"]:
                lost.append(dict(sample_id=sample, lost_protected_pixels=record["protected_pixels"]))
            continue
        if result["protected_pixels"] != record["protected_pixels"]:
            raise ValueError("original protected population changed")
        if result["lost_protected_pixels"]:
            lost.append(dict(sample_id=sample, lost_protected_pixels=result["lost_protected_pixels"]))
        if event and result["known_event_survives"]: retained.add(event)
    if not known: raise ValueError("no audited original events")
    guaranteed = not lost
    return dict(original_known_events=len(known), retained_known_events=len(retained),
                unresolved_guaranteed=guaranteed, lost_protected=lost,
                event_retention_lower_bound=len(retained) / len(known) if guaranteed else None,
                accepted=guaranteed and len(retained) / len(known) >= .7)
