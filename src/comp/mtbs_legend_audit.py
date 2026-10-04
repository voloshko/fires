"""SPEC-110: attribute legacy errors; never change benchmark truth/predictions."""
import numpy as np
from .paired_severity import mtbs_evaluation_labels, severity_confusion


def attribute(raw, prediction, valid):
    raw, prediction, valid = np.asarray(raw), np.asarray(prediction), np.asarray(valid, bool)
    if raw.shape != prediction.shape or raw.shape != valid.shape or raw.ndim != 2:
        raise ValueError("legend audit grids differ")
    if (not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw != np.floor(raw))
            or not np.isfinite(prediction).all() or np.any(prediction < 0) or np.any(prediction > 3)
            or np.any(prediction != np.floor(prediction))):
        raise ValueError("invalid raw code/prediction")
    truth = mtbs_evaluation_labels(raw, valid)
    by_code = {}
    reconstruction = np.zeros((4, 4), np.int64)
    for code in sorted(set(raw[valid].tolist())):
        subset = valid & (raw == code)
        known = subset & (truth >= 0)
        cm = severity_confusion(np.where(subset, truth, -1), prediction)
        reconstruction += cm
        by_code[str(int(code))] = dict(observed_pixels=int(subset.sum()), legacy_known_pixels=int(known.sum()),
            damage_predictions=[int((subset & (prediction == c)).sum()) for c in (1, 2, 3)],
            legacy_errors=int((known & (truth != prediction)).sum()) if known.any() else None,
            legacy_confusion=cm.tolist())
    expected = severity_confusion(truth, prediction)
    if not np.array_equal(reconstruction, expected):
        raise ValueError("raw-code decomposition does not recombine")
    total = raw.size
    return dict(by_raw_code=by_code, legacy_confusion=expected.tolist(),
        total_pixels=total, observed_pixels=int(valid.sum()), observed_fraction=float(valid.mean()),
        assessed_1_to_5_pixels=int((valid & np.isin(raw, [1, 2, 3, 4, 5])).sum()),
        graded_2_to_4_pixels=int((valid & np.isin(raw, [2, 3, 4])).sum()),
        raw0_outside_assessment_pixels=int((valid & (raw == 0)).sum()),
        raw5_increased_greenness_pixels=int((valid & (raw == 5)).sum()),
        unknown_codes=sorted(int(c) for c in set(raw[valid].tolist()) if c not in range(7)))
