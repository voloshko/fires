"""SPEC-101: full original supervision with paired source/mask exposure."""
import numpy as np

REFERENCE_RESULT_SHA256 = "242acac3cf2ccd30728e5919997895e7cf1d83de223a8802b06f3c4daaaa296d"


def epoch_view(epoch, has_twin, variant):
    if epoch < 1 or variant not in ("control", "replay"):
        raise ValueError("invalid replay schedule")
    if epoch % 2 or not has_twin:
        return "hls", "original"
    return ("l2a" if variant == "replay" else "hls"), "common"


def weighted_order(records, rng):
    weights = [r["sampling_weight"] for r in records]
    if any(type(w) is not int or w not in (1, 2) for w in weights):
        raise ValueError("invalid frozen sampling weights")
    return rng.permutation(np.repeat(np.arange(len(records)), weights))


def original_valid(image, labels):
    if image.shape != (6,) + labels.shape or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("original supervision shape/classes differ")
    return np.isfinite(image).all(0) & (image != -9999).all(0) & (labels >= 0)


def matched_view(image, labels, original, common=None):
    valid = original if common is None else original & common
    if image.shape != (6,) + labels.shape or valid.shape != labels.shape:
        raise ValueError("source/mask shapes differ")
    if (valid & (~np.isfinite(image).all(0) | (image == -9999).any(0))).any():
        raise ValueError("invalid source pixel inside supervision")
    return np.where(valid[None], image, 0).astype(np.float32), np.where(valid, labels, -1).astype(np.int8), valid
