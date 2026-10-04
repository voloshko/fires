"""SPEC-114: equal background/fire CE, unchanged foreground Dice."""
import torch
import torch.nn.functional as F

LOSS_ID = "equal-ce-dice-v1"


def equal_cost_loss(logits, labels):
    known = labels >= 0
    if not known.any():
        raise ValueError("batch has no supervision")
    ce = F.cross_entropy(logits.float(), labels, ignore_index=-1)
    probability = logits.float().softmax(1)[:, 1][known]
    truth = (labels[known] == 1).float()
    return ce + 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)


def validate_equal_cost_binding(bundle, base_sha256, variant):
    if (bundle.get("spec_id") != "SPEC-114" or bundle.get("loss_id") != LOSS_ID
            or bundle.get("base_sha256") != base_sha256
            or bundle.get("variant") != variant or bundle.get("rank") != 8):
        raise ValueError("equal-cost adapter loss/base/variant binding differs")
