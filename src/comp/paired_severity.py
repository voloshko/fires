"""SPEC-94: severity inside a separately fixed burn boundary."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.comp.forest_pair import ForestPairUNet


class PairedSeverityUNet(ForestPairUNet):
    def __init__(self, width=32, depth=7):
        super().__init__(width, depth)
        self.head = nn.Conv2d(width, 3, 1)


def mtbs_targets(mtbs, valid):
    mtbs, valid = np.asarray(mtbs), np.asarray(valid, dtype=bool)
    if mtbs.shape != valid.shape: raise ValueError("misaligned severity validity")
    labels = np.full(mtbs.shape, -1, dtype=np.int64)
    for raw, target in ((2, 0), (3, 1), (4, 2)):
        labels[(mtbs == raw) & valid] = target
    return labels


def fit_class_weights(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (3,) or not np.isfinite(counts).all() or np.any(counts <= 0):
        raise ValueError("all three severity classes need valid fit supervision")
    weights = 1 / np.sqrt(counts)
    return np.minimum(weights / weights.mean(), 4).astype(np.float32)


def severity_loss(logits, labels, weights):
    if logits.shape[1] != 3: raise ValueError("severity head must have three classes")
    known = labels >= 0
    if not known.any(): raise ValueError("batch has no severity supervision")
    weights = torch.as_tensor(weights, dtype=torch.float32, device=logits.device)
    if weights.shape != (3,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("invalid severity class weights")
    ce = F.cross_entropy(logits.float(), labels, ignore_index=-1, weight=weights)
    prob = logits.float().softmax(1).permute(0, 2, 3, 1)[known]
    truth = labels[known]
    dice = []
    for c in range(3):
        target = (truth == c).float()
        if target.any():
            p = prob[:, c]
            dice.append(1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1))
    return ce + torch.stack(dice).mean()


def apply_fixed_boundary(probabilities, boundary, valid):
    probabilities = np.asarray(probabilities)
    boundary, valid = np.asarray(boundary, dtype=bool), np.asarray(valid, dtype=bool)
    if probabilities.shape != (3,) + boundary.shape or valid.shape != boundary.shape:
        raise ValueError("misaligned fixed boundary")
    if not np.isfinite(probabilities).all(): raise ValueError("nonfinite severity probabilities")
    return np.where(boundary & valid, probabilities.argmax(0) + 1, 0).astype(np.uint8)
