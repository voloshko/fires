"""SPEC-94: severity inside a separately fixed burn boundary."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from src.comp.forest_pair import ForestPairUNet
from src.comp.accuracy_contract import _percentile


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


def mtbs_evaluation_labels(mtbs, valid):
    """Known background counts in evaluation even though it does not train severity."""
    raw, valid = np.asarray(mtbs), np.asarray(valid, dtype=bool)
    if raw.shape != valid.shape: raise ValueError("unaligned evaluation labels")
    labels = np.full(raw.shape, -1, np.int64)
    labels[np.isin(raw, [0, 1, 5]) & valid] = 0
    for source, target in ((2, 1), (3, 2), (4, 3)):
        labels[(raw == source) & valid] = target
    return labels


def severity_confusion(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape: raise ValueError("unaligned severity predictions")
    known = truth >= 0
    if np.any(truth[known] > 3) or np.any(prediction[known] < 0) or np.any(prediction[known] > 3):
        raise ValueError("invalid severity class")
    return np.bincount((4 * truth[known] + prediction[known]).astype(np.int64), minlength=16).reshape(4, 4)


def severity_metrics(confusion):
    cm = np.asarray(confusion, np.float64)
    if cm.shape != (4, 4) or not np.isfinite(cm).all() or np.any(cm < 0) or np.any(cm != np.floor(cm)):
        raise ValueError("invalid confusion")
    tp = np.diag(cm)[1:]; union = cm.sum(0)[1:] + cm.sum(1)[1:] - tp
    if np.any(union == 0): raise ValueError("severity IoU undefined for an absent class")
    per_class = tp / union
    burn_tp, burn_fp, burn_fn = cm[1:, 1:].sum(), cm[0, 1:].sum(), cm[1:, 0].sum()
    denominator = burn_tp + burn_fp + burn_fn
    if denominator == 0: raise ValueError("undefined burned-area IoU")
    return dict(miou=float(per_class.mean()), per_class=per_class.tolist(),
                burn_iou=float(burn_tp / denominator))


def grouped_severity_comparison(groups, candidate, baseline, n_boot=2000, seed=20261003):
    import random
    if not groups or len(groups) != len(candidate) or len(groups) != len(baseline):
        raise ValueError("unaligned severity groups")
    if n_boot < 2 or any(not group for group in groups): raise ValueError("invalid severity bootstrap")
    a, b = {}, {}
    for group, ca, cb in zip(groups, candidate, baseline):
        for cm in (ca, cb):
            cm = np.asarray(cm)
            if cm.shape != (4, 4) or not np.isfinite(cm).all() or np.any(cm < 0) or np.any(cm != np.floor(cm)):
                raise ValueError("invalid window confusion")
        a[group] = a.get(group, np.zeros((4, 4))) + ca
        b[group] = b.get(group, np.zeros((4, 4))) + cb
    keys = sorted(a)
    if len(keys) < 2: raise ValueError("at least two severity groups required")
    ga, gb = np.stack([a[k] for k in keys]), np.stack([b[k] for k in keys])
    rng = random.Random(seed); deltas = []
    for _ in range(n_boot):
        idx = rng.choices(range(len(keys)), k=len(keys))
        deltas.append(severity_metrics(ga[idx].sum(0))["miou"] - severity_metrics(gb[idx].sum(0))["miou"])
    ma, mb = severity_metrics(ga.sum(0)), severity_metrics(gb.sum(0))
    lo, hi = _percentile(deltas, .025), _percentile(deltas, .975)
    return dict(samples=len(groups), groups=len(keys), candidate=ma, baseline=mb,
                delta=ma["miou"] - mb["miou"], ci95=[lo, hi], n_boot=n_boot, seed=seed)
