"""Метрика кейса (SPEC-17/18): IoU гари и mIoU по степеням поражения.

**Усреднение микро, а не макро.** Постановка требует пула по всем пикселям
выборки, а не среднего из IoU отдельных чипов. Разница не косметическая: чип с
крошечной гарью в макро-среднем весит столько же, сколько чип, где выгорело
всё. Сравнивать числа, посчитанные разными способами, нельзя — на этом уже
получилось разойтись на 0.08 при сравнении бустинга с сетью.
"""

from __future__ import annotations

import numpy as np


def iou(truth: np.ndarray, pred: np.ndarray) -> float:
    union = (truth | pred).sum()
    return float((truth & pred).sum() / union) if union else float("nan")


def score_bs(truth: np.ndarray, pred: np.ndarray) -> dict:
    """IoU_burn (любой класс > 0) и mIoU по классам 1..3."""
    per_class = {cls: iou(truth == cls, pred == cls) for cls in (1, 2, 3)}
    present = [v for v in per_class.values() if not np.isnan(v)]
    return {
        "iou_burn": iou(truth > 0, pred > 0),
        "miou_sev": float(np.mean(present)) if present else float("nan"),
        "per_class": per_class,
    }


def score_bs_micro(truths, preds) -> dict:
    """Метрика по пулу пикселей всей выборки — так считает проверяющая система.

    Принимает последовательности масок; формы могут различаться.
    """
    truth = np.concatenate([np.asarray(t).reshape(-1) for t in truths])
    pred = np.concatenate([np.asarray(p).reshape(-1) for p in preds])
    return score_bs(truth, pred)
