# src/metrics.py
"""Segmentation metrics.

The important distinction here, because it changes the numbers a lot:

  dataset IoU  - accumulate TP/FP/FN over the WHOLE validation set, divide once.
                 This is what the DeepGlobe leaderboard and the papers report.
  per-image IoU - compute IoU per image, then average. Images with almost no road
                 count as much as dense ones, so this reads differently (usually
                 lower on this dataset). Reported alongside for continuity with
                 the project's earlier runs, but `dataset IoU` is the headline.
"""
import numpy as np
import torch


class BinaryConfusion:
    """Streaming TP/FP/FN accumulator for a single decision threshold."""

    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, preds, targets):
        """preds/targets: bool or {0,1} arrays/tensors of identical shape."""
        if isinstance(preds, torch.Tensor):
            preds = preds.bool()
            targets = targets.bool()
            self.tp += torch.count_nonzero(preds & targets).item()
            self.fp += torch.count_nonzero(preds & ~targets).item()
            self.fn += torch.count_nonzero(~preds & targets).item()
        else:
            preds = preds.astype(bool)
            targets = targets.astype(bool)
            self.tp += float(np.count_nonzero(preds & targets))
            self.fp += float(np.count_nonzero(preds & ~targets))
            self.fn += float(np.count_nonzero(~preds & targets))

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "iou": round(self.iou, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


class ThresholdSweep:
    """Score many thresholds in one pass over the data.

    Picking the operating point on validation is free accuracy: the model's 0.5
    default is rarely optimal when the positive class is ~5-10% of pixels.
    """

    def __init__(self, thresholds=None):
        self.thresholds = list(thresholds) if thresholds is not None else [
            round(t, 2) for t in np.arange(0.20, 0.81, 0.05)
        ]
        self.meters = {t: BinaryConfusion() for t in self.thresholds}

    def update(self, probs, targets):
        for t, meter in self.meters.items():
            meter.update(probs >= t, targets)

    def best(self):
        """-> (threshold, BinaryConfusion) with the highest dataset IoU."""
        t = max(self.thresholds, key=lambda t: self.meters[t].iou)
        return t, self.meters[t]

    def curve(self) -> dict:
        return {str(t): round(m.iou, 4) for t, m in self.meters.items()}


def per_image_iou(preds, targets, eps: float = 1e-6) -> float:
    """Mean over images of per-image IoU. Kept to stay comparable with the
    project's earlier reported numbers - see module docstring."""
    preds = preds.float()
    targets = targets.float()
    dims = tuple(range(1, preds.dim()))
    inter = (preds * targets).sum(dim=dims)
    union = ((preds + targets) > 0).float().sum(dim=dims)
    return ((inter + eps) / (union + eps)).mean().item()
