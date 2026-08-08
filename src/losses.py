# src/losses.py
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Weighted sum of pixelwise BCE and (1 - soft Dice).

    Why the mix, and why we lean Dice-heavy for roads:

    BCE grades every pixel independently and equally. On a 256x256 road crop
    roughly 5-10% of pixels are road, so a model that predicts "background"
    everywhere already scores well on ~90% of pixels - the gradient from the few
    road pixels gets diluted into noise.

    Dice grades region *overlap* as a ratio, so predicting nothing scores 0 no
    matter how much background it got right. There is nowhere to hide, and the
    gradient pushes specifically toward recovering positive pixels (recall).
    Thin structures make BCE's dilution worse and Dice's framing more corrective,
    hence bce_weight < 0.5 as the default for this dataset.
    """

    def __init__(self, bce_weight: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        if not 0.0 <= bce_weight <= 1.0:
            raise ValueError(f"bce_weight must be in [0, 1], got {bce_weight}")
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def build_criterion(cfg: dict) -> nn.Module:
    return BCEDiceLoss(bce_weight=cfg.get("bce_weight", 0.3))
