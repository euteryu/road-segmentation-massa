# src/postprocess.py
"""Morphological clean-up of predicted road masks.

Two cheap operations that exploit priors the network does not enforce:

  closing  - roads are continuous. A dilate-then-erode pass bridges the small
             gaps left where a tree or a vehicle occludes the road, without
             fattening the road overall (the erode undoes the dilate everywhere
             except across gaps narrower than the kernel).

  min_area - roads are long. An isolated 20-pixel blob in the middle of a field
             is almost certainly a rooftop or a shadow, not a road, so drop
             connected components below an area floor.

Both raise precision and connectivity for zero GPU time. Neither can invent a
road the model missed entirely - post-processing polishes, it does not rescue.
"""
import cv2
import numpy as np


def postprocess_mask(mask, close_kernel: int = 5, min_area: int = 64) -> np.ndarray:
    """Args: boolean/0-1 array (H, W). Returns: boolean array (H, W)."""
    out = (np.asarray(mask) > 0).astype(np.uint8)

    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    if min_area and min_area > 0:
        # Connectivity 8 so diagonal road pixels stay one component - roads are
        # frequently 1px-wide diagonals at this resolution.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        keep = np.zeros(n_labels, dtype=bool)
        for label in range(1, n_labels):  # 0 is background
            keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_area
        out = keep[labels].astype(np.uint8)

    return out.astype(bool)
