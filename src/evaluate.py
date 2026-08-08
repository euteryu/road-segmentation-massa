# src/evaluate.py
"""Score saved probability maps. Pure CPU - no GPU, no model, no retraining.

Reports the metric at three stages so you can see what each step actually bought:

    raw          IoU at the default 0.5 threshold
    tuned        IoU at the best threshold found on validation
    postproc     IoU after morphological clean-up at that threshold

Run standalone against an existing run:
    python -m src.evaluate --name mit_b0_scaled
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.data.splits import make_split, resolve_data_root
from src.metrics import BinaryConfusion, ThresholdSweep
from src.postprocess import postprocess_mask
from src.utils import load_config, save_json


def _load_pair(pred_dir: Path, data_root: str, img_id: str):
    prob = cv2.imread(str(pred_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
    if prob is None:
        raise FileNotFoundError(pred_dir / f"{img_id}.png")
    gt = cv2.imread(str(Path(data_root) / f"{img_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(Path(data_root) / f"{img_id}_mask.png")
    return prob.astype(np.float32) / 255.0, gt > 128


def evaluate_predictions(pred_dir, data_root, val_ids, close_kernel=5, min_area=64,
                         thresholds=None) -> dict:
    pred_dir = Path(pred_dir)

    sweep = ThresholdSweep(thresholds)
    raw = BinaryConfusion()
    for img_id in tqdm(val_ids, desc="scoring", leave=False):
        prob, gt = _load_pair(pred_dir, data_root, img_id)
        raw.update(prob >= 0.5, gt)
        sweep.update(prob, gt)

    best_thr, tuned = sweep.best()

    post = BinaryConfusion()
    for img_id in tqdm(val_ids, desc="post-processing", leave=False):
        prob, gt = _load_pair(pred_dir, data_root, img_id)
        post.update(postprocess_mask(prob >= best_thr, close_kernel, min_area), gt)

    return {
        "n_val_images": len(val_ids),
        "best_threshold": best_thr,
        "threshold_curve": sweep.curve(),
        "postproc": {"close_kernel": close_kernel, "min_area": min_area},
        "stages": {
            "raw@0.50": raw.as_dict(),
            f"tuned@{best_thr:.2f}": tuned.as_dict(),
            f"postproc@{best_thr:.2f}": post.as_dict(),
        },
        "final_iou": round(post.iou, 4),
        "gain_from_threshold": round(tuned.iou - raw.iou, 4),
        "gain_from_postproc": round(post.iou - tuned.iou, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Score saved predictions (CPU only).")
    parser.add_argument("--name", required=True, help="run name, e.g. mit_b0_scaled")
    parser.add_argument("--config", default=None, help="defaults to configs/<name>.yaml")
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config or f"configs/{args.name}.yaml")
    cfg["data_root"] = resolve_data_root(cfg["data_root"])
    _, val_ids = make_split(cfg["data_root"], cfg["train_size"], cfg["val_size"], cfg["seed"])

    report = evaluate_predictions(
        Path("outputs/preds") / args.name,
        cfg["data_root"],
        val_ids,
        close_kernel=args.close_kernel,
        min_area=args.min_area,
    )
    save_json(report, Path("outputs/results") / f"{args.name}_eval.json")
    print(json.dumps(report["stages"], indent=2))


if __name__ == "__main__":
    main()
