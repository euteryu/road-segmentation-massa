# src/sweep_postproc.py
"""Grid-search the post-processing constants against already-saved predictions.

Why this exists: `min_area: 64` and `close_kernel: 5` were chosen when inference
ran on 256px crops, but inference now runs on full 1024px images. Area scales
with the square of resolution, so a 64px floor at 1024px drops essentially
nothing - which is exactly what E1/E2 measured (+0.0006, +0.0009). The stage has
never been tested at a size that could plausibly matter.

This is the cheapest honest experiment in the project:

  - Pure CPU. No model, no GPU, no retraining.
  - Perfectly paired. Every setting scores the SAME probability maps, so the
    run-to-run training variance that muddies E4's TTA number cannot touch it.
    Whatever Delta this reports is the post-processing stage and nothing else.

Run against any completed run:
    python -m src.sweep_postproc --name mit_b0_scaled
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.data.splits import make_split, resolve_data_root
from src.evaluate import _load_pair
from src.metrics import BinaryConfusion
from src.postprocess import postprocess_mask
from src.utils import get_logger, load_config, save_json

log = get_logger()

# close_kernel 1 and min_area 0 are the no-ops, present so the baseline sits in
# the same table as everything else rather than in a different run's json.
# The upper end is sized for 1024px: roads are ~12px wide here, so a kernel above
# ~11 starts merging genuinely separate roads instead of bridging occlusions, and
# an area floor of 2000 is ~a 12x170px road stub - already aggressive.
DEFAULT_KERNELS = [1, 5, 9, 11]
DEFAULT_AREAS = [0, 64, 500, 1000, 2000]


def sweep(pred_dir, data_root, val_ids, kernels, areas, threshold=0.5) -> list:
    pred_dir = Path(pred_dir)
    meters = {(k, a): BinaryConfusion() for k in kernels for a in areas}

    # One pass over the data, not one pass per setting: the images are 1024x1024
    # PNGs and decoding them 20 times over is the bulk of the cost. Within an
    # image the closing is also done once per kernel and reused across areas.
    for img_id in tqdm(val_ids, desc="sweeping", leave=False):
        prob, gt = _load_pair(pred_dir, data_root, img_id)
        binary = prob >= threshold
        for k in kernels:
            closed = postprocess_mask(binary, close_kernel=k, min_area=0)
            for a in areas:
                mask = postprocess_mask(closed, close_kernel=1, min_area=a) if a else closed
                meters[(k, a)].update(mask, gt)

    rows = [
        {"close_kernel": k, "min_area": a, **meters[(k, a)].as_dict()}
        for k in kernels for a in areas
    ]
    # Deltas are quoted against the no-op setting when it is in the grid; with a
    # custom grid that omits it, fall back to the weakest setting present so the
    # column still means "gain over the mildest clean-up here".
    baseline = min(rows, key=lambda r: (r["close_kernel"], r["min_area"]))
    for r in rows:
        r["delta_vs_none"] = round(r["iou"] - baseline["iou"], 4)
    return rows


def print_table(rows, kernels, areas) -> None:
    """IoU as a kernel x area grid - the shape of the surface matters more than
    the winner. A flat grid means the stage is genuinely dead at this resolution;
    a ridge means it is worth tuning properly."""
    lookup = {(r["close_kernel"], r["min_area"]): r["iou"] for r in rows}
    header = "kernel\\area".ljust(12) + "".join(str(a).rjust(9) for a in areas)
    print("\n" + header)
    print("-" * len(header))
    for k in kernels:
        print(str(k).ljust(12) + "".join(f"{lookup[(k, a)]:.4f}".rjust(9) for a in areas))
    best = max(rows, key=lambda r: r["iou"])
    print(
        f"\nbest: close_kernel={best['close_kernel']} min_area={best['min_area']} "
        f"-> IoU {best['iou']:.4f} ({best['delta_vs_none']:+.4f} vs no post-processing), "
        f"recall {best['recall']:.4f} precision {best['precision']:.4f}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Grid-search post-processing (CPU only).")
    parser.add_argument("--name", required=True, help="run name, e.g. mit_b0_scaled")
    parser.add_argument("--config", default=None, help="defaults to configs/<name>.yaml")
    parser.add_argument("--preds", default=None, help="defaults to outputs/preds/<name>")
    # 0.50 has come back as the tuned optimum on three separate converged runs
    # (E1, E2, E4), so sweeping the threshold again here would just burn CPU.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--kernels", type=int, nargs="+", default=DEFAULT_KERNELS)
    parser.add_argument("--areas", type=int, nargs="+", default=DEFAULT_AREAS)
    args = parser.parse_args()

    cfg = load_config(args.config or f"configs/{args.name}.yaml")
    cfg["data_root"] = resolve_data_root(cfg["data_root"])
    _, val_ids = make_split(cfg["data_root"], cfg["train_size"], cfg["val_size"], cfg["seed"])

    pred_dir = Path(args.preds or Path("outputs/preds") / args.name)
    if not pred_dir.exists():
        raise SystemExit(
            f"No predictions at {pred_dir}. Run the experiment first, or point "
            f"--preds at a previous Kaggle version's output attached as a dataset."
        )

    log.info(
        f"[{args.name}] sweeping {len(args.kernels)}x{len(args.areas)} settings "
        f"over {len(val_ids)} images at threshold {args.threshold}"
    )
    rows = sweep(pred_dir, cfg["data_root"], val_ids, args.kernels, args.areas, args.threshold)
    print_table(rows, args.kernels, args.areas)
    out = Path("outputs/results") / f"{args.name}_postproc_sweep.json"
    save_json({"name": args.name, "threshold": args.threshold, "rows": rows}, out)
    log.info(f"[{args.name}] sweep -> {out}")


if __name__ == "__main__":
    main()
