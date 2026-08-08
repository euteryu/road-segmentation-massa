# src/visualize.py
"""Turn saved probability maps into figures a human can actually judge.

Metrics tell you *how much* is wrong; these tell you *how* it is wrong. For roads
that distinction matters more than usual: IoU is a pixel-count metric but roads
are a connectivity problem, so a model that predicts slightly-too-thick but
unbroken roads and one that predicts correct-width roads shattered into fragments
can score identically and be worth very different amounts downstream.

Like src/evaluate.py this is pure CPU and reads only what dump_predictions
already wrote, so it costs no GPU time and can be re-run against any past run:

    python -m src.visualize --name mit_b0_scaled
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

# Headless: a committed Kaggle kernel has no display, and importing pyplot with
# the default backend there either warns or dies depending on the image.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data.splits import make_split, resolve_data_root  # noqa: E402
from src.postprocess import postprocess_mask  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

log = get_logger()

# Error-map palette. Magenta/cyan is the pair that has to survive colour-vision
# deficiency, and it does comfortably (OKLab ΔE 14.5 under deuteranopia, 35.4 at
# normal vision; the working threshold is 8). The satellite base is darkened
# before compositing rather than picking a pale TP colour, because DeepGlobe
# roads are frequently pale grey asphalt - a white overlay on an undimmed tile
# camouflages against exactly the thing it is marking.
C_TP = (1.00, 1.00, 1.00)   # white   - correctly found road
C_FN = (0.90, 0.00, 0.43)   # magenta - road that was missed  (recall failure)
C_FP = (0.00, 0.65, 0.81)   # cyan    - road that isn't there (precision failure)
BASE_DIM = 0.55             # multiply the satellite tile by this before overlay

# Single-series charts: one hue, recessive everything else.
C_LINE = "#2a78d6"
C_GRID = "#d8d8d4"
C_INK = "#0b0b0b"
C_INK_SOFT = "#52514e"


def _style_axes(ax):
    """Recessive grid and axes so the data is the only thing that carries weight."""
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK_SOFT, labelsize=9, length=0)


def _load_triplet(pred_dir: Path, data_root: str, img_id: str):
    """-> (rgb uint8 HxWx3, prob float 0-1, gt bool). Raises if any is missing."""
    sat = cv2.imread(str(Path(data_root) / f"{img_id}_sat.jpg"), cv2.IMREAD_COLOR)
    if sat is None:
        raise FileNotFoundError(Path(data_root) / f"{img_id}_sat.jpg")
    prob = cv2.imread(str(pred_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
    if prob is None:
        raise FileNotFoundError(pred_dir / f"{img_id}.png")
    gt = cv2.imread(str(Path(data_root) / f"{img_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(Path(data_root) / f"{img_id}_mask.png")
    return cv2.cvtColor(sat, cv2.COLOR_BGR2RGB), prob.astype(np.float32) / 255.0, gt > 128


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    return inter / union if union > 0 else float("nan")


def _overlay(sat: np.ndarray, masks_colors) -> np.ndarray:
    """Composite solid colours onto a dimmed satellite tile."""
    out = sat.astype(np.float32) / 255.0 * BASE_DIM
    for mask, colour in masks_colors:
        out[mask] = colour
    return np.clip(out, 0, 1)


def per_image_iou_table(pred_dir, data_root, val_ids, threshold, close_kernel, min_area):
    """Per-image IoU at the final operating point, worst-first.

    A dataset-level mean of 0.55 hides which of two very different situations you
    are in: uniformly mediocre everywhere, or strong on highways and failing flat
    on unpaved tracks. Those have different fixes, so rank the images.
    """
    pred_dir = Path(pred_dir)
    rows = []
    for img_id in tqdm(val_ids, desc="per-image IoU", leave=False):
        prob = cv2.imread(str(pred_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(Path(data_root) / f"{img_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if prob is None or gt is None:
            continue
        gt = gt > 128
        pred = postprocess_mask(prob.astype(np.float32) / 255.0 >= threshold, close_kernel, min_area)
        rows.append({
            "id": img_id,
            "iou": round(_iou(pred.astype(bool), gt), 4),
            "gt_road_frac": round(float(np.count_nonzero(gt)) / gt.size, 5),
        })
    rows.sort(key=lambda r: (np.isnan(r["iou"]), r["iou"]))
    return rows


def panel_figure(pred_dir, data_root, img_id, threshold, close_kernel, min_area, out_path,
                 subtitle=""):
    """4-panel qualitative view of one validation image."""
    sat, prob, gt = _load_triplet(Path(pred_dir), data_root, img_id)
    pred = postprocess_mask(prob >= threshold, close_kernel, min_area).astype(bool)

    tp, fn, fp = pred & gt, ~pred & gt, pred & ~gt
    iou = _iou(pred, gt)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    panels = [
        (sat.astype(np.float32) / 255.0, "satellite"),
        (_overlay(sat, [(gt, C_TP)]), "ground truth"),
        (_overlay(sat, [(pred, C_TP)]), f"prediction @ {threshold:.2f}"),
        (_overlay(sat, [(tp, C_TP), (fn, C_FN), (fp, C_FP)]), "error map"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color=C_INK, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Label the error classes directly on the panel that uses them - identity is
    # never carried by colour alone.
    axes[3].legend(
        handles=[
            plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                       markerfacecolor=C_TP, markeredgecolor=C_INK_SOFT, label="hit"),
            plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                       markerfacecolor=C_FN, markeredgecolor="none", label="missed (FN)"),
            plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                       markerfacecolor=C_FP, markeredgecolor="none", label="false road (FP)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
        fontsize=9, labelcolor=C_INK_SOFT, handletextpad=0.4, columnspacing=1.4,
    )

    head = f"{img_id}   IoU {iou:.3f}"
    fig.suptitle(f"{head}   ·   {subtitle}" if subtitle else head,
                 fontsize=12, color=C_INK, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return iou


def threshold_curve_figure(curve: dict, best_thr: float, out_path, name=""):
    """Dataset IoU against decision threshold. Flat top => tuning cannot help."""
    items = sorted((float(t), v) for t, v in curve.items())
    thresholds = [t for t, _ in items]
    ious = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style_axes(ax)
    ax.plot(thresholds, ious, color=C_LINE, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    # Headroom first, so the peak label has somewhere to go that isn't the title.
    span = max(ious) - min(ious)
    pad = max(span * 0.18, 0.004)
    ax.set_ylim(min(ious) - pad * 0.4, max(ious) + pad)

    peak = max(range(len(ious)), key=lambda i: ious[i])
    # Keep the label inside the axes when the peak sits at either end of the sweep.
    mid = (thresholds[0] + thresholds[-1]) / 2
    ha = "center" if abs(thresholds[peak] - mid) < (thresholds[-1] - thresholds[0]) * 0.3 else (
        "left" if thresholds[peak] < mid else "right")
    ax.annotate(
        f"best {ious[peak]:.4f} @ {thresholds[peak]:.2f}",
        xy=(thresholds[peak], ious[peak]), xytext=(0, 10), textcoords="offset points",
        ha=ha, fontsize=10, color=C_INK,
    )
    ax.set_title(
        f"{name}  ·  IoU vs decision threshold"
        + (f"   (range {span:.4f} — flat, so tuning buys nothing)" if span < 0.01 else ""),
        fontsize=11, color=C_INK, loc="left", pad=10,
    )
    ax.set_xlabel("threshold", fontsize=10, color=C_INK_SOFT)
    ax.set_ylabel("dataset IoU", fontsize=10, color=C_INK_SOFT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def iou_distribution_figure(rows, out_path, name=""):
    """Spread of per-image IoU. Where the mass sits is the actionable part."""
    vals = [r["iou"] for r in rows if not np.isnan(r["iou"])]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _style_axes(ax)
    ax.hist(vals, bins=20, range=(0, 1), color=C_LINE, edgecolor="white", linewidth=1.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts, not fractions of an image

    mean_v, med_v = float(np.mean(vals)), float(np.median(vals))
    top = ax.get_ylim()[1]
    # Staggered heights and outward-facing alignment: mean and median often land
    # within a few hundredths of each other and would otherwise overprint.
    for value, label, style, dy in ((mean_v, "mean", "-", -12), (med_v, "median", "--", -28)):
        ax.axvline(value, color=C_INK, linewidth=1.4, linestyle=style, zorder=4)
        right_half = value > 0.5
        ax.annotate(f"{label} {value:.3f}", xy=(value, top),
                    xytext=(-5 if right_half else 5, dy), textcoords="offset points",
                    fontsize=9, color=C_INK, ha="right" if right_half else "left")
    ax.set_title(f"{name}  ·  per-image IoU across {len(vals)} validation images",
                 fontsize=11, color=C_INK, loc="left", pad=10)
    ax.set_xlabel("per-image IoU", fontsize=10, color=C_INK_SOFT)
    ax.set_ylabel("images", fontsize=10, color=C_INK_SOFT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def make_figures(name, pred_dir, data_root, val_ids, report, out_dir,
                 close_kernel=5, min_area=64, n_examples=3):
    """Everything, from saved predictions. Returns the per-image IoU table.

    Examples are picked at the worst / median / best of the per-image IoU ranking
    rather than chosen by hand, so the figures show the honest range of behaviour
    instead of a flattering sample.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = report["best_threshold"]

    rows = per_image_iou_table(pred_dir, data_root, val_ids, threshold, close_kernel, min_area)
    if not rows:
        log.warning(f"[{name}] no predictions found to visualise")
        return rows

    threshold_curve_figure(report["threshold_curve"], threshold,
                           out_dir / "threshold_curve.png", name=name)
    iou_distribution_figure(rows, out_dir / "iou_distribution.png", name=name)

    picks = [
        (label, rows[idx]["id"])
        for label, idx in (("worst", 0), ("median", len(rows) // 2), ("best", len(rows) - 1))
    ][:n_examples]

    seen = set()
    for label, img_id in picks:
        if img_id in seen:  # tiny val sets can land the same image in two slots
            continue
        seen.add(img_id)
        try:
            panel_figure(pred_dir, data_root, img_id, threshold, close_kernel, min_area,
                         out_dir / f"example_{label}_{img_id}.png", subtitle=f"{label} of {len(rows)}")
        except FileNotFoundError as exc:
            log.warning(f"[{name}] skipped {label} example: {exc}")

    log.info(f"[{name}] figures -> {out_dir}/  ({len(seen)} examples + 2 charts)")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build figures from saved predictions (CPU only).")
    parser.add_argument("--name", required=True, help="run name, e.g. mit_b0_scaled")
    parser.add_argument("--config", default=None, help="defaults to configs/<name>.yaml")
    parser.add_argument("--close-kernel", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config or f"configs/{args.name}.yaml")
    cfg["data_root"] = resolve_data_root(cfg["data_root"])
    _, val_ids = make_split(cfg["data_root"], cfg["train_size"], cfg["val_size"], cfg["seed"])

    # Reuse the eval report if it exists so the figures sit at the same operating
    # point as the reported numbers; otherwise recompute it.
    import json

    eval_path = Path("outputs/results") / f"{args.name}.json"
    if eval_path.exists():
        report = json.loads(eval_path.read_text())["full_image_eval"]
    else:
        from src.evaluate import evaluate_predictions
        report = evaluate_predictions(
            Path("outputs/preds") / args.name, cfg["data_root"], val_ids,
            close_kernel=cfg.get("close_kernel", 5), min_area=cfg.get("min_area", 64),
        )

    make_figures(
        args.name, Path("outputs/preds") / args.name, cfg["data_root"], val_ids, report,
        Path("outputs/figures") / args.name,
        close_kernel=args.close_kernel if args.close_kernel is not None else cfg.get("close_kernel", 5),
        min_area=args.min_area if args.min_area is not None else cfg.get("min_area", 64),
    )


if __name__ == "__main__":
    main()
