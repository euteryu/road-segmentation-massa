# src/compare.py
"""Two runs of the same checkpoint, side by side, on the images that separate them.

The ablation table says TTA is worth +0.016 dataset IoU; this shows what those
points physically are. Each figure is one validation image with run A's error map
next to run B's. Images are auto-picked at the largest gain / median / largest
loss of the per-image delta (B - A) - same policy as src/visualize.py, so the
sample shows the honest range instead of a flattering one.

Pure CPU, reads only saved prob maps and the results jsons:

    python -m src.compare --a mit_b0_scaled__no_tta --b mit_b0_scaled__base \
        --label-a "no TTA" --label-b "8x TTA"

Each run is rendered at its own reported operating point (threshold and postproc
constants from its json), so every panel matches the numbers already quoted for
that run rather than some third setting neither was scored at.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.splits import make_split, resolve_data_root  # noqa: E402
from src.postprocess import postprocess_mask  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402
from src.visualize import (  # noqa: E402
    C_FN, C_FP, C_INK, C_INK_SOFT, C_TP, _iou, _load_triplet, _overlay,
    per_image_iou_table,
)

log = get_logger()


def _load_run(name: str, cfg: dict, val_ids):
    """-> ({img_id: iou}, threshold, close_kernel, min_area) for one run.

    Prefers the run's own json so the figure sits at the operating point its
    numbers were reported at. src.predict jsons store the constants under
    "inference"; run_experiment jsons don't store them at all, so those fall
    back to the config - which is where run_experiment read them from anyway.
    """
    data = None
    for candidate in (f"{name}_predict.json", f"{name}.json"):
        path = Path("outputs/results") / candidate
        if path.exists():
            data = json.loads(path.read_text())
            break
    if data is None:
        raise SystemExit(f"no outputs/results/ json for {name} - run src.predict first")

    inf = data.get("inference", {})
    threshold = data["full_image_eval"]["best_threshold"]
    close_kernel = inf.get("close_kernel", cfg.get("close_kernel", 5))
    min_area = inf.get("min_area", cfg.get("min_area", 64))

    rows = data.get("per_image_iou") or per_image_iou_table(
        Path("outputs/preds") / name, cfg["data_root"], val_ids,
        threshold, close_kernel, min_area,
    )
    return {r["id"]: r["iou"] for r in rows}, threshold, close_kernel, min_area


def _error_masks(pred_dir, data_root, img_id, threshold, close_kernel, min_area):
    sat, prob, gt = _load_triplet(Path(pred_dir), data_root, img_id)
    pred = postprocess_mask(prob >= threshold, close_kernel, min_area).astype(bool)
    return sat, (pred & gt, ~pred & gt, pred & ~gt), _iou(pred, gt)


def compare_figure(run_a, run_b, cfg, img_id, label_a, label_b, out_path, subtitle=""):
    """One image: satellite, error map under A, error map under B."""
    ids_a, thr_a, ck_a, ma_a = run_a
    ids_b, thr_b, ck_b, ma_b = run_b
    dir_a = Path("outputs/preds") / label_a["name"]
    dir_b = Path("outputs/preds") / label_b["name"]

    sat, (tp_a, fn_a, fp_a), iou_a = _error_masks(dir_a, cfg["data_root"], img_id, thr_a, ck_a, ma_a)
    _, (tp_b, fn_b, fp_b), iou_b = _error_masks(dir_b, cfg["data_root"], img_id, thr_b, ck_b, ma_b)

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.3))
    panels = [
        (sat.astype(np.float32) / 255.0, "satellite"),
        (_overlay(sat, [(tp_a, C_TP), (fn_a, C_FN), (fp_a, C_FP)]),
         f"{label_a['text']}   IoU {iou_a:.3f}"),
        (_overlay(sat, [(tp_b, C_TP), (fn_b, C_FN), (fp_b, C_FP)]),
         f"{label_b['text']}   IoU {iou_b:.3f}"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color=C_INK, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[2].legend(
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

    head = f"{img_id}   Δ {iou_b - iou_a:+.3f}"
    fig.suptitle(f"{head}   ·   {subtitle}" if subtitle else head,
                 fontsize=12, color=C_INK, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", required=True, help="baseline run, e.g. mit_b0_scaled__no_tta")
    parser.add_argument("--b", required=True, help="comparison run, e.g. mit_b0_scaled__base")
    parser.add_argument("--label-a", default=None, help="panel title for --a (defaults to its tag)")
    parser.add_argument("--label-b", default=None, help="panel title for --b (defaults to its tag)")
    parser.add_argument("--config", default=None,
                        help="defaults to configs/<source>.yaml, source = name before '__'")
    args = parser.parse_args()

    source = args.a.split("__")[0]
    cfg = load_config(args.config or f"configs/{source}.yaml")
    cfg["data_root"] = resolve_data_root(cfg["data_root"])
    _, val_ids = make_split(cfg["data_root"], cfg["train_size"], cfg["val_size"], cfg["seed"])

    run_a = _load_run(args.a, cfg, val_ids)
    run_b = _load_run(args.b, cfg, val_ids)

    shared = sorted(set(run_a[0]) & set(run_b[0]))
    if not shared:
        raise SystemExit("no validation images shared between the two runs")
    deltas = sorted(shared, key=lambda i: run_b[0][i] - run_a[0][i])

    # Largest loss / median / largest gain of the per-image delta. The extremes
    # show what the change is capable of in both directions; the median shows
    # what it typically does - which for a +0.016 dataset delta is the honest
    # headline image, not the best case.
    picks = [
        ("loss", deltas[0]),
        ("median", deltas[len(deltas) // 2]),
        ("gain", deltas[-1]),
    ]

    out_dir = Path("outputs/figures") / f"compare_{args.a}_vs_{args.b}"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_a = {"name": args.a, "text": args.label_a or (args.a.split("__")[-1])}
    label_b = {"name": args.b, "text": args.label_b or (args.b.split("__")[-1])}

    seen = set()
    for kind, img_id in picks:
        if img_id in seen:
            continue
        seen.add(img_id)
        try:
            compare_figure(run_a, run_b, cfg, img_id, label_a, label_b,
                           out_dir / f"delta_{kind}_{img_id}.png",
                           subtitle=f"{kind} of {len(shared)} Δ({label_b['text']} − {label_a['text']})")
        except FileNotFoundError as exc:
            log.warning(f"skipped {kind} example: {exc}")

    mean_d = float(np.mean([run_b[0][i] - run_a[0][i] for i in shared]))
    log.info(f"figures -> {out_dir}/  ({len(seen)} images, mean per-image Δ {mean_d:+.4f})")


if __name__ == "__main__":
    main()
