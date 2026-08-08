# src/train.py
"""One experiment = one config. `run_experiment(cfg)` is the unit of work;
main.py just loops over configs and calls it.

Usage (single run):
    python -m src.train --config configs/mit_b0_scaled.yaml
"""
import argparse
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RoadDataset
from src.data.splits import make_split, resolve_data_root
from src.data.transforms import get_full_image_transform, get_train_transform, get_val_transform
from src.engine import dump_predictions, train_one_epoch, validate_crops
from src.evaluate import evaluate_predictions
from src.losses import build_criterion
from src.models.factory import build_model
from src.utils import (
    count_params, env_summary, format_duration, get_device, get_logger,
    load_config, output_dirs, save_json, set_seed,
)

log = get_logger()


def build_loaders(cfg):
    # Mutates cfg so the resolved path is also what evaluation reads ground-truth
    # masks from later, and so it lands in the saved result json.
    cfg["data_root"] = resolve_data_root(cfg["data_root"])
    train_ids, val_ids = make_split(
        cfg["data_root"], cfg["train_size"], cfg["val_size"], cfg["seed"]
    )
    crop = cfg["crop_size"]
    workers = cfg.get("num_workers", 2)
    common = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)

    train_loader = DataLoader(
        RoadDataset(train_ids, cfg["data_root"], get_train_transform(crop)),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True, **common,
    )
    # Cheap per-epoch validation used for checkpoint selection only.
    val_loader = DataLoader(
        RoadDataset(val_ids, cfg["data_root"], get_val_transform(crop)),
        batch_size=cfg["batch_size"], shuffle=False, **common,
    )
    # Full-resolution loader for the final, comparable-to-literature evaluation.
    full_loader = DataLoader(
        RoadDataset(val_ids, cfg["data_root"], get_full_image_transform(), return_id=True),
        batch_size=1, shuffle=False, num_workers=workers, pin_memory=True,
    )
    return train_loader, val_loader, full_loader, train_ids, val_ids


def run_experiment(cfg: dict) -> dict:
    name = cfg["name"]
    set_seed(cfg["seed"])
    device = get_device()
    dirs = output_dirs(name)
    ckpt_path = dirs["checkpoints"] / f"{name}.pth"

    train_loader, val_loader, full_loader, train_ids, val_ids = build_loaders(cfg)
    model = build_model(cfg).to(device)
    n_params = count_params(model)

    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4)
    )
    # Cosine decay stepped per batch, not per epoch: with runs this short (a few
    # thousand steps) per-epoch stepping is too coarse to shape the schedule.
    total_steps = max(1, cfg["epochs"] * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    log.info(
        f"[{name}] arch={cfg.get('arch', 'unet')} encoder={cfg['encoder_name']} "
        f"params={n_params / 1e6:.1f}M | train={len(train_ids)} val={len(val_ids)} "
        f"| {cfg['epochs']} epochs @ {cfg['crop_size']}px bs={cfg['batch_size']} amp={amp}"
    )

    history = []
    best_iou = -1.0
    start = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, cfg["epochs"], scheduler,
        )
        val_loss, val_iou = validate_crops(model, val_loader, criterion, device, amp=amp)
        epoch_time = time.time() - epoch_start
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_iou_crop": round(val_iou, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "epoch_time_sec": round(epoch_time, 1),
        })
        # Select on validation IoU, not on loss: loss and IoU disagree often
        # enough on imbalanced segmentation that picking on loss costs points.
        marker = ""
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1}, ckpt_path)
            marker = "  <- best"
        remaining = (cfg["epochs"] - epoch - 1) * epoch_time
        log.info(
            f"[{name}] epoch {epoch + 1}/{cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_iou(crop)={val_iou:.4f} "
            f"[{format_duration(epoch_time)}/epoch, ~{format_duration(remaining)} left]{marker}"
        )

    train_time = time.time() - start
    log.info(f"[{name}] training done in {format_duration(train_time)}; best crop IoU {best_iou:.4f}")

    # Reload the best checkpoint before the real evaluation - the final epoch is
    # not reliably the best one on a run this short.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    infer_start = time.time()
    dump_predictions(
        model, full_loader, device, dirs["preds"],
        tile=cfg["crop_size"], overlap=cfg.get("tile_overlap", 64), amp=amp,
        tta=cfg.get("tta", False),
    )
    infer_time = time.time() - infer_start

    report = evaluate_predictions(
        dirs["preds"], cfg["data_root"], val_ids,
        close_kernel=cfg.get("close_kernel", 5),
        min_area=cfg.get("min_area", 64),
    )

    # Qualitative output. Wrapped because a plotting bug must never destroy the
    # numbers from a run that already cost GPU minutes - same reasoning as the
    # per-config try/except in main.py.
    per_image = []
    try:
        from src.visualize import make_figures

        per_image = make_figures(
            name, dirs["preds"], cfg["data_root"], val_ids, report, dirs["figures"],
            close_kernel=cfg.get("close_kernel", 5),
            min_area=cfg.get("min_area", 64),
        )
    except Exception:
        log.warning(f"[{name}] figure generation failed (metrics unaffected):\n{traceback.format_exc()}")

    result = {
        "name": name,
        "arch": cfg.get("arch", "unet"),
        "encoder": cfg["encoder_name"],
        "params_millions": round(n_params / 1e6, 2),
        "train_images": len(train_ids),
        "epochs": cfg["epochs"],
        "crop_size": cfg["crop_size"],
        "bce_weight": cfg.get("bce_weight", 0.3),
        "tile_overlap": cfg.get("tile_overlap", 64),
        "tta": cfg.get("tta", False),
        "train_time_sec": round(train_time, 1),
        "infer_time_sec": round(infer_time, 1),
        "best_val_iou_crop": round(best_iou, 4),
        "full_image_eval": report,
        "per_image_iou": per_image,
        "history": history,
        "env": env_summary(),
    }
    save_json(result, dirs["results"] / f"{name}.json")

    stages = report["stages"]
    log.info(f"[{name}] FULL-IMAGE dataset IoU  " + "  ".join(
        f"{k}={v['iou']:.4f}" for k, v in stages.items()
    ))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(load_config(args.config))


if __name__ == "__main__":
    main()
