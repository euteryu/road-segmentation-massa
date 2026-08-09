# src/predict.py
"""Inference + scoring from a frozen checkpoint. No training.

Why this exists (logbook E6): `run_experiment` always trains, so every
inference-side question - TTA on/off, tile overlap, threshold, post-processing -
cost a full ~31-minute retrain AND dragged training variance into the answer.
`cudnn.benchmark = True` means two runs with identical configs land on different
checkpoints, which is exactly what left E4's +0.0217 unable to separate TTA from
noise.

This makes those comparisons **paired**: one checkpoint, loaded once, scored under
whatever inference settings you name. A ~3-minute run whose delta is attributable
to the thing you changed and nothing else.

    # re-score the frozen checkpoint at its config's settings
    python -m src.predict --name mit_b0_scaled

    # the ablation E4 could not do: same weights, TTA off vs on
    python -m src.predict --name mit_b0_scaled --tag no_tta  --no-tta
    python -m src.predict --name mit_b0_scaled --tag ov64    --tile-overlap 64

Each --tag writes its own outputs/preds/<name>__<tag>/ so variants never clobber
each other and every one stays re-scorable on CPU afterwards.

On Kaggle the checkpoint comes from a previous version's output attached as a
dataset; --checkpoint is autodetected under /kaggle/input if it is not local.
"""
import argparse
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RoadDataset
from src.data.splits import KAGGLE_INPUT, make_split, resolve_data_root
from src.data.transforms import get_full_image_transform
from src.engine import dump_predictions
from src.evaluate import evaluate_predictions
from src.models.factory import build_model
from src.utils import (
    env_summary, format_duration, get_device, get_logger, load_config, output_dirs, save_json,
)

log = get_logger()


def find_checkpoint(name: str) -> Path:
    """Local checkpoint if present, else search /kaggle/input for one.

    Same reasoning as `find_data_root`: on Kaggle the mount path of an attached
    dataset depends on how it was attached, and losing a session to a path typo
    is not worth it.
    """
    local = Path("outputs/checkpoints") / f"{name}.pth"
    if local.exists():
        return local

    if KAGGLE_INPUT.is_dir():
        hits = sorted(KAGGLE_INPUT.glob(f"**/checkpoints/{name}.pth"))
        if hits:
            log.info(f"[{name}] checkpoint not local; using attached dataset {hits[0]}")
            return hits[0]

    attached = (
        ", ".join(sorted(p.name for p in KAGGLE_INPUT.iterdir()))
        if KAGGLE_INPUT.is_dir() else "(no /kaggle/input on this machine)"
    )
    raise SystemExit(
        f"No checkpoint at {local} and none found under /kaggle/input.\n"
        f"Attached datasets: {attached}\n"
        f"Attach a previous version's output, or pass --checkpoint explicitly."
    )


def load_frozen_model(ckpt_path: Path, cfg: dict, device):
    """Build the architecture from the checkpoint's own config, not the yaml.

    The yaml on disk can drift after the weights were written (that is the whole
    point of freezing a checkpoint), and a state_dict built against the wrong
    encoder either raises a wall of shape errors or - worse - loads partially.
    `encoder_weights=None` skips the ImageNet download we are about to overwrite.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = dict(ckpt.get("cfg") or cfg)
    train_cfg["encoder_weights"] = None
    train_cfg["checkpoint_epoch"] = ckpt.get("epoch")

    model = build_model(train_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(
        f"[{cfg['name']}] loaded {ckpt_path} "
        f"(arch={train_cfg.get('arch', 'unet')} encoder={train_cfg['encoder_name']} "
        f"epoch={train_cfg['checkpoint_epoch']} trained on {train_cfg.get('train_size', '?')} imgs)"
    )
    return model, train_cfg


def predict_and_score(cfg: dict, ckpt_path: Path, tag: str = "") -> dict:
    name = cfg["name"]
    run_id = f"{name}__{tag}" if tag else name
    device = get_device()
    dirs = output_dirs(run_id)

    cfg["data_root"] = resolve_data_root(cfg["data_root"])
    _, val_ids = make_split(cfg["data_root"], cfg["train_size"], cfg["val_size"], cfg["seed"])

    model, train_cfg = load_frozen_model(ckpt_path, cfg, device)
    amp = cfg.get("amp", True) and device.type == "cuda"
    # Tile size follows the checkpoint, not the yaml: some encoders only accept
    # the resolution they were built for, and a model trained on 256px crops
    # evaluated with 512px tiles is a different experiment than the one intended.
    tile = train_cfg.get("crop_size", cfg["crop_size"])

    full_loader = DataLoader(
        RoadDataset(val_ids, cfg["data_root"], get_full_image_transform(), return_id=True),
        batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 2), pin_memory=True,
    )

    log.info(
        f"[{run_id}] inference on {len(val_ids)} images | tile={tile} "
        f"overlap={cfg.get('tile_overlap', 64)} tta={cfg.get('tta', False)} amp={amp}"
    )

    infer_start = time.time()
    dump_predictions(
        model, full_loader, device, dirs["preds"],
        tile=tile, overlap=cfg.get("tile_overlap", 64), amp=amp,
        tta=cfg.get("tta", False),
    )
    infer_time = time.time() - infer_start
    log.info(f"[{run_id}] inference done in {format_duration(infer_time)}")

    report = evaluate_predictions(
        dirs["preds"], cfg["data_root"], val_ids,
        close_kernel=cfg.get("close_kernel", 5),
        min_area=cfg.get("min_area", 64),
    )

    # Same try/except as run_experiment: a plotting bug must not destroy numbers
    # that already cost GPU minutes.
    per_image = []
    try:
        from src.visualize import make_figures

        per_image = make_figures(
            run_id, dirs["preds"], cfg["data_root"], val_ids, report, dirs["figures"],
            close_kernel=cfg.get("close_kernel", 5),
            min_area=cfg.get("min_area", 64),
        )
    except Exception:
        log.warning(f"[{run_id}] figure generation failed (metrics unaffected):\n{traceback.format_exc()}")

    result = {
        "name": run_id,
        "source_run": name,
        "tag": tag,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": train_cfg.get("checkpoint_epoch"),
        "encoder": train_cfg["encoder_name"],
        "arch": train_cfg.get("arch", "unet"),
        # The inference settings are the experimental variables here - record them
        # at the top level so a diff between two tags is a diff between these.
        "inference": {
            "tile": tile,
            "tile_overlap": cfg.get("tile_overlap", 64),
            "tta": cfg.get("tta", False),
            "close_kernel": cfg.get("close_kernel", 5),
            "min_area": cfg.get("min_area", 64),
        },
        "infer_time_sec": round(infer_time, 1),
        "full_image_eval": report,
        "per_image_iou": per_image,
        "env": env_summary(),
    }
    save_json(result, dirs["results"] / f"{run_id}_predict.json")

    log.info(f"[{run_id}] FULL-IMAGE dataset IoU  " + "  ".join(
        f"{k}={v['iou']:.4f}" for k, v in report["stages"].items()
    ))
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True, help="run name, e.g. mit_b0_scaled")
    parser.add_argument("--config", default=None, help="defaults to configs/<name>.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="defaults to outputs/checkpoints/<name>.pth, then /kaggle/input")
    parser.add_argument("--tag", default="",
                        help="suffix for outputs, so variants of one checkpoint don't collide")
    # Inference-side overrides. These are the levers this entry point exists for;
    # None means "leave the config value alone" so an unflagged run reproduces the
    # original run's operating point exactly.
    parser.add_argument("--tile-overlap", type=int, default=None)
    parser.add_argument("--tta", dest="tta", action="store_true", default=None)
    parser.add_argument("--no-tta", dest="tta", action="store_false", default=None)
    parser.add_argument("--close-kernel", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config or f"configs/{args.name}.yaml")
    for key, value in (
        ("tile_overlap", args.tile_overlap), ("tta", args.tta),
        ("close_kernel", args.close_kernel), ("min_area", args.min_area),
    ):
        if value is not None:
            cfg[key] = value

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint(args.name)
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}")

    predict_and_score(cfg, ckpt_path, tag=args.tag)


if __name__ == "__main__":
    main()
