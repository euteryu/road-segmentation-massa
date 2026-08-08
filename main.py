#!/usr/bin/env python
"""Launcher. The notebook's job is to call this; this file's job is everything else.

    python main.py                          # run the default experiment set
    python main.py --only mit_b0_scaled     # run one experiment
    python main.py --only resnet18 mit_b0   # run a named subset
    python main.py --all                    # every config in configs/
    python main.py --dry-run                # print the plan, touch no GPU

Keeping the pipeline in a script rather than notebook cells is not just tidiness:
it is diffable, reviewable, re-runnable from a clean state, and it is how a real
training job gets launched (a cloud worker runs a script, it does not click
through cells). The notebook stays a three-line launcher.
"""
import argparse
import sys
import traceback
from pathlib import Path

from src.utils import CONFIG_DIR, env_summary, format_duration, get_logger, load_config, save_json

log = get_logger()

# The default run set. Ordered cheapest-first so a failure surfaces early rather
# than 40 minutes in.
DEFAULT_EXPERIMENTS = [
    "mit_b0_scaled",
    "dlinknet",
]


def discover_configs(names):
    configs = []
    for name in names:
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if not p.stem.startswith("_"))
            raise SystemExit(f"No config '{name}'. Available: {', '.join(available)}")
        configs.append(load_config(path))
    return configs


def summarise(results, path="outputs/results/summary.json"):
    """One table is worth more than N json files when comparing runs."""
    rows = []
    for r in results:
        stages = r["full_image_eval"]["stages"]
        final_key = [k for k in stages if k.startswith("postproc")][0]
        rows.append({
            "name": r["name"],
            "encoder": r["encoder"],
            "params_M": r["params_millions"],
            "train_imgs": r["train_images"],
            "epochs": r["epochs"],
            "train_min": round(r["train_time_sec"] / 60, 1),
            "iou_raw": stages["raw@0.50"]["iou"],
            "iou_tuned": stages[[k for k in stages if k.startswith("tuned")][0]]["iou"],
            "iou_final": stages[final_key]["iou"],
            "recall_final": stages[final_key]["recall"],
            "best_thr": r["full_image_eval"]["best_threshold"],
        })
    rows.sort(key=lambda r: r["iou_final"], reverse=True)
    save_json({"env": env_summary(), "rows": rows}, path)

    headers = list(rows[0].keys())
    widths = [max(len(h), *(len(str(r[h])) for r in rows)) for h in headers]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))
    print(f"\nfull results -> outputs/results/   summary -> {path}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME", help="config names to run")
    parser.add_argument("--all", action="store_true", help="run every config in configs/")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--epochs", type=int, help="override epochs (smoke tests)")
    parser.add_argument("--train-size", type=int, help="override train_size (smoke tests)")
    # TTA is 8x inference over the FULL val set regardless of train_size, so a
    # smoke test without this flag costs ~12 minutes and stops being a smoke test.
    parser.add_argument("--no-tta", action="store_true", help="disable test-time augmentation (smoke tests)")
    args = parser.parse_args()

    if args.all:
        names = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if not p.stem.startswith("_"))
    else:
        names = args.only or DEFAULT_EXPERIMENTS

    configs = discover_configs(names)
    for cfg in configs:
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        if args.train_size is not None:
            cfg["train_size"] = args.train_size
        if args.no_tta:
            cfg["tta"] = False

    log.info(f"env: {env_summary()}")
    log.info(f"plan: {' -> '.join(c['name'] for c in configs)}")
    for cfg in configs:
        log.info(
            f"  {cfg['name']:<18} {cfg.get('arch', 'unet'):<9} {cfg['encoder_name']:<24} "
            f"train={cfg['train_size']:<5} epochs={cfg['epochs']}"
        )
    if args.dry_run:
        return 0

    # Imported late so --dry-run and --help stay instant and work without a GPU
    # or the heavy deps installed.
    from src.train import run_experiment

    results, failures = [], []
    for cfg in configs:
        try:
            results.append(run_experiment(cfg))
        except Exception:
            # One bad config (OOM, unsupported encoder resolution) must not throw
            # away the runs that already succeeded - on a metered GPU that is
            # expensive. Record it, keep going, report at the end.
            log.error(f"[{cfg['name']}] FAILED:\n{traceback.format_exc()}")
            failures.append(cfg["name"])

    if results:
        summarise(results)
    if failures:
        log.error(f"failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
