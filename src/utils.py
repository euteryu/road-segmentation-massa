# src/utils.py
"""Small shared helpers: seeding, config loading, checkpoints, logging."""
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "_base.yaml"


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Note cudnn.benchmark stays on: we accept a little
    nondeterminism in exchange for the speed, which is the usual trade for
    experiments that are compared on held-out metrics rather than bit-exactness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path) -> dict:
    """Load a model config layered on top of configs/_base.yaml.

    Keeps per-model files down to just what actually differs, so a shared change
    (epochs, loss weight, data root) is a one-line edit instead of five.
    """
    cfg = {}
    if BASE_CONFIG.exists():
        with open(BASE_CONFIG) as f:
            cfg.update(yaml.safe_load(f) or {})
    with open(path) as f:
        cfg.update(yaml.safe_load(f) or {})
    cfg["config_path"] = str(path)
    return cfg


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def output_dirs(name: str, root: str = "outputs") -> dict:
    """Every run writes under outputs/<kind>/<name>/ so runs never clobber each other."""
    paths = {
        "results": Path(root) / "results",
        "checkpoints": Path(root) / "checkpoints",
        "preds": Path(root) / "preds" / name,
        "logs": Path(root) / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_logger(name: str = "roadseg") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def env_summary() -> dict:
    return {
        "device": gpu_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_count": os.cpu_count(),
    }
