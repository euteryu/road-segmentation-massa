# src/data/splits.py
"""Deterministic train/val splitting.

Design point that matters: the validation set is carved out FIRST and is
identical for every run, regardless of `train_size`. Earlier the split was taken
as 80/20 of a `subset_size` slice, which meant a 400-image run and a 2000-image
run were scored on different validation images - so their IoUs were not
comparable, and "we got better" could just have been "we got an easier val set".

Training images are then taken as a prefix of the remaining pool, so a larger
`train_size` is a strict superset of a smaller one. That makes the data-scaling
comparison clean.

Note we split the official `train/` folder only: DeepGlobe's `valid/` and `test/`
folders ship without public masks, so they cannot be scored offline.
"""
import os
import random
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")


def _looks_like_train_dir(path: Path) -> bool:
    """A usable training directory has BOTH satellite images and masks.

    Requiring masks is what distinguishes DeepGlobe's `train/` from its `valid/`
    and `test/` folders - those ship satellite images with no public labels, so
    matching on `*_sat.jpg` alone would happily pick an unusable directory.
    """
    return (
        next(path.glob("*_sat.jpg"), None) is not None
        and next(path.glob("*_mask.png"), None) is not None
    )


def find_data_root(max_depth: int = 4):
    """Breadth-first search of /kaggle/input for a labelled training directory.

    Kaggle's mount path depends on how the dataset was attached (slug, owner
    prefix, nesting), and it is not worth losing a session to a path typo -
    especially since a wrong path only surfaces after the environment has
    already been set up.
    """
    if not KAGGLE_INPUT.is_dir():
        return None

    level = [KAGGLE_INPUT]
    for _ in range(max_depth):
        children = []
        for directory in level:
            try:
                entries = sorted(p for p in directory.iterdir() if p.is_dir())
            except OSError:
                continue
            for child in entries:
                if _looks_like_train_dir(child):
                    return child
                children.append(child)
        if not children:
            break
        level = children
    return None


def resolve_data_root(configured: str) -> str:
    """Use the configured path if it exists, otherwise autodetect and report."""
    if configured and os.path.isdir(configured):
        return configured

    found = find_data_root()
    if found is None:
        available = (
            ", ".join(sorted(p.name for p in KAGGLE_INPUT.iterdir()))
            if KAGGLE_INPUT.is_dir() else "(no /kaggle/input on this machine)"
        )
        raise FileNotFoundError(
            f"data_root does not exist: {configured}\n"
            f"Autodetection found no directory containing both *_sat.jpg and "
            f"*_mask.png.\nAttached datasets: {available}\n"
            f"Attach the DeepGlobe dataset, or set data_root in configs/_base.yaml."
        )

    print(f"[splits] data_root '{configured}' not found; using autodetected '{found}'")
    return str(found)


def list_image_ids(root_dir: str) -> list:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"data_root does not exist: {root_dir}")
    ids = sorted(
        f[: -len("_sat.jpg")] for f in os.listdir(root_dir) if f.endswith("_sat.jpg")
    )
    if not ids:
        raise FileNotFoundError(f"No *_sat.jpg files found under {root_dir}")
    return ids


def make_split(root_dir: str, train_size: int, val_size: int = 200, seed: int = 42):
    """-> (train_ids, val_ids). `train_size <= 0` means 'use everything left'."""
    ids = list_image_ids(root_dir)
    random.Random(seed).shuffle(ids)

    if val_size >= len(ids):
        raise ValueError(f"val_size={val_size} >= dataset size {len(ids)}")

    val_ids = ids[:val_size]
    pool = ids[val_size:]
    train_ids = pool if train_size <= 0 else pool[:train_size]

    if len(train_ids) < train_size:
        print(
            f"[splits] warning: requested train_size={train_size} but only "
            f"{len(train_ids)} images remain after holding out {val_size} for val."
        )
    return train_ids, val_ids
