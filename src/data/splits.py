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


def list_image_ids(root_dir: str) -> list:
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"data_root does not exist: {root_dir}\n"
            "On Kaggle, check the dataset is attached and the path matches "
            "/kaggle/input/<dataset-slug>/train"
        )
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
