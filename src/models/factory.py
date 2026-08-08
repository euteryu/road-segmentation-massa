# src/models/factory.py
"""Single place that turns a config dict into a model.

Everything downstream (training loop, evaluation, checkpointing) is
architecture-agnostic, so adding a model means adding a branch here plus a
config file - nothing else changes.
"""
import segmentation_models_pytorch as smp

from src.models.dlinknet import DLinkNet


def build_model(cfg: dict):
    arch = cfg.get("arch", "unet").lower()
    encoder_name = cfg["encoder_name"]
    weights = cfg.get("encoder_weights", "imagenet")

    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
        )
    if arch == "dlinknet":
        return DLinkNet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            classes=1,
        )
    raise ValueError(f"Unknown arch '{arch}' (expected one of: unet, dlinknet)")
