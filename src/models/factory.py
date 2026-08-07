# src/models/factory.py
import segmentation_models_pytorch as smp

def build_model(encoder_name, encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )