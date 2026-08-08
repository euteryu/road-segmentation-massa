# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics - the encoders are ImageNet-pretrained, so inputs must be
# normalised the same way they were during pretraining.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transform(crop_size: int):
    """Random crops act as both augmentation and the mechanism that lets us train
    on 1024px imagery at 256px cost. Roads have no canonical orientation, so the
    full dihedral group (flips + 90-degree rotations) is safe here - unlike, say,
    text or medical images with a fixed anatomical orientation."""
    return A.Compose([
        A.RandomCrop(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transform(crop_size: int):
    """Cheap per-epoch validation: one deterministic centre crop per image.
    Used only for checkpoint selection, never for the headline number."""
    return A.Compose([
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_full_image_transform():
    """Final evaluation: no cropping, full 1024x1024 image. Scoring on a centre
    crop only ever sees ~6% of each image, which is not the protocol the
    published DeepGlobe numbers use."""
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
