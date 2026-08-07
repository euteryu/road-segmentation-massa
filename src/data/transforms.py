# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(crop_size):
    return A.Compose([
        A.RandomCrop(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_transform(crop_size):
    return A.Compose([
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(),
        ToTensorV2(),
    ])