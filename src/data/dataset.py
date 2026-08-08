# src/data/dataset.py
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class RoadDataset(Dataset):
    """DeepGlobe road tiles: `<id>_sat.jpg` paired with `<id>_mask.png`.

    Masks ship as 8-bit greyscale with JPEG-ish intermediate values around the
    edges, so they are binarised at 128 per the dataset spec.
    """

    def __init__(self, ids, root_dir, transform=None, return_id: bool = False):
        self.ids = list(ids)
        self.root_dir = root_dir
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.root_dir, f"{img_id}_sat.jpg")
        mask_path = os.path.join(self.root_dir, f"{img_id}_mask.png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if self.return_id:
            return image, mask, img_id
        return image, mask
