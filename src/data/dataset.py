# src/data/dataset.py
import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class RoadDataset(Dataset):
    def __init__(self, ids, root_dir, transform=None):
        self.ids = ids
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.root_dir, f"{img_id}_sat.jpg")
        mask_path = os.path.join(self.root_dir, f"{img_id}_mask.png")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.float32)  # binarize per dataset spec

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        return image, mask