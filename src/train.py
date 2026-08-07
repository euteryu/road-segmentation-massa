# src/train.py
import os
import json
import time
import yaml
import argparse
import random
import torch
from torch.utils.data import DataLoader

from data.dataset import RoadDataset
from data.transforms import get_train_transform, get_val_transform
from models.factory import build_model
from losses import BCEDiceLoss
from metrics import iou_score, f1_score

def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = cfg["data_root"]
    all_ids = sorted({
        f.split("_sat.jpg")[0] for f in os.listdir(root_dir) if f.endswith("_sat.jpg")
    })
    random.shuffle(all_ids)
    subset_ids = all_ids[: cfg["subset_size"]]
    split = int(0.8 * len(subset_ids))
    train_ids, val_ids = subset_ids[:split], subset_ids[split:]

    train_ds = RoadDataset(train_ids, root_dir, get_train_transform(cfg["crop_size"]))
    val_ds = RoadDataset(val_ids, root_dir, get_val_transform(cfg["crop_size"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = build_model(cfg["encoder_name"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    start = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start

    model.eval()
    ious, f1s = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            ious.append(iou_score(logits, masks))
            f1s.append(f1_score(logits, masks))

    result = {
        "name": cfg["name"],
        "encoder": cfg["encoder_name"],
        "params": n_params,
        "train_time_sec": round(train_time, 1),
        "val_iou": round(sum(ious) / len(ious), 4),
        "val_f1": round(sum(f1s) / len(f1s), 4),
    }

    os.makedirs("outputs/results", exist_ok=True)
    with open(f"outputs/results/{cfg['name']}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)