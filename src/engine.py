# src/engine.py
"""Training and inference loops. Deliberately architecture-agnostic."""
import numpy as np
import torch
from tqdm.auto import tqdm

from src.metrics import BinaryConfusion


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, epochs, scheduler=None):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, masks)

        # Mixed precision: fp16 forward/backward with an fp32 master copy of the
        # weights. Roughly 2x throughput and ~half the activation memory on a T4,
        # for no measurable accuracy cost on this task. The scaler exists because
        # fp16 gradients underflow to zero without it.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running / len(loader.dataset)


@torch.no_grad()
def validate_crops(model, loader, criterion, device, threshold=0.5, amp=True):
    """Fast per-epoch validation on centre crops - used only to pick the best
    checkpoint. The headline metric comes from full-image evaluation at the end,
    which is far too slow to run every epoch."""
    model.eval()
    meter = BinaryConfusion()
    running = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        running += loss.item() * images.size(0)
        meter.update(torch.sigmoid(logits.float()) >= threshold, masks > 0.5)

    return running / len(loader.dataset), meter.iou


@torch.no_grad()
def predict_tiled(model, image, device, tile=256, overlap=64, tile_batch=16, amp=True):
    """Predict a full-resolution image by sliding a window over it.

    Needed because the models are trained on 256px crops but the images are
    1024px, and some encoders (MaxViT in particular) only accept the resolution
    they were built for. Tiles overlap and are averaged through a count map, so
    there are no visible seams where a road crosses a tile boundary.

    This is the same pattern used for whole-slide pathology and for 3D medical
    volumes - the input never fits in memory, so you tile, predict, and stitch.

    Args:
        image: (3, H, W) normalised tensor.
    Returns:
        (H, W) float32 probability map on CPU.
    """
    _, height, width = image.shape
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile ({tile})")

    def starts(size):
        if size <= tile:
            return [0]
        pos = list(range(0, size - tile + 1, stride))
        if pos[-1] != size - tile:
            pos.append(size - tile)
        return pos

    ys, xs = starts(height), starts(width)
    coords = [(y, x) for y in ys for x in xs]

    prob = torch.zeros(height, width, dtype=torch.float32, device=device)
    count = torch.zeros(height, width, dtype=torch.float32, device=device)
    image = image.to(device, non_blocking=True)

    for i in range(0, len(coords), tile_batch):
        chunk = coords[i : i + tile_batch]
        batch = torch.stack([image[:, y : y + tile, x : x + tile] for y, x in chunk])
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(batch)
        probs = torch.sigmoid(logits.float())[:, 0]
        for (y, x), p in zip(chunk, probs):
            prob[y : y + tile, x : x + tile] += p
            count[y : y + tile, x : x + tile] += 1

    return (prob / count.clamp(min=1)).cpu()


def _d4(x, k: int, mirror: bool):
    """Rotate by k*90 degrees, then optionally mirror. Operates on the last two dims."""
    y = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if mirror else y


def _d4_inverse(x, k: int, mirror: bool):
    """Undo _d4 - mirror first, then rotate back, because the forward order was
    rotate-then-mirror and these do not commute."""
    y = torch.flip(x, dims=(-1,)) if mirror else x
    return torch.rot90(y, -k, dims=(-2, -1))


@torch.no_grad()
def predict_tiled_tta(model, image, device, tile=256, overlap=64, amp=True, tta=True):
    """predict_tiled averaged over the 8 symmetries of the square.

    These are exactly the symmetries get_train_transform samples from, so every
    view is in-distribution for the model - averaging over a transform it never
    trained on would add noise rather than remove it.

    The reason this is worth 8x inference on this task specifically: error here is
    dominated by boundary placement on thin structures, and boundary noise is
    largely uncorrelated between orientations, so it averages down. Detection
    errors (a road missed in every orientation) are not helped.
    """
    if not tta:
        return predict_tiled(model, image, device, tile=tile, overlap=overlap, amp=amp)

    _, height, width = image.shape
    if height != width:
        raise ValueError(f"TTA rotations assume a square image, got {height}x{width}")

    acc = None
    for k in range(4):
        for mirror in (False, True):
            prob = predict_tiled(
                model, _d4(image, k, mirror), device, tile=tile, overlap=overlap, amp=amp
            )
            back = _d4_inverse(prob, k, mirror)
            acc = back if acc is None else acc + back
    return acc / 8.0


@torch.no_grad()
def dump_predictions(model, loader, device, out_dir, tile=256, overlap=64, amp=True, tta=False):
    """Run full-image inference once and write probability maps to disk as 8-bit
    PNGs.

    Separating "predict" from "score" is the point: threshold sweeps and
    post-processing experiments then cost zero GPU time and can be re-run as many
    times as you like on the saved outputs. Quantising to uint8 loses ~1/255 of
    probability resolution, which is far below the granularity of any threshold
    worth choosing.
    """
    import cv2

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    desc = "full-image inference" + (" (8x TTA)" if tta else "")
    for images, _masks, ids in tqdm(loader, desc=desc, leave=False):
        for image, img_id in zip(images, ids):
            prob = predict_tiled_tta(
                model, image, device, tile=tile, overlap=overlap, amp=amp, tta=tta
            )
            arr = (prob.numpy() * 255).round().astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{img_id}.png"), arr)
            written.append(img_id)

    return written
