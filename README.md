# Road Segmentation — DeepGlobe

Binary road extraction from satellite imagery, built as an end-to-end
segmentation pipeline rather than a leaderboard attempt. The medium-term goal is
3D CCTA (cardiac CT angiography) vessel segmentation — roads are deliberate
practice for the same problem class: thin, elongated, sparse, connectivity-critical
structures.

Dataset: [DeepGlobe Road Extraction](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
(1024×1024 satellite tiles, `<id>_sat.jpg` + `<id>_mask.png`).

## Running it

```bash
python main.py                        # default experiment set
python main.py --only mit_b0_scaled   # one experiment
python main.py --all                  # every config
python main.py --dry-run              # print the plan, touch no GPU

# smoke test before spending real GPU time
python main.py --only mit_b0_scaled --epochs 1 --train-size 64

# re-score saved predictions — no GPU, no retraining
python -m src.evaluate --name mit_b0_scaled --min-area 128 --close-kernel 7
```

On Kaggle the notebook is a single cell — see `notebooks/kaggle_launcher.py`.

## Layout

```
main.py               orchestrator: loops configs, runs them, prints a summary table
configs/_base.yaml    shared defaults; per-model configs only list what differs
src/train.py          run_experiment(cfg) — one config, end to end
src/engine.py         train/val loops + sliding-window full-image inference
src/evaluate.py       scores saved predictions (CPU only, re-runnable for free)
src/models/dlinknet.py  D-LinkNet, the 2018 challenge winner
src/postprocess.py    morphological closing + small-component removal
```

Adding a model = one branch in `src/models/factory.py` + one config file.
Nothing else changes.

## How results are measured

Read this before comparing any number here to a published one.

**Dataset IoU, not per-image IoU.** TP/FP/FN accumulate over the whole
validation set and divide once. Averaging per-image IoU instead makes
near-empty tiles count as much as dense ones and reads systematically lower.
The papers report dataset IoU.

**Full 1024×1024 images, not centre crops.** The models train on 256px crops but
are evaluated on whole images via overlapping sliding-window inference (25 tiles
per image, averaged through a count map so roads crossing a tile boundary have
no seam). Scoring on a 256px centre crop only ever sees ~6% of each image.

Tiling also sidesteps a practical constraint: MaxViT's windowed attention is
built for a fixed input size and will not simply accept a 1024px image. And it
is the pattern that transfers to the CCTA work — a 3D volume never fits in
memory either, so you tile, predict, and stitch.

**Full-image evaluation runs once, at the end.** Per-epoch validation uses cheap
centre crops purely to pick the best checkpoint; running full-image inference
every epoch would cost more than the training.

**Three stages are reported**, so each step's contribution is visible:

| stage | what it is |
|---|---|
| `raw@0.50` | default 0.5 threshold |
| `tuned@t` | best threshold found on validation (0.20–0.80 sweep) |
| `postproc@t` | + morphological closing and small-component removal |

Threshold tuning matters more than it sounds: 0.5 is rarely the optimum when the
positive class is 5–10% of pixels. Both stages cost zero GPU time.

> ⚠️ Numbers from before this refactor (e.g. mit_b0 ≈ 0.39) were per-image IoU on
> 256px centre crops. They are **not** comparable to anything reported here.

## Deliberate design choices

**The validation set is fixed across every run.** 200 images are held out first,
by seeded shuffle; `train_size` then takes a prefix of what remains, so a
2000-image run trains on a strict superset of a 400-image run and both are
scored on the same 200 images. Previously the split was 80/20 of a `subset_size`
slice, which meant changing `subset_size` also changed the validation set — and
"we improved" could just have been "we got an easier val set".

**Dice-weighted loss (`bce_weight: 0.3`).** BCE grades every pixel equally, so on
a tile that is ~92% background a model can score well while finding no roads at
all — the gradient from road pixels is diluted into noise. Dice grades region
*overlap* as a ratio, so predicting nothing scores 0 regardless of background
accuracy, and the gradient pushes toward recall. Thin structures make BCE's
dilution worse and Dice's framing more corrective.

**Best-checkpoint selection on IoU, not loss.** They disagree often enough on
imbalanced segmentation that selecting on loss costs points. The final epoch is
not reliably the best one.

**Mixed precision, cosine LR, AdamW.** ~2× throughput on a T4 for no measurable
accuracy cost, which is what buys the larger run. LR is stepped per batch, not
per epoch — at a few thousand total steps, per-epoch stepping is too coarse.

**Prediction and scoring are separate.** Training dumps probability maps to
`outputs/preds/<name>/` as 8-bit PNGs; `src/evaluate.py` scores them on CPU.
Post-processing and threshold experiments are then free and repeatable.

**Failures are isolated.** One config that OOMs does not discard the runs that
already succeeded — on a metered GPU that matters.

## Models

| config | arch | encoder | note |
|---|---|---|---|
| `resnet18` | U-Net | ResNet-18 | classic CNN baseline |
| `efficientnet_b0` | U-Net | EfficientNet-B0 | efficiency-oriented CNN |
| `mit_b0` | U-Net | MiT-B0 (SegFormer) | transformer; best at fixed small budget |
| `maxvit_tiny` | U-Net | MaxViT-Tiny | CNN/transformer hybrid |
| `mit_b0_scaled` | U-Net | MiT-B0 | the scale-up: 2000 images, 18 epochs |
| `dlinknet` | D-LinkNet | ResNet-34 | 2018 challenge winner, same budget as above |

**Why D-LinkNet.** Roads are thin, long and connected. A plain encoder
downsamples 32×, by which point a 3px-wide road is sub-pixel, and the bottleneck
receptive field may not span a road crossing the whole tile. D-LinkNet inserts a
cascade of dilated convolutions (rates 1/2/4/8) at the bottleneck, widening the
receptive field to cover the full tile without another downsample. It is the one
architecture here designed for this specific failure mode rather than being a
general-purpose backbone.

## Published reference points

Full training set, many more epochs than anything here:

| method | IoU |
|---|---|
| U-Net / LinkNet | ~0.63 |
| D-LinkNet (2018 winner) | 0.6466 val / 0.6342 test |
| NL-LinkNet (2019) | ~0.65 |
| recent DeepLabV3+ variants | 0.67–0.71 |

The gap from a small-budget run to these is overwhelmingly a data-and-epochs
gap, not an architecture or code gap. DeepGlobe's labels are also known to be
noisy and to under-label roads, which caps what any method achieves here.

## Compute notes

Built for Kaggle's free T4 (~30 h/week), so the priority is informative runs over
big ones. Per-epoch timing and a running "~Xm left" estimate print after every
epoch — if the projection is longer than you want, interrupt; the best
checkpoint so far is already on disk, and predictions can be re-scored later
without a GPU.
