# CLAUDE.md

DeepGlobe road segmentation. Runs on a Kaggle T4; the repo is cloned fresh by a
notebook cell each run, so **nothing exists until it is pushed to GitHub**.

## Working style

Answer short. The user is time-poor and treats this as pipeline practice, not a
leaderboard chase — state the decision and the one fact that drove it, not the
survey of options. They are a visual thinker: reach for `src/visualize.py`
figures before another metrics table.

## Layout

```
main.py                  launcher; loops configs, calls run_experiment
configs/_base.yaml       shared defaults; per-model files override only what differs
src/train.py             run_experiment(cfg) = one experiment, end to end
src/engine.py            train_one_epoch, validate_crops, dump_predictions (tiled + TTA)
src/predict.py           frozen checkpoint -> preds + scores + figures; NO training
src/evaluate.py          score saved prob maps at one setting  (CPU, standalone)
src/sweep_postproc.py    score saved prob maps over a grid     (CPU, standalone)
src/visualize.py         error maps, threshold curve, IoU distribution (CPU, standalone)
docs/LOGBOOK.md          E0..E6 — every experiment, prediction recorded BEFORE the run
notebooks/kaggle_launcher.py   the single Kaggle cell, kept in sync with the above
```

Anything reading `outputs/preds/<name>/` needs no GPU and re-runs against any
past run. That is where cheap experiments live.

## Hard-won facts — do not re-derive

- **The headline metric is dataset IoU on full 1024px images** (TP/FP/FN
  accumulated over the val set). `val_iou(crop)` in the epoch log is the cheap
  checkpoint-selection metric and a *different quantity*. Anything from before
  commit `daa4257` (the "0.39") used the old metric — never quote a delta
  against it.
- **Threshold tuning is dead.** `best_thr` came back 0.50 on three separate
  converged runs, and 0.45 on a fourth for +0.0001. Dice-heavy loss
  (`bce_weight: 0.3`) leaves the model calibrated. Not a bug.
- **Post-processing is spent.** Grid-searched at the correct 1024px scale in E7:
  best cell (`close_kernel: 9`, `min_area: 500`) is worth **+0.0028** on a
  perfectly paired comparison, and the surface is flat. Adopted in `_base.yaml`
  because it's free. Do not re-tune it — it targets isolated blobs, and this
  model's error is parallel boundary fringe.
- **Data is spent.** 6,026 training images is everything after the 200-image val
  holdout.
- **Epochs are spent.** Plateaus by ~epoch 15; val_loss flat to 4 decimals.
- **`crop_size` is a context lever, not a resolution one.** `get_train_transform`
  uses `A.RandomCrop`, so training already happens at native 1024px pixel scale.
  256→512 buys context (the ~7-image semantic-confusion tail), not edge precision
  (the bulk of the error).
- **Training is not reproducible.** `set_seed` leaves `cudnn.benchmark = True`
  and does not request deterministic algorithms (`src/utils.py:26`). E2 and E4
  had identical data/seed/schedule and landed on different checkpoints. Spread on
  the selection metric is ~0.003 — quote that when a delta is small.
- **Inference-side changes must not go through `main.py`.** `run_experiment`
  always retrains (~31 min) and `cudnn.benchmark` gives you a different
  checkpoint, so the Δ you measure is your change *plus* training variance —
  that is what left E4 unable to separate TTA from overlap. Use
  `python -m src.predict --name X --tag Y ...` instead: one frozen checkpoint,
  ~3 min, genuinely paired. Only the *training* half of E6 is still open.

## Where the error actually is

Per-image IoU is unimodal around 0.57 with only 7/200 below 0.3 — there is no
distinct failing category to hunt. The bulk of the loss is **boundary placement**:
in the median image every road is found and the topology is right, and the error
map shows magenta/cyan running *parallel* along the same edges. On a ~12px road,
2px off per side costs ~30% IoU. Part of that fringe is label imprecision —
DeepGlobe masks are fixed-width buffers around centrelines — so it cannot all be
trained away.

## Conventions

- Predict the outcome in `docs/LOGBOOK.md` *before* running, then record the miss
  honestly. A logged wrong prediction is the point of the logbook.
- Change one lever per run, or say explicitly in the logbook that you didn't and
  why.
- Comments explain *why*, not what. Match the existing density.
- `main.py` and `run_experiment` swallow per-config failures on purpose — a bad
  config must not discard runs that already cost GPU minutes.

## Kaggle

Long runs go through **Save Version → Save & Run All (Commit)**, never the
interactive session — the interactive `/kaggle/working` is discarded when the tab
closes. Edit the launcher cell *in place*; Save & Run All executes every cell in a
fresh kernel, so a leftover cell re-runs and bills GPU twice. The leading `%cd`
matters: without it `rm -rf` destroys the directory the kernel is standing in.

Smoke test first, and pass `--no-tta` when you do — TTA runs over the full
200-image val set regardless of `train_size`, so without it a smoke test costs
~12 min instead of ~2.5 and stops being a smoke test.

```
!python main.py --only mit_b0_scaled --epochs 1 --train-size 64 --no-tta
```

## Score history

| entry | change | iou_final |
|---|---|---|
| E1 | MiT-B0, 2,000 imgs | 0.5494 |
| E2 | → 6,026 imgs | 0.5754 |
| E4 | → 8× TTA, tile_overlap 128 | **0.5971** |

Published D-LinkNet on DeepGlobe is ~0.63, trained on full images rather than
256px crops.
