# Logbook — DeepGlobe road segmentation

Chronological and **append-only**. Entries are never rewritten, even when a later
entry proves one wrong — the wrong turns are the point. When something is
superseded, the newer entry says so and links back.

This is the *why*. The *what* is already machine-readable in
`outputs/results/<name>.json` (every run records its full config, per-epoch
history, staged metrics, and per-image IoU table). Don't duplicate numbers here
that the JSON already holds; record the reasoning the JSON can't.

Entry format: **Question → Change → Result → Read → Decision.** "Read" is the
interpretation, and it is the part worth writing carefully.

---

## Standing facts

Things true across all entries. If one changes, that gets its own entry.

**Data.** DeepGlobe Road Extraction, `train/` folder only: 6,226 images, 1024×1024,
~0.5 m/pixel, rural Thailand / India / Indonesia. The official `valid/` and `test/`
folders ship without public masks, so they cannot be scored offline.

**Split** (`src/data/splits.py`). Seeded shuffle (seed 42, fixed), then:

| split | n | notes |
|---|---|---|
| val | 200 | carved out **first**, identical in every run |
| train | up to 6,026 | prefix of the remainder |
| test | **0** | ← known gap, see E5 |

Val-first means runs at different `train_size` are scored on the same images and
so are directly comparable. Train-as-prefix means a larger train set is a strict
superset of a smaller one, so data-scaling comparisons are clean.

**Metric.** Headline is **dataset IoU** — TP/FP/FN accumulated over the whole val
set, divided once — on full 1024px images via sliding-window inference. Reported
at three stages: `raw@0.50`, `tuned@t` (best threshold on val), `postproc@t`
(morphological closing + small-component removal).

`val_iou(crop)` in the per-epoch logs is a *different, cheaper* metric — per-image
IoU on a 256px centre crop — used only for checkpoint selection. Never quote it as
a result.

**Noise floor.** 200 val images carries roughly ±0.01 on the absolute number.
Cross-run differences below **~0.02 are not clearly real**. Paired comparisons on
the same images and checkpoint (e.g. TTA on/off) are far more sensitive, because
shared image difficulty cancels.

**Hardware / timing.** Kaggle Tesla T4, batch 16 @ 256px, AMP on. Measured
throughput: **~34 s/epoch at 2,000 images**, **~96 s/epoch at 6,026**. Budget from
these, not from guesses. Anything over ~10 min goes through Save Version → Save &
Run All, never the interactive session (which discards `/kaggle/working`).

---

## E0 — Pre-history: the 0.39 that doesn't count (before 2026-08-08)

**Result.** An early `mit_b0` run reported ≈0.39 IoU.

**Read.** Not comparable to anything current, for two independent reasons: it was
measured as a *mean of per-image IoU on 256px centre crops* (~6% of each image),
and the validation split itself was different. It was never comparable to the
published DeepGlobe numbers either, since those use dataset IoU on full images.

**Decision.** Discard. Superseded at commit `daa4257`. **Never report a delta
against 0.39** — any apparent jump is mostly the metric definition changing.

---

## E1 — Architecture at fixed budget: MiT-B0 vs D-LinkNet (2026-08-08)

**Question.** At an equal data and epoch budget, does the task-specific
architecture (D-LinkNet, the DeepGlobe-era standard) beat a small generic
transformer encoder (MiT-B0 in a U-Net)?

**Change.** Both at 2,000 train images, 18 epochs, 256px crops, batch 16.

**Result.**

| model | encoder | params | train | iou_raw | iou_tuned | iou_final | recall | best_thr |
|---|---|---|---|---|---|---|---|---|
| mit_b0_scaled | MiT-B0 | 5.55 M | 11.0 min | 0.5488 | 0.5488 | **0.5494** | 0.7251 | 0.50 |
| dlinknet | ResNet-34 | 31.1 M | 9.8 min | 0.5387 | 0.5387 | 0.5396 | 0.7099 | 0.50 |

**Read.** The 0.0098 gap is **below the noise floor — this is a tie on accuracy,
not a win.** The real result is parameter efficiency: MiT-B0 matches D-LinkNet
with **5.6× fewer parameters**. D-LinkNet is slightly faster per epoch despite
being 6× larger, which is expected — ResNet convolutions are better optimised on a
T4 than MiT attention.

Two secondary findings, both initially surprising:

1. **Threshold tuning did nothing.** `best_thr` came back 0.50 — the default — so
   `iou_raw == iou_tuned` exactly. But the 1-epoch/64-image smoke test *did* pick
   0.70–0.75 and gained ~+0.01. That contrast is the explanation, not a bug: an
   undertrained model is under-confident, so raising the threshold helps; a
   converged model on dice-heavy loss (`bce_weight: 0.3`) is already calibrated at
   0.5. The sweep works, it simply has nothing to find.
2. **Post-processing did almost nothing** (+0.0006). `min_area: 64` and
   `close_kernel: 5` are sized for 256px crops but applied to 1024px images —
   64 px is 0.006% of the frame. Never scaled these; still unscaled as of E4.

**Decision.** Drop D-LinkNet — it lost at equal budget and costs 6× the parameters.
Carry MiT-B0 forward. Stop expecting anything from threshold tuning.

---

## E2 — Data scaling: 2,000 → 6,026 images (2026-08-08)

**Question.** Both models plateaued by epoch ~14 (val_loss flat to 4 decimal
places over the last four epochs). Is the ceiling epochs, or data?

**Change.** MiT-B0 only, `train_size: 0` (= all 6,026 after the val holdout),
18 epochs, everything else identical to E1.

**Result.** 29.9 min. `iou_raw` 0.5749 → `iou_final` **0.5754**, recall 0.7418,
`best_thr` 0.50 again. Per-image mean 0.5587, median 0.5698.

**Read.** **+0.026 for 3× the data** — clears the noise floor, so it's real, but
clearly diminishing. And it plateaued *again* (val_loss 0.4664 / 0.4666 / 0.4663 /
0.4663 over the last four epochs; crop IoU peaked at epoch 12). So neither epochs
nor data is the active constraint any more, and **the data lever is now spent** —
6,026 is the entire labelled pool.

Threshold tuning was a no-op for a second time at a different data scale. That
hypothesis is now confirmed, not merely suspected.

**Decision.** Stop scaling data. Find out what the remaining error actually *is*
before spending another 30 GPU-minutes guessing — which requires being able to
look at predictions, which the pipeline could not yet do. → E3.

---

## E3 — Building visual QA, and what it showed (2026-08-09)

**Question.** IoU says *how much* is wrong. What is wrong?

**Rationale.** IoU is a pixel-counting metric, but road extraction is a
*connectivity* problem. Two models at identical IoU — one predicting
slightly-thick but unbroken roads, one predicting correct-width roads shattered at
every tree shadow — score the same and are worth very different amounts
downstream. This is why the literature pairs IoU with connectivity metrics such as
APLS. A recall of 0.7418 says 26% of road pixels are missed but says nothing about
*whether they are scattered noise or systematic breaks*.

**Change.** Added `src/visualize.py` — 4-panel qualitative views (satellite /
ground truth / prediction / error map), the threshold curve, and the per-image IoU
distribution. Pure CPU, reads only the probability maps `dump_predictions` already
writes, so it costs no GPU time and re-runs against any past run. Wired into
`run_experiment` inside a try/except so a plotting bug can't destroy the numbers
from a paid run. Examples are auto-picked at worst/median/best per-image IoU,
never hand-chosen.

**Result — three findings, in order of importance.**

1. **The bulk of the loss is boundary placement, not detection.** In the median
   image (IoU 0.570) every road is found and the topology is correct; the error
   map shows magenta (FN) and cyan (FP) running *parallel along the same road
   edges*. That signature means correct position, wrong width. On a ~12px-wide
   road, being 2px off per side mismatches ~30% of it. Thin structures are
   punished brutally by an overlap metric.
2. **The distribution is unimodal** — one hump centred at 0.57, only 7 of 200
   below 0.3, none below 0.05. So there is **no distinct failing category**; this
   kills the "unpaved tracks fail as a class" hypothesis before it cost anything.
3. **The tail is semantic confusion.** The worst image (IoU 0.100) traces a dry
   riverbed as road — a pale sinuous channel that genuinely resembles a dirt
   track — while missing the real tracks. Only ~7 images; not the main story.

**Read + a correction.** I had been calling `crop_size` a *resolution* lever. It
isn't: `get_train_transform` uses `A.RandomCrop` (`src/data/transforms.py:17`), so
training already happens at native 1024px pixel scale. `crop_size` buys
**context**, not resolution. Context would help finding #3 (rivers branch
differently and don't connect to settlements) but does little for finding #1,
which is where the score actually leaks. So the obvious-looking 512px run targets
the tail, not the bulk — much lower value than it first appeared.

An honest ceiling: DeepGlobe masks are **fixed-width buffers around centrelines**,
but real road width varies. Part of that parallel magenta/cyan fringe is *label*
imprecision that no amount of training can remove.

**Decision.** Attack boundary precision, not context, and do it at inference cost
only. → E4.

---

## E4 — Test-time augmentation + denser tiling (2026-08-09) — *in flight*

**Question.** How much of the boundary fuzziness is model *noise* (removable by
averaging) rather than label imprecision (not removable at all)?

**Change.** Both inference-only; no retraining logic touched.

- `tta: true` — average predictions over the 8 symmetries of the square
  (`predict_tiled_tta` in `src/engine.py`). These are exactly the symmetries
  `get_train_transform` samples, so every view is in-distribution; averaging over
  a transform the model never trained on would add noise instead of removing it.
- `tile_overlap: 64 → 128` — 25 → 49 tiles per image, so each pixel gets more
  votes at tile seams.

Mechanism: boundary noise is largely uncorrelated between orientations, so it
averages down; the true road is identical in all 8 views and survives. Detection
errors (a road missed in every orientation) are *not* helped — which is fine,
because E3 showed detection isn't the problem.

Added `--no-tta` for smoke tests: TTA runs over the full 200-image val set
regardless of `train_size`, so without the flag a smoke test costs ~12 min instead
of ~2.5 and stops being a smoke test.

Verified before running: D4 forward/inverse roundtrip exact for all 8 combinations
on asymmetric fixtures, the 8 views genuinely distinct, an identity model averages
back to its input. A wrong inverse would not raise — it would silently misalign
the averages and quietly cost IoU.

**Expected** (recorded before running). +0.005–0.015 over 0.5754. ~38 min.

**Result.** 35 min total (31.6 train, 3m16s TTA inference over 200 images — I had
budgeted ~6.5 min for inference and overestimated).

| | E2 (no TTA, overlap 64) | E4 (8× TTA, overlap 128) | Δ |
|---|---|---|---|
| iou_raw | 0.5749 | 0.5968 | +0.0219 |
| iou_final | 0.5754 | **0.5971** | **+0.0217** |
| recall | 0.7418 | 0.7638 | +0.0220 |
| best_thr | 0.50 | 0.50 | — |

**Read.** **The prediction was too conservative — +0.0217, above the stated
+0.005–0.015 band.** Recording that as a miss: I under-estimated how much of the
boundary error was recoverable noise.

The mechanism looks right. Recall rose by almost exactly as much as IoU (+0.0220),
which is what averaging 8 views should do — genuine road pixels sitting marginally
below 0.5 in any single view get pushed over the line once the views agree, while
uncorrelated edge noise cancels. Threshold came back 0.50 for a **third** time.

**Confound — the comparison is not perfectly paired.** `set_seed` leaves
`cudnn.benchmark = True` and does not request deterministic algorithms
(`src/utils.py:26`), so training is not bit-reproducible. E2 and E4 used identical
data, seed and schedule but landed on **different checkpoints**: best crop IoU
0.5583 @ epoch 12 (E2) vs 0.5550 @ epoch 16 (E4). So +0.0217 = TTA + overlap +
training variance, not TTA alone.

Two reasons it's still credible: run-to-run spread on the selection metric is
~0.003, roughly an order of magnitude below the gain; and E4's checkpoint was the
*worse* of the two by that metric yet scored higher on the full-image metric, so
variance plausibly worked against TTA rather than for it. (Caveat: crop IoU and
full-image dataset IoU are different quantities, so that argument is suggestive,
not conclusive.)

**Which of TTA vs overlap contributed what is unmeasured** — they were changed
together. Deliberate: they attack the same mechanism and the goal was one cheap
run, not an ablation.

**Decision.** Keep both. Boundary error was substantially model noise, not purely
label imprecision — so the E3 hypothesis holds and the pre-registered "if the gain
is ~0" branch does not fire. Also: **separate inference from training** so future
inference-side ablations reuse one frozen checkpoint and are genuinely paired. That
architectural gap is what made this result fuzzier than it needed to be. → E6.

---

## E6 — Open: training is not reproducible, and inference can't run standalone

**Problem, two halves of one issue.**

1. `cudnn.benchmark = True` with no deterministic algorithms means identical
   configs produce different checkpoints (demonstrated in E4). The existing
   docstring calls this a deliberate speed trade — fine as a default, but it means
   **no two runs are strictly comparable**, which quietly weakens every Δ in this
   logbook.
2. `run_experiment` always trains. There is no way to load a checkpoint and re-run
   only inference + scoring, so any inference-side change (TTA, overlap,
   post-processing constants, threshold) forces a full 31-minute retrain **and**
   drags training variance into the measurement.

**Fix.** An inference-only entry point (`python -m src.predict --name X
--checkpoint ...`) that loads a frozen checkpoint and runs
dump_predictions → evaluate → figures. Turns a 31-minute confounded comparison into
a ~3-minute paired one. On Kaggle it needs the prior version's output attached as a
dataset.

Determinism itself: leave `benchmark = True`, but quantify the variance once by
running the same config 3× and reporting the spread. Cheaper and more honest than
pretending seeds make it exact.

**Not yet done.** Sequenced after the E5 test-set carve.

---

## E5 — Open: no held-out test set

**Problem.** There is no test split. The same 200 val images currently do three
jobs: select the best epoch, tune the decision threshold, and produce the headline
number. Choosing on the data you report is self-grading, so **0.5754 is mildly
optimistic.**

Mitigating facts, for honesty in both directions: threshold tuning selected 0.50,
the default, so nothing actually adapted; and checkpoint selection uses the *crop*
metric while the headline is the *full-image* metric, so they aren't the same
quantity. The optimism is small — but it is the first thing a reviewer would
challenge.

**Why it happened.** DeepGlobe's official `valid/` and `test/` folders have no
public masks, so the only labelled data available is the `train/` folder. That
explains the constraint; it does not excuse not carving a test set from what's
there.

**Planned fix.** Hold out a further 200 images as a true test set — never touched
until every decision is frozen, then scored once. Costs 3% of training data and
upgrades the claim from "0.5754 on the set I tuned against" to "0.5754 val,
0.57x test".

**Deliberately deferred** until E4 lands: changing the split mid-flight would break
comparability with E1–E4.

**Separately — cross-dataset generalization.** Full pre-registered method in
[`docs/protocols/cross-dataset-eval.md`](protocols/cross-dataset-eval.md), written
before running anything so the result can't be retrofitted. Summary: **CHN6-CUG is
the primary target, not Massachusetts** — it is also 0.5 m/pixel, so no resampling
is needed and the ground-sample-distance confound disappears entirely. Three arms:
zero-shot, fine-tuned, and from-scratch-on-target. The last two are what make it an
experiment rather than a limitations paragraph, because B-vs-C answers whether the
DeepGlobe training is a reusable asset or whether labelling target data would have
been cheaper.

Sequenced **after** E4 and after the E5 test-set carve — an in-distribution test
set has to exist before an out-of-distribution number means anything.

**Naming note.** This repo is called `road-segmentation-massa` after the
Massachusetts Roads Dataset, which it has never used — the project has been
DeepGlobe throughout. Kept as-is to avoid breaking the Kaggle launcher's clone URL;
recorded here so the name doesn't mislead later.

---

## E7 — Post-processing at the correct scale: the lever is dead (2026-08-09)

**Question.** E1 blamed the flat post-processing result on a units mistake —
`min_area: 64` and `close_kernel: 5` were sized for 256px crops but applied to
1024px images, where 64 px is 0.006% of the frame. Was the stage genuinely
worthless, or just mis-tuned by ~16× in area?

**Change.** `src/sweep_postproc.py` — a 4×5 grid (kernels 1/5/9/11, areas
0/64/500/1000/2000) scored against saved probability maps. Pure CPU, and
**perfectly paired**: every cell scores the *same* prob maps from the *same*
checkpoint, so training variance — the thing that muddied E4 — cannot touch this
Δ. The ±0.02 cross-run noise floor does not apply here; a paired Δ of a few
thousandths is real, it is just small.

The predictions came from a fresh 2,000-image / 18-epoch run (11.6 min) rather
than the 6,026-image E4 checkpoint, because E6 is still open and there is no way
to re-run inference alone.

**Implied prediction** (from E1, not separately pre-registered): scaling the
constants to 1024px should recover something meaningful — E1's +0.0006 was
supposed to be a units bug, not a verdict.

**Result.** Best cell `close_kernel=9, min_area=500` → **0.5660**, vs 0.5632 with
no post-processing at all. **+0.0028.** Recall 0.7185, precision 0.7273.

The shape matters more than the winner, and the shape is **flat**: every cell
from area 0 to 1000 sits within 0.003 of every other, across all four kernels.
The only real movement in the grid is *downward* at area 2000 (−0.007), where the
floor starts eating genuine road stubs.

**Read.** **Miss — the units bug was not hiding a gain.** Correctly scaled, the
stage is worth +0.0028, an order of magnitude below what TTA bought for the same
zero training cost. The lever moves from "untested at correct scale" to **spent**.

Why it's flat, in hindsight: E3 showed the error is boundary placement running
*parallel* along real roads, not scattered blobs. Small-component removal deletes
isolated islands and closing bridges gaps — neither operation moves an edge that
is 2px too wide. Post-processing is aimed at a failure mode this model does not
have. The +0.0028 is presumably the ~7-image semantic-confusion tail, where a
traced riverbed is a large connected blob that no area floor would remove either.

Two side observations from the same run:

1. **A second, unpaired TTA confirmation.** This run is E1's exact config
   (2,000 images, 18 epochs) plus TTA and overlap 128: 0.5494 → 0.5632 raw,
   **+0.0138** at a data scale where E4's +0.0217 was measured at 6,026. Two
   different data scales, same direction, comparable magnitude. Different
   checkpoints again, so still not a clean ablation — but TTA is now the
   best-supported result in this logbook.
2. **Threshold tuning, fourth run.** `best_thr` came back **0.45** this time,
   not 0.50 — and bought +0.0001. That is the lever wobbling inside its own
   noise, not waking up. Still dead.

**Decision.** Adopt `close_kernel: 9, min_area: 500` in `configs/_base.yaml` —
it is free and non-negative — and stop touching this stage. Do **not** re-run
inference at 6,026 just to re-confirm; a +0.0028 paired gain does not justify
31 GPU-minutes. Every remaining cheap-inference idea is now exhausted, which
promotes E6 (standalone inference) from "nice architecture" to the actual
blocker: the next real levers (boundary-weighted loss, crop size) all require
retraining, and none of them can be measured cleanly until inference is paired
and a test set exists. → E6, then E5.

---

## E8 — Standalone inference: closing half of E6 (2026-08-09)

**Question.** Not a science question — a measurement-apparatus one. E4's +0.0217
could not separate TTA from overlap from training variance, and E7 could only be
trusted *because* it was paired. Every remaining lever is worth less than the
noise that the apparatus currently injects. Fix the apparatus.

**Change.** `src/predict.py` — loads a frozen checkpoint and runs
dump_predictions → evaluate → figures, with no training loop anywhere in the
path. Inference settings are CLI overrides (`--tta/--no-tta`, `--tile-overlap`,
`--close-kernel`, `--min-area`), and `--tag` gives each variant its own
`outputs/preds/<name>__<tag>/` so two settings of one checkpoint can be held side
by side instead of overwriting each other.

Three decisions worth recording:

- **The architecture is rebuilt from the checkpoint's own stored `cfg`, not from
  `configs/<name>.yaml`.** The yaml drifts after weights are written — that is
  the point of freezing a checkpoint — and a `state_dict` loaded against a
  drifted encoder either throws a wall of shape errors or, worse, loads
  partially and silently scores a half-random model.
- `encoder_weights=None` when loading, so it doesn't download ImageNet weights it
  is about to overwrite.
- Checkpoints are autodetected under `/kaggle/input/**/checkpoints/<name>.pth`,
  same reasoning as `find_data_root`: on Kaggle the mount path depends on how the
  dataset was attached, and a path typo costs a session.

This closes **half** of E6. The other half — `cudnn.benchmark = True`, so no two
*training* runs are identical — is untouched and deliberately so: pairing the
inference side removes variance from the comparisons that were actually being
confounded, and it costs 3 minutes instead of 31.

**Prediction, recorded before running.** The first use is the ablation E4 could
not do: one checkpoint, TTA off vs on, overlap 64 vs 128.

- TTA alone: **+0.010 to +0.018**. E4 measured +0.0217 at 6,026 images and E7
  +0.0138 at 2,000, both with the overlap change and a different checkpoint
  folded in. Paired, I expect it to land inside that pair rather than above it.
- Overlap 64 → 128 alone: **+0.002 to +0.005** — real but small, since it only
  adds votes near tile seams, and seams are a minority of the pixels where E3
  located the error.
- The two should roughly sum to E4's +0.0217. **If they sum to noticeably less,
  the residual was training variance** — which would retroactively weaken E4 and
  is exactly the thing worth knowing.

**Cost.** Two 3-minute inference runs against one existing checkpoint. No
training. Kaggle needs the prior version's output attached as a dataset.

**Result** (run 2026-08-09, same session as one fresh training run — no prior
checkpoint was attached, so the frozen weights came from an 11.7-min 2,000-image
run in the same kernel: best crop IoU 0.5218 @ epoch 12). Three inference passes
over those weights:

| tag | TTA | overlap | infer time | raw@0.50 | tuned | postproc |
|---|---|---|---|---|---|---|
| base | on | 128 | 3m00s | 0.5582 | 0.5585 @ 0.45 | 0.5616 |
| no_tta | off | 128 | 26s | 0.5422 | 0.5422 @ 0.50 | 0.5455 |
| ov64 | off | 64 | 17s | 0.5413 | 0.5413 @ 0.50 | 0.5447 |

Paired, at the fixed 0.50 threshold: **TTA +0.0160, overlap 128 +0.0009.** Both
deltas are stable to ±0.0003 across all three scoring stages.

Apparatus check passed: the `base` tag (no overrides) reproduced the training
run's own eval to all four decimals at every stage, so the checkpoint round-trip
through `src/predict.py` is exact — the thing E8 was built to guarantee.

**Read.** One hit, one miss.

- TTA, predicted +0.010 to +0.018 → **+0.0160, hit**, upper half of the band.
- Overlap, predicted +0.002 to +0.005 → **+0.0009, miss, below the band.**
  Indistinguishable from zero and an order of magnitude below its cost (1.5×
  inference time). Overlap 128 was adopted inside E4's bundle without ever being
  measured on its own; measured, it is nothing. "More votes at tile seams" was a
  mechanism, not a magnitude — and with 8× TTA already averaging every pixel,
  the extra seam votes were probably redundant from the start.

The pre-registered sum test can't be applied as literally as written: E4's
+0.0217 was at 6,026 images, this checkpoint is at 2,000. The at-scale
comparison is E7's side observation — the same bundle measured *unpaired* at
2,000 images gave +0.0138, where this paired run gives +0.0172. The ~0.003
disagreement is the size of the training-variance spread quoted in E4.
Everything is consistent with one story: **the E4 gain is TTA, full stop.**

Two tallies updated in passing: threshold tuning came back 0.45 for a fifth
run, buying +0.0003 — still wobble, still dead. And this run vs E7's run of the
identical config (raw 0.5582 vs 0.5632) is a second observation of training
variance on the full-image metric: **~0.005**, larger than the ~0.003 crop-metric
spread, and larger than several deltas this logbook cares about. That number is
the strongest argument yet for finishing E5 and quantifying variance properly.

**Decision.** `tile_overlap: 64` becomes the default in `_base.yaml`; TTA stays.
The 0.5971 headline stands as measured — it used overlap 128, and +0.0009 does
not buy a re-run to purify it. One recorded caveat: overlap was ablated with TTA
*off*, so a TTA×overlap interaction is formally unmeasured — implausible, since
both work by adding votes and votes are redundant, but unmeasured is unmeasured.
The inference half of E6 is closed. Next: E5's test-set carve, now the oldest
open item and the first thing a reviewer would ask about. → E5.

---

## Ledger of levers

| lever | status | evidence |
|---|---|---|
| architecture (D-LinkNet vs MiT-B0) | **settled** — tie, MiT-B0 on efficiency | E1 |
| training data volume | **spent** — all 6,026 used | E2 |
| epochs | **spent** — plateaus by ~12–14 | E1, E2 |
| threshold tuning | **dead** — no-op at two data scales | E1, E2 |
| post-processing | **spent** — +0.0028 paired at correct scale; surface is flat | E7 |
| TTA | **settled — +0.0160 paired**; the whole of E4's bundle gain | E4, E7, E8 |
| tile overlap 128 | **dead — +0.0009 paired**; default reverted to 64 | E8 |
| crop size (context) | open — targets the ~7-image tail, not the bulk | E3 |
| boundary-weighted loss | open, untried | E3 |
| held-out test set | open, planned | E5 |
| standalone inference (paired ablations) | **done** — round-trip exact, first ablation delivered | E8 |
| training reproducibility | open — `cudnn.benchmark`, weakens every Δ above | E6 |
| cross-dataset generalization | open, optional | E5 + protocol doc |

## Score history

| entry | change | iou_final |
|---|---|---|
| E0 | *(different metric — not comparable)* | ~~0.39~~ |
| E1 | MiT-B0, 2,000 imgs | 0.5494 |
| E2 | → 6,026 imgs | 0.5754 |
| E4 | → 8× TTA, overlap 128 | **0.5971** |
| E7 | → post-proc 9/500 | *(+0.0028, measured at 2,000 imgs; headline unchanged)* |
| E8 | *(paired decomposition at 2,000 imgs: TTA +0.0160, overlap +0.0009)* | *(headline unchanged)* |

Published D-LinkNet on DeepGlobe is ~0.63, at full-image training rather than
256px crops.
