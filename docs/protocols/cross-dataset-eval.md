# Protocol — cross-dataset evaluation and transfer

**Status:** planned, not executed. Written *before* running anything.

This is a **pre-registration**: the method, the confounds, and the falsifiable
expectations are fixed in writing first, so the result cannot be retrofitted into
whichever story it happens to support. If the protocol changes after seeing a
number, that change gets recorded here with its reason, not silently applied.

Companion to `docs/LOGBOOK.md` E5. The logbook records **what happened**; this
records **how a not-yet-run experiment will be run**. Keeping them separate is the
point — a logbook that also holds plans stops being a record.

---

## 1. The question

Two distinct questions, often conflated. Answer them in order.

**Q1 (generalization).** Does a model trained on DeepGlobe work on road imagery it
has never seen — different sensor, geography, road type? This is *zero-shot
transfer*.

**Q2 (transfer value).** If it doesn't, is the DeepGlobe training still worth
reusing as a starting point, versus just labelling target-domain data and training
from scratch? This is the question a startup actually has to answer, because it
decides whether to spend money on labelling or on compute.

Q1 alone is a limitations paragraph. Q1 + Q2 is an experiment.

---

## 2. Ground sample distance — the confound to kill first

The model did not learn "roads". It learned **roads at 0.5 m/pixel**. A 6 m road is
~12 px wide in DeepGlobe. Feed it 1 m/px imagery and that same road is ~6 px wide —
half the width every learned filter expects. It would score badly for a reason that
has nothing to do with generalization.

**Rule: resample the new data to the training GSD, never the reverse.** The model's
learned scale is frozen; the data is what's flexible.

### Resampling protocol

**Never resample the ground truth.** Resample the input, run inference, bring the
prediction back down, score against the untouched original mask.

| step | operation | interpolation | why |
|---|---|---|---|
| 1 | upsample image to 0.5 m/px | `INTER_CUBIC` | continuous brightness values; bicubic is sharpest for upsampling photos |
| 2 | sliding-window inference | — | unchanged: 256px tiles, same overlap, same TTA as the DeepGlobe runs |
| 3 | downsample **probability map** to native | `INTER_AREA` | continuous [0,1] values; area averaging cannot overshoot the range |
| 4 | threshold, then score vs **original** mask | — | ground truth never touched, so it cannot be corrupted |

**Why interpolation choice matters, and where it silently destroys data.**
Averaging is only meaningful on *continuous* values. A probability of 0.7 halfway
between 0.9 and 0.5 genuinely means "moderately confident". A **mask** holds
*categorical labels* — 0 = not road, 1 = road — and halfway between them is 0.5,
which means nothing; there is no half a road-label. Bilinear-resizing a mask
produces a blurred ring of fractional values around every road edge, which must
then be thresholded, arbitrarily shifting the boundary. Ground truth is corrupted
and **nothing raises an error**. If a mask must ever be resized: `INTER_NEAREST`,
no exceptions. Step 4 avoids the situation entirely.

`INTER_CUBIC` on the probability map would also be wrong, for a smaller reason:
bicubic's negative lobes overshoot, yielding "probabilities" below 0 or above 1.

**Upsampling adds no information.** 1 m/px imagery genuinely resolves less detail;
×2 merely presents it at the scale the model expects. Scores still drop — but the
drop is then attributable to *domain shift*, which is what we're measuring, rather
than *scale mismatch*, which is an artifact.

---

## 3. Dataset choice

Preferred order. Earlier entries have fewer confounds, so a clean result there is
worth more than a dramatic result later.

| dataset | GSD | resample | notes |
|---|---|---|---|
| **CHN6-CUG** (primary) | 0.5 m/px | **none** | 4,511 images, 512×512, 6 Chinese cities. Same GSD as DeepGlobe, so §2 does not apply at all and there is no resampling argument to have. Urban China vs rural South/SE Asia is still a real shift. |
| **SpaceNet 3** (rigorous) | 0.3 m/px | ×0.6 | 4 cities (Vegas, Paris, Shanghai, Khartoum) = four independent shift measurements. Satellite, same sensor family. Labels are centreline **vectors**, so road width is buffered by us — we can match DeepGlobe's width exactly and eliminate the label-convention confound below. |
| Massachusetts Roads | 1 m/px | ×2 | 1,171 images @1500², Boston. Aerial not satellite, biggest resample, unknown label width. Most confounded — do last, if at all. (Named in this repo for historical reasons only; the project has always used DeepGlobe.) |

### Label-convention check — run before trusting any number

Both DeepGlobe and Massachusetts render roads as fixed-width buffers around
centrelines. **If the two use different widths in metres, IoU is penalised by a
labelling convention rather than by model quality**, and the comparison is void.

Cheap to rule out: measure mean road width in metres in both mask sets
(road pixel count / centreline length, × GSD). Record both numbers here before
running. If they differ by more than ~20%, either switch to SpaceNet (where we
control the buffer) or report the mismatch alongside the result.

Also: Massachusetts tiles contain **blank no-data regions**. Mask them out of
scoring or they register as false positives.

---

## 4. Experiment arms

Three arms. Arm A is the baseline the other two are measured against — running B
without A is uninterpretable.

| arm | training | measures |
|---|---|---|
| **A — zero-shot** | none; DeepGlobe checkpoint as-is | Q1. Pure generalization. |
| **B — fine-tuned** | A + small target-domain sample (start ~200 images) | Q2. What transfer buys. |
| **C — from scratch** | target-domain sample only, same size, same schedule | Q2. The honest control. |

**B vs C is the whole point.** If B ≫ C, the DeepGlobe training is a reusable asset
and the right move is to buy compute. If B ≈ C, it isn't, and the right move is to
buy labels. That comparison is only valid if B and C see the **same target images
and the same schedule** — otherwise it measures budget, not transfer.

### Fine-tuning strategy for arm B

Cheapest first; stop when it stops helping.

1. Freeze encoder, train decoder only — fastest, least prone to overfitting on a
   small target set.
2. Unfreeze the last encoder stage as well.
3. Fine-tune everything at low LR, encoder at ~1/10 the decoder's rate.

Early layers hold generic edge and texture detectors that transfer across domains;
later layers hold domain-specific structure. That is why the freezing order runs
back-to-front.

Note this project already does transfer learning: MiT-B0 arrives
ImageNet-pretrained and was fine-tuned onto DeepGlobe. Arm B is the same mechanism
one step further out.

---

## 5. Protocol constants

- Metric: dataset IoU, identical definition to `src/evaluate.py`. Report the same
  raw / tuned / postproc stages so numbers are comparable to E1–E4.
- **Threshold is NOT retuned on the target set** for arm A. Zero-shot means
  zero-shot; retuning on target data makes it few-shot and answers a different
  question. Report target-tuned as a *separate* line if wanted.
- Evaluate on the target dataset's **official test split** where one exists
  (Massachusetts: 49 images). Measuring, not training — do not run all 1,171.
- Fixed seed 42, as everywhere else in this project.

---

## 6. Falsifiable expectations — recorded before the run

Written in advance so they can be wrong in public.

1. Arm A drops **substantially** below the DeepGlobe val figure. A large drop is
   the expected, legitimate finding — not a failure, and a better report section
   than a flattering number.
2. Arm A degrades **least on CHN6-CUG** (no resample, same GSD) and **most on
   Massachusetts** (aerial, 2× resample).
3. Arm A's failures are **detection**, not boundary — the opposite of the
   DeepGlobe-domain finding in E3, which was boundary-dominated. Unfamiliar road
   types should be missed outright rather than traced with fuzzy edges.
4. Arm B ≫ arm C at ~200 target images. Transfer should dominate when target data
   is scarce; the gap should narrow as target data grows.

If (3) is wrong and errors are still boundary-dominated, that means the model
transfers *semantically* but not *geometrically*, which would point at label-width
mismatch rather than domain shift — go back and re-check §3.

---

## 7. Sequencing

Do **not** start before E4 (TTA) lands. Changing the evaluation surface mid-flight
breaks comparability with E1–E4. Order:

1. E4 completes; logbook Result/Read filled in.
2. Carve the held-out DeepGlobe test set (logbook E5) — an in-distribution test
   must exist before an out-of-distribution one means anything.
3. Label-width check (§3).
4. Arm A on CHN6-CUG.
5. Arms B and C, only if arm A shows a gap worth closing.
