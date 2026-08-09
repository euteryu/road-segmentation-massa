# Kaggle notebook - paste this as a SINGLE cell.
# Everything else lives in the repo, version-controlled and diffable.

# 1. Step OUT of the repo before deleting it. On a re-run the kernel is still
#    sitting inside road-segmentation-massa from last time, and removing the
#    directory you are standing in leaves the process with no working directory -
#    after which every shell command fails with "getcwd: cannot access parent
#    directories" until you chdir somewhere that exists.
%cd /kaggle/working
!rm -rf road-segmentation-massa
!git clone -q https://github.com/euteryu/road-segmentation-massa.git
%cd /kaggle/working/road-segmentation-massa
!pip install -q -r requirements.txt

# ---------------------------------------------------------------------------
# E8: the paired TTA/overlap ablation. Needs a checkpoint, NOT a training run.
#
# Attach a previous version's output as a dataset (Add Input -> Your Work ->
# the version whose log ends in "training done"); src/predict.py autodetects it
# under /kaggle/input/**/checkpoints/. Each --tag writes its own preds dir, so
# all three score the SAME weights and the deltas are attributable to the flag
# and nothing else. ~3 min each, no GPU spent on training.
# ---------------------------------------------------------------------------
!python -m src.predict --name mit_b0_scaled --tag base                        # tta on,  overlap 128
!python -m src.predict --name mit_b0_scaled --tag no_tta  --no-tta            # tta OFF, overlap 128
!python -m src.predict --name mit_b0_scaled --tag ov64    --no-tta --tile-overlap 64  # neither

# If no checkpoint is attached, predict exits immediately with the list of
# attached datasets - uncomment to retrain one first (~12 min at train_size 2000),
# then re-run the three lines above.
# !python main.py --only mit_b0_scaled

# Save Version (Quick Save) to persist outputs/ as a permanent snapshot.
