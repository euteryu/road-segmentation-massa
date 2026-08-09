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

# 2. Train once (~12 min at train_size 2000) to produce the checkpoint the
#    ablation below is measured on. Skip this line and attach a previous
#    version's output instead (Add Input -> Notebook Output) if you already have
#    one - src/predict.py autodetects /kaggle/input/**/checkpoints/*.pth.
!python main.py --only mit_b0_scaled

# 3. E8: the paired ablation E4 could not do. All three load the SAME frozen
#    weights, so each delta is attributable to the flag and nothing else - no
#    retraining, and none of the cudnn.benchmark variance that made E4's +0.0217
#    impossible to split. ~3 min each.
!python -m src.predict --name mit_b0_scaled --tag base                                # tta on,  overlap 128
!python -m src.predict --name mit_b0_scaled --tag no_tta --no-tta                     # tta OFF, overlap 128
!python -m src.predict --name mit_b0_scaled --tag ov64   --no-tta --tile-overlap 64   # neither

# TTA         = base   - no_tta
# overlap     = no_tta - ov64
# Compare the sum against E4's +0.0217; the shortfall was training variance.

# No post-processing sweep this time: E7 settled that stage (+0.0028, flat
# surface) and _base.yaml already carries its winner.

# 4. Save Version -> Save & Run All (Commit). ~21 min total, so NOT the
#    interactive session: interactive /kaggle/working is discarded on tab close.
