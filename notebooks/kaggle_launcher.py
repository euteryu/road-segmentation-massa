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

# 2. Sanity check first - 1 epoch on 64 images, ~1 min. Catches a broken path or
#    a bad config before you spend 40 minutes finding out.
!python main.py --only mit_b0_scaled --epochs 1 --train-size 64

# 3. The real run. Watch the "~Xm left" estimate printed after epoch 1; if it is
#    more than you want to sit through, interrupt and rerun with fewer epochs -
#    the best checkpoint so far is already on disk either way.
!python main.py

# 4. Re-scoring costs no GPU and reads the predictions already saved in
#    outputs/preds/, so every setting below scores the SAME probability maps -
#    perfectly paired, immune to the training variance that muddies E4.
!python -m src.sweep_postproc --name mit_b0_scaled

# 5. Save Version (Quick Save) to persist outputs/ as a permanent snapshot.
