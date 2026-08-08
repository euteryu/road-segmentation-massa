# Kaggle notebook - paste this as a SINGLE cell.
# Everything else lives in the repo, version-controlled and diffable.

# 1. Fresh code every session (~30s). Kaggle wipes /kaggle/working between sessions;
#    only the GPU-heavy part is worth being careful about, not this.
!rm -rf /kaggle/working/road-segmentation-massa
!git clone -q https://github.com/euteryu/road-segmentation-massa.git /kaggle/working/road-segmentation-massa
%cd /kaggle/working/road-segmentation-massa
!pip install -q -r requirements.txt

# 2. Sanity check first - 1 epoch on 64 images, ~1 min. Catches a broken path or
#    a bad config before you spend 40 minutes finding out.
!python main.py --only mit_b0_scaled --epochs 1 --train-size 64

# 3. The real run. Watch the "~Xm left" estimate printed after epoch 1; if it is
#    more than you want to sit through, interrupt and rerun with fewer epochs -
#    the best checkpoint so far is already on disk either way.
!python main.py

# 4. Re-scoring costs no GPU. Tune post-processing as many times as you like
#    against the predictions already saved in outputs/preds/.
# !python -m src.evaluate --name mit_b0_scaled --min-area 128 --close-kernel 7

# 5. Save Version (Quick Save) to persist outputs/ as a permanent snapshot.
