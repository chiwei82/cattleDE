#!/usr/bin/env bash

#SBATCH --job-name=cattle_act_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --account=COSC021063
#SBATCH --output=log/out/%j.out
#SBATCH --error=log/err/%j.err

set -euo pipefail

cd /user/work/yx25778/cattleDE

module load languages/python/3.12.3
source venv/bin/activate

DATA_YAML=data/object/object.yaml
RUN_DIR=log/yolo          # ultralytics writes runs here
RUN_NAME=obb_train
IMGSZ=1280
EPOCHS=50

# ── 1. Dataset prep (skipped if already built) ────────────────────────────────
# if [ ! -f "$DATA_YAML" ]; then
#     echo "=== Building YOLO OBB dataset ==="
#     python prep/yolo_prep.py
# fi

# ── 2. Train (disabled for now — dataset prep only) ───────────────────────────
echo "=== Training YOLO OBB ==="
yolo obb train \
    model=yolo11n-obb.pt \
    data="$DATA_YAML" \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    project="$RUN_DIR" \
    name="$RUN_NAME" \
    exist_ok=True

# Publish the best checkpoint where the downstream scripts expect it
BEST="$RUN_DIR/$RUN_NAME/weights/best.pt"
mkdir -p checkpoints
cp "$BEST" checkpoints/yolo.pt
echo "Best checkpoint copied to checkpoints/yolo.pt"

# ── 3. Evaluate on the unseen test split (disabled for now) ───────────────────
echo "=== Evaluating on test split ==="
yolo obb val \
    model=checkpoints/yolo.pt \
    data="$DATA_YAML" \
    split=test \
    imgsz=$IMGSZ \
    project="$RUN_DIR" \
    name="${RUN_NAME}_test" \
    exist_ok=True

echo "=== Done. Test results in $RUN_DIR/${RUN_NAME}_test ==="

echo "=== Done. Dataset built at data/object ==="
