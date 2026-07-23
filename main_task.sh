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

# Load global_config.yaml once: python prints KEY=value lines, eval binds them.
eval "$(python3 <<'PY'
import shlex, yaml
c = yaml.safe_load(open("global_config.yaml"))
def q(v): return shlex.quote(str(v))
print(f"DATA_YAML={q(c['yolo_prep']['output_dir'] + '/object.yaml')}")
print(f"MODEL={q(c['yolo_train']['model'])}")
print(f"EPOCHS={q(c['yolo_train']['epochs'])}")
print(f"IMGSZ={q(c['yolo_train']['imgsz'])}")
print(f"CFG_RUN_DIR={q(c['yolo_train']['run_dir'])}")
print(f"RUN_NAME={q(c['yolo_train']['run_name'])}")
# Pseudo-label retrain
pseudo_dir = c['yolo_prep']['output_dir'] + c['pseudo_label']['pseudo_suffix']
print(f"PSEUDO_YAML={q(pseudo_dir + '/object_pseudo.yaml')}")
print(f"PSEUDO_NAME={q(c['pseudo_label']['run_name'])}")
print(f"PSEUDO_CKPT={q(c['pseudo_label']['ckpt_out'])}")
PY
)"
# Absolute path: ultralytics prepends its default runs dir (runs/obb/) to any
# RELATIVE project= path, which is how runs/obb/runs/obb/... doubling happens.
RUN_DIR="$PWD/$CFG_RUN_DIR"

# ── 1. Dataset prep (skipped if already built) ────────────────────────────────
# if [ ! -f "$DATA_YAML" ]; then
#     echo "=== Building YOLO OBB dataset ==="
#     python prep/yolo_prep.py
# fi

# ── 2. Train (disabled for now — dataset prep only) ───────────────────────────
# echo "=== Training YOLO OBB ==="
# yolo obb train \
#     model="$MODEL" \
#     data="$DATA_YAML" \
#     epochs=$EPOCHS \
#     imgsz=$IMGSZ \
#     project="$RUN_DIR" \
#     name="$RUN_NAME" \
#     exist_ok=True

# Publish the best checkpoint where the downstream scripts expect it
# BEST="$RUN_DIR/$RUN_NAME/weights/best.pt"
# mkdir -p checkpoints
# cp "$BEST" checkpoints/yolo.pt
# echo "Best checkpoint copied to checkpoints/yolo.pt"

# ── 3. Evaluate on the unseen test split (disabled for now) ───────────────────
# echo "=== Evaluating on test split ==="
# yolo obb val \
#     model=checkpoints/yolo.pt \
#     data="$DATA_YAML" \
#     split=test \
#     imgsz=$IMGSZ \
#     project="$RUN_DIR" \
#     name="${RUN_NAME}_test" \
#     exist_ok=True

# echo "=== Done. Test results in $RUN_DIR/${RUN_NAME}_test ==="

# echo "=== Done. Dataset built at data/object ==="

# ── 4. Pseudo-label all splits with the independent COCO model ────────────────
# Fills the ~22% missing cattle labels so retraining no longer punishes the
# model for detecting real cows (the cause of confidence suppression).
echo "=== Building pseudo-labeled dataset (all splits) ==="
python prep/pseudo_label.py

# ── 5. Retrain the OBB model on the corrected labels ──────────────────────────
echo "=== Retraining YOLO OBB on pseudo-labels ==="
yolo obb train \
    model="$MODEL" \
    data="$PSEUDO_YAML" \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    project="$RUN_DIR" \
    name="$PSEUDO_NAME" \
    exist_ok=True

# Publish the retrained best.pt under a NEW name (does not overwrite yolo.pt,
# so you can compare before switching interaction_prep over to it).
PSEUDO_BEST="$RUN_DIR/$PSEUDO_NAME/weights/best.pt"
mkdir -p "$(dirname "$PSEUDO_CKPT")"
cp "$PSEUDO_BEST" "$PSEUDO_CKPT"
echo "Retrained checkpoint copied to $PSEUDO_CKPT"

# ── 6. Evaluate the retrained model on the pseudo test split ──────────────────
echo "=== Evaluating retrained model on pseudo test split ==="
yolo obb val \
    model="$PSEUDO_CKPT" \
    data="$PSEUDO_YAML" \
    split=test \
    imgsz=$IMGSZ \
    project="$RUN_DIR" \
    name="${PSEUDO_NAME}_test" \
    exist_ok=True

echo "=== Done. Retrained model: $PSEUDO_CKPT ==="
echo "    Compare F1/P/R curves in $RUN_DIR/${PSEUDO_NAME}_test against the"
echo "    original model. If better, point paths.yolo_ckpt at $PSEUDO_CKPT."

# ── (later) Interaction dataset prep — run AFTER settling on a detector ────────
# echo "=== Building interaction dataset ==="
# rm -rf data/interaction data/annotated/annotated_interaction.csv
# python prep/interaction_prep.py
# python prep/pose_vis.py

