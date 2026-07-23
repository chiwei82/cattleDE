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
print(f"DATA_YAML={shlex.quote(c['yolo_prep']['output_dir'] + '/object.yaml')}")
print(f"MODEL={shlex.quote(str(c['yolo_train']['model']))}")
print(f"EPOCHS={shlex.quote(str(c['yolo_train']['epochs']))}")
print(f"IMGSZ={shlex.quote(str(c['yolo_train']['imgsz']))}")
print(f"CFG_RUN_DIR={shlex.quote(str(c['yolo_train']['run_dir']))}")
print(f"RUN_NAME={shlex.quote(str(c['yolo_train']['run_name']))}")
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

# ── 4. Interaction dataset prep (video_dir comes from global_config.yaml) ─────
# Filter parameters changed (yolo_conf/iou/sample_fps), so previous outputs are
# stale AND the incremental-resume logic would skip every video if the old CSV
# is still present. Clean rebuild:
echo "=== Removing stale interaction outputs ==="
rm -rf data/interaction data/annotated/annotated_interaction.csv

echo "=== Building interaction dataset ==="
python prep/interaction_prep.py

# ── 5. Pose visualization sanity check (simu/pose) ────────────────────────────
# Sample count and keypoint threshold come from pose_vis in global_config.yaml.
echo "=== Rendering pose visualizations ==="
python prep/pose_vis.py

echo "=== Done. Interaction data in data/interaction, pose check in simu/pose ==="

