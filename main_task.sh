#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
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

# ── 4-6. Pseudo-label + retrain + eval (DONE — retrained model published) ─────
# Completed: checkpoints/yolo_pseudo.pt is now paths.yolo_ckpt. Re-enable this
# block only to rebuild the pseudo dataset or retrain the detector.
# echo "=== Building pseudo-labeled dataset (all splits) ==="
# python prep/pseudo_label.py
#
# echo "=== Retraining YOLO OBB on pseudo-labels ==="
# yolo obb train \
#     model="$MODEL" \
#     data="$PSEUDO_YAML" \
#     epochs=$EPOCHS \
#     imgsz=$IMGSZ \
#     project="$RUN_DIR" \
#     name="$PSEUDO_NAME" \
#     exist_ok=True
#
# PSEUDO_BEST="$RUN_DIR/$PSEUDO_NAME/weights/best.pt"
# mkdir -p "$(dirname "$PSEUDO_CKPT")"
# cp "$PSEUDO_BEST" "$PSEUDO_CKPT"
# echo "Retrained checkpoint copied to $PSEUDO_CKPT"
#
# echo "=== Evaluating retrained model on pseudo test split ==="
# yolo obb val \
#     model="$PSEUDO_CKPT" \
#     data="$PSEUDO_YAML" \
#     split=test \
#     imgsz=$IMGSZ \
#     project="$RUN_DIR" \
#     name="${PSEUDO_NAME}_test" \
#     exist_ok=True

# ── 7. Action dataset prep (data/action + annotated_action.csv) ───────────────
# echo "=== Building action dataset ==="
# python prep/action_prep.py

# ── 8. Action metric-learning training (triplet + zero-mean reg, image-only) ──
# Produces paths.action_ckpt (timm ViT-B/16 encoder); its latent space is reused
# as the interaction model's ViT backbone.
echo "=== Training action encoder (metric learning) ==="
python -m train.action_with_image

# ── 9. Merge Label Studio labels -> annotated_interaction_test.csv ────────────
# data/ann.json (Label Studio export) must be present. Writes a NEW _test CSV
# (label_v1/label_v2) without touching the original blank CSV.
# echo "=== Merging interaction labels ==="
# python data/merge_label_studio_export.py

# ── 10. Interaction training (BINARY output; LDAM + InfoNCE; image-only) ──────
# Reads annotated_interaction_test.csv; inits ViT from paths.action_ckpt.
echo "=== Training interaction classifier (binary) ==="
python -m train.interaction_with_image --csv annotated_interaction_test.csv

echo "=== Done. action.ckpt + interaction.ckpt trained. ==="

# ── NOTE: interaction dataset prep (the crops) is intentionally NOT re-run here.
# The crops already exist and ann.json labels reference them by image_path;
# re-running would delete + regenerate them with a blank CSV and invalidate the
# labels. Only run it to (re)generate a fresh candidate set to annotate:
#   rm -rf data/interaction data/annotated/annotated_interaction.csv
#   python prep/interaction_prep.py
#
# ── Demo video — DISABLED: action is now a metric-learning ENCODER, so
# scripts/short_demo.py's per-cow action inference (which expects a classifier)
# is broken until it is switched to a k-NN gallery.
#   python scripts/short_demo.py

