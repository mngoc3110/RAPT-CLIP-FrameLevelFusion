#!/bin/bash

# Evaluation Script — EMOTIC Test Set
# Usage: bash train_sh/emotic_eval.sh /path/to/model_best.pth

CHECKPOINT="${1:-outputs/EMOTIC_RAPT_CLIP_ASL_v3/model_best.pth}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Usage: bash train_sh/emotic_eval.sh /path/to/model_best.pth"
    exit 1
fi

echo "=> Evaluating checkpoint: $CHECKPOINT"

# Kaggle Dataset Paths
BASE_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn"
ROOT_DIR="${BASE_DIR}/cvpr_emotic/cvpr_emotic"
ANNOT_DIR="${BASE_DIR}"
TEST_ANNOT="${ANNOT_DIR}/test_bbox.txt"
BBOX_FACE="${ANNOT_DIR}/emotic_face_bboxes_mtcnn.json"
BBOX_BODY="${ANNOT_DIR}/emotic_body_bboxes.json"

python main.py \
    --mode eval \
    --eval-checkpoint ${CHECKPOINT} \
    --dataset EMOTIC \
    --gpu 0 \
    --workers 4 \
    --batch-size 32 \
    --clip-path ViT-B/16 \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --num-segments 1 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07 \
    --temporal-layers 1 \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --text-type prompt_ensemble \
    --root-dir ${ROOT_DIR} \
    --test-annotation ${TEST_ANNOT} \
    --train-annotation ${ANNOT_DIR}/train_bbox.txt \
    --val-annotation ${ANNOT_DIR}/val_bbox.txt \
    --bounding-box-face ${BBOX_FACE} \
    --bounding-box-body ${BBOX_BODY}
