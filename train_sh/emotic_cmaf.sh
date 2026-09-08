#!/bin/bash

# Configuration — Balanced Anti-Overfit v3
EXP_NAME="EMOTIC_RAPT_CLIP_ASL_v3"
BATCH_SIZE=16
EPOCHS=20  # Valid plateaus ~Epoch 10, no need for 35 epochs

# Kaggle Dataset Paths
BASE_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn"
ROOT_DIR="${BASE_DIR}/cvpr_emotic/cvpr_emotic"  # actual images live here

# Annotation files live at BASE_DIR level
ANNOT_DIR="${BASE_DIR}"
TRAIN_ANNOT="${ANNOT_DIR}/train_bbox.txt"
VAL_ANNOT="${ANNOT_DIR}/val_bbox.txt"
TEST_ANNOT="${ANNOT_DIR}/test_bbox.txt"

BBOX_FACE="${ANNOT_DIR}/emotic_face_bboxes_mtcnn.json"
BBOX_BODY="${ANNOT_DIR}/emotic_body_bboxes.json"

python main.py \
    --mode train \
    --exper-name ${EXP_NAME} \
    --dataset EMOTIC \
    --gpu 0 \
    --workers 4 \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --optimizer AdamW \
    --lr 2e-5 \
    --lr-image-encoder 0 \
    --lr-prompt-learner 5e-5 \
    --lr-adapter 1e-4 \
    --weight-decay 0.01 \
    --milestones 5 8 \
    --gamma 0.1 \
    --use-amp \
    --freeze-image-encoder \
    --mixup-alpha 0.2 \
    --modality-dropout 0.3 \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --num-segments 1 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07 \
    --drop-path-rate 0.1 \
    --grad-clip 1.0 \
    --lambda_mi 0.1 \
    --lambda_dc 0.1 \
    --mi-warmup 7 \
    --mi-ramp 5 \
    --dc-warmup 7 \
    --dc-ramp 5 \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --root-dir ${ROOT_DIR} \
    --train-annotation ${TRAIN_ANNOT} \
    --val-annotation ${VAL_ANNOT} \
    --test-annotation ${TEST_ANNOT} \
    --bounding-box-face ${BBOX_FACE} \
    --bounding-box-body ${BBOX_BODY}
