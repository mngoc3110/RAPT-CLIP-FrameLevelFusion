#!/bin/bash

# Configuration
EXP_NAME="EMOTIC_RAPT_CLIP_ASL"
BATCH_SIZE=8
EPOCHS=35

# Kaggle Dataset Paths
BASE_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn"
ROOT_DIR="${BASE_DIR}/cvpr_emotic/cvpr_emotic"  # actual images live here

# Annotation files live in the cvpr_emotic/ (outer) folder
ANNOT_DIR="${BASE_DIR}/cvpr_emotic"
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
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 3e-4 \
    --lr-adapter 1e-4 \
    --weight-decay 0.005 \
    --milestones 10 15 \
    --gamma 0.1 \
    --use-amp \
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
    --mi-warmup 2 \
    --mi-ramp 5 \
    --dc-warmup 2 \
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
