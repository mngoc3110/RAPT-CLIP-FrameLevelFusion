#!/bin/bash

# Configuration
EXP_NAME="EMOTIC_RAPT_CLIP_ASL"
BATCH_SIZE=8
EPOCHS=15

# Kaggle Dataset Paths
ROOT_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn"
TRAIN_ANNOT="${ROOT_DIR}/Annotations/train.txt"
VAL_ANNOT="${ROOT_DIR}/Annotations/val.txt"
TEST_ANNOT="${ROOT_DIR}/Annotations/test.txt"

# Assuming bbox JSONs are also placed in Annotations
BBOX_FACE="${ROOT_DIR}/Annotations/face_bbox.json"
BBOX_BODY="${ROOT_DIR}/Annotations/body_bbox.json"

python main.py \
    --exper-name ${EXP_NAME} \
    --dataset EMOTIC \
    --gpu 0 \
    --workers 4 \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --optimizer AdamW \
    --lr 2e-5 \
    --lr-prompt-learner 2e-4 \
    --lr-adapter 1e-4 \
    --use-amp \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --num-segments 1 \
    --duration 1 \
    --root-dir ${ROOT_DIR} \
    --train-annotation ${TRAIN_ANNOT} \
    --val-annotation ${VAL_ANNOT} \
    --test-annotation ${TEST_ANNOT} \
    --bounding-box-face ${BBOX_FACE} \
    --bounding-box-body ${BBOX_BODY}
