#!/bin/bash

# Configuration — v6 (anti-overfit + tail-class rescue)
# Changes from v5:
#   - Scheduler: MultiStepLR [10,15] → CosineAnnealing + 3-epoch warmup
#   - weight_decay: 0.005 → 0.02
#   - lr_prompt_learner: 5e-5 → 5e-6 (prevent prompt memorization)
#   - lr_adapter: 1e-4 → 2e-5 (slower adapter learning)
#   - lr_image_encoder: 1e-6 → 0 (fully frozen)
#   - mixup_alpha: 0.0 → 0.4 (data augmentation)
#   - modality_dropout: 0.0 → 0.1 (prevent modality co-dependence)
#   - asl_gamma_neg: 4.0 → 2.0 (less tail-class suppression)
#   - epochs: 25 → 35 (cosine needs more epochs)
#   - early_stopping_patience: 0 → 8
#   - Dynamic weights: per-batch → global frequency

EXP_NAME="EMOTIC_CMAF_DECOUPLED_v6"
BATCH_SIZE=26
EPOCHS=35

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
    --lr-prompt-learner 5e-6 \
    --lr-adapter 2e-5 \
    --weight-decay 0.02 \
    --scheduler cosine \
    --warmup-epochs 3 \
    --use-amp \
    --freeze-image-encoder \
    --mixup-alpha 0.4 \
    --modality-dropout 0.1 \
    --fusion-type cmaf_decoupled \
    --loss-type dynamic_asl \
    --asl-gamma-neg 2.0 \
    --asl-gamma-pos 0.0 \
    --asl-clip 0.05 \
    --early-stopping-patience 8 \
    --use-context \
    --crop-body \
    --num-segments 1 \
    --duration 1 \
    --image-size 224 \
    --temperature 1.0 \
    --drop-path-rate 0.0 \
    --grad-clip 1.0 \
    --lambda_mi 0.0 \
    --lambda_dc 0.0 \
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
