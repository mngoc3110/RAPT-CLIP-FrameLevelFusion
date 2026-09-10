#!/bin/bash

# DAiSEE (Engagement) Training Script
EXP_NAME="DAiSEE_RAPT_CLIP"
BATCH_SIZE=8 # Small batch size due to 8 frames per clip
EPOCHS=35

# Kaggle Dataset Paths
ROOT_DIR="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
TRAIN_ANNOT="${ROOT_DIR}/daisee_train.txt"
VAL_ANNOT="${ROOT_DIR}/daisee_val.txt"
TEST_ANNOT="${ROOT_DIR}/daisee_test.txt"

python main.py \
    --mode train \
    --exper-name ${EXP_NAME} \
    --dataset DAiSEE \
    --gpu 0 \
    --workers 4 \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --optimizer AdamW \
    --lr 2e-5 \
    --lr-image-encoder 5e-7 \
    --lr-prompt-learner 3e-4 \
    --lr-adapter 1e-4 \
    --weight-decay 0.005 \
    --milestones 10 15 \
    --gamma 0.1 \
    --use-amp \
    --mixup-alpha 0.0 \
    --modality-dropout 0.3 \
    --fusion-type cmaf \
    --use-context \
    --crop-body \
    --num-segments 8 \
    --duration 1 \
    --image-size 224 \
    --temperature 0.07 \
    --drop-path-rate 0.1 \
    --grad-clip 1.0 \
    --lambda_mi 0.1 \
    --lambda_dc 0.1 \
    --mi-warmup 2 \
    --mi-ramp 5 \
    --loss-type ldam \
    --ldam-s 1.0 \
    --ldam-max-m 0.5 \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --root-dir ${ROOT_DIR} \
    --train-annotation ${TRAIN_ANNOT} \
    --val-annotation ${VAL_ANNOT} \
    --test-annotation ${TEST_ANNOT}
