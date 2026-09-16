#!/bin/bash

# Script huấn luyện RAPT-CLIP sử dụng trực tiếp các file .npy 
# (Tăng tốc I/O, nạp toàn bộ ma trận ảnh thẳng vào RAM)

EXP_NAME="EMOTIC_NPY"
BATCH_SIZE=16
EPOCHS=25

echo "Starting EMOTIC training with preprocessed .npy files..."

python3 main.py \
    --mode train \
    --exper-name ${EXP_NAME} \
    --dataset EMOTIC \
    --use-npy \
    --npy-dir emotic_dataset/emotic_pre \
    --gpu mps \
    --workers 4 \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --optimizer AdamW \
    --lr 1e-4 \
    --lr-image-encoder 1e-5 \
    --lr-prompt-learner 5e-5 \
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
    --num-segments 1 \
    --duration 1 \
    --image-size 224 \
    --temperature 1.0 \
    --loss-type dynamic_asl \
    --drop-path-rate 0.0 \
    --grad-clip 1.0 \
    --lambda_mi 0.0 \
    --lambda_dc 0.0 \
    --lambda_vad 0.1 \
    --mi-warmup 7 \
    --mi-ramp 5 \
    --dc-warmup 7 \
    --dc-ramp 5 \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True

echo "Training complete!"
