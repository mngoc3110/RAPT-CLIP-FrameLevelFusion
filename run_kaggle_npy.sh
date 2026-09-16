#!/bin/bash

# Script chuyên dụng để chạy trên KAGGLE

# 1. Đường dẫn thư mục Dataset trên Kaggle (Lưu ý: phải trỏ vào trong thư mục con cvpr_emotic vì dataset bị lồng 2 lớp)
KAGGLE_DATA_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn/cvpr_emotic"

# 2. Thư mục Output (có quyền ghi) trên Kaggle
WORKING_DIR="/kaggle/working/emotic_pre"

echo "=== BƯỚC 1: TIỀN XỬ LÝ (TẠO FILE NPY) ==="
# Chỉ tạo NPY nếu thư mục chưa tồn tại (để tránh tạo lại nhiều lần nếu bạn chạy lại cell)
if [ ! -d "$WORKING_DIR" ]; then
    echo "Đang khởi tạo các file NPY. Quá trình này sẽ mất một lúc..."
    python3 mat2py.py \
        --data_dir "$KAGGLE_DATA_DIR" \
        --save_dir "$WORKING_DIR" \
        --generate_npy
else
    echo "Thư mục $WORKING_DIR đã tồn tại. Bỏ qua bước tiền xử lý!"
fi

echo "=== BƯỚC 2: BẮT ĐẦU HUẤN LUYỆN (TRAINING) ==="

EXP_NAME="EMOTIC_KAGGLE_NPY"
BATCH_SIZE=16
EPOCHS=25

python3 main.py \
    --mode train \
    --exper-name ${EXP_NAME} \
    --dataset EMOTIC \
    --use-npy \
    --npy-dir "$WORKING_DIR" \
    --gpu cuda \
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

echo "=== HOÀN TẤT ==="
