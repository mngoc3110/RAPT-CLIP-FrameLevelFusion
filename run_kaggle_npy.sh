#!/bin/bash

# Script chuyên dụng để chạy trên KAGGLE

# 1. Đường dẫn thư mục Dataset trên Kaggle
KAGGLE_DATA_DIR="/kaggle/input/datasets/bearmn/emotic-dataset-rapt-clip-bearmn"

# 2. Thư mục Output (có quyền ghi) trên Kaggle
WORKING_DIR="/kaggle/working/emotic_pre"

echo "=== BƯỚC 1: TIỀN XỬ LÝ (TẠO FILE NPY) ==="

# Hàm kiểm tra file có tồn tại và lớn hơn 1MB không (tránh file lỗi rỗng)
check_valid_npy() {
    local file="$1"
    if [ -f "$file" ]; then
        local size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
        if [ "$size" -gt 1000000 ]; then
            return 0 # Hợp lệ
        fi
    fi
    return 1 # Không hợp lệ hoặc rỗng
}

if ! check_valid_npy "$WORKING_DIR/train_context_arr.npy"; then
    echo "Phát hiện NPY chưa có hoặc bị rỗng do lỗi cũ. Đang dọn dẹp và chạy lại..."
    rm -rf "$WORKING_DIR"
    python3 mat2py.py \
        --data_dir "$KAGGLE_DATA_DIR" \
        --save_dir "$WORKING_DIR" \
        --generate_npy
else
    echo "File NPY hợp lệ đã tồn tại trong $WORKING_DIR. Bỏ qua bước tiền xử lý!"
fi

echo "=== BƯỚC 2: BẮT ĐẦU HUẤN LUYỆN (TRAINING) ==="

EXP_NAME="EMOTIC_KAGGLE_NPY"
BATCH_SIZE=26
EPOCHS=25

python3 main.py \
    --mode train \
    --exper-name ${EXP_NAME} \
    --dataset EMOTIC \
    --use-npy \
    --npy-dir "$WORKING_DIR" \
    --gpu 0 \
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
