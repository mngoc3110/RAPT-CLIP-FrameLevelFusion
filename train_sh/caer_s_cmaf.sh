#!/bin/bash
# ============================================================================
# CAER-S Training Script — Context-Aware Emotion Recognition (Static Images)
# ============================================================================
# CAER-S Dataset:
# - 7 emotion classes (0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise)
# - Static images (num_segments=1, duration=1)
# - Triple-stream architecture: Face + Body (torso estimation) + Context (full scene)
# - Fusion: Tridirectional Frame-Level CMAF (Cross-Modal Attention Fusion)
# - Loss: LDAM Loss (s=1.0, max_m=0.5) + MI Loss + DC Loss
# ============================================================================

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 1. Tự động nhận diện thư mục dataset trên Kaggle
POSSIBLE_DIRS=(
  "/kaggle/input/datasets/lcngtr/caer-s"
  "/kaggle/input/caer-s"
  "/kaggle/input/datasets/bearmn/caers-dataset-rapt-clip"
)

DATA_DIR=""
for d in "${POSSIBLE_DIRS[@]}"; do
  if [ -d "$d" ]; then
    DATA_DIR="$d"
    break
  fi
done

if [ -z "$DATA_DIR" ]; then
  # Fallback search nếu đặt tên thư mục tùy chỉnh
  DETECTED_DIR=$(find /kaggle/input -maxdepth 4 -type f -name "*bbox_*.csv" 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  else
    DATA_DIR="/kaggle/input/datasets/lcngtr/caer-s"
  fi
fi

echo "=> Using CAER-S dataset root: ${DATA_DIR}"

# 2. Tự động chuyển đổi đường dẫn Colab -> Kaggle và tạo file annotation
ANNOTATION_DIR="/kaggle/working/caers_data"
if [ "$REGEN" = "1" ] || [ ! -f "${ANNOTATION_DIR}/train.txt" ] || [ ! -f "${ANNOTATION_DIR}/face.json" ] || [ ! -f "${ANNOTATION_DIR}/validation.txt" ]; then
  echo "=> [Preprocessor] Đang tự động chuyển đổi đường dẫn Colab sang Kaggle..."
  python tools/prepare_caers_dataset.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ANNOTATION_DIR}"
fi

TRAIN_TXT="${ANNOTATION_DIR}/train.txt"
VAL_TXT="${ANNOTATION_DIR}/validation.txt"
TEST_TXT="${ANNOTATION_DIR}/test.txt"

# Nếu val.txt không tồn tại, tự động trỏ về test.txt
if [ ! -f "${VAL_TXT}" ] && [ -f "${TEST_TXT}" ]; then
  VAL_TXT="${TEST_TXT}"
fi

FACE_BOX="${ANNOTATION_DIR}/face.json"
BODY_BOX="${ANNOTATION_DIR}/body.json"

echo "=> File Train annotation: ${TRAIN_TXT}"
echo "=> File Val annotation:   ${VAL_TXT}"
echo "=> File Face BBox:        ${FACE_BOX}"
echo "=> File Body BBox:        ${BODY_BOX}"
echo "=> Thư mục CSVs đã đổi đường dẫn: ${ANNOTATION_DIR}/converted_csvs/"

# 3. Checkpoint Resume: Tự động tìm checkpoint thông minh (Auto-Resume)
RESUME_ARG=()
TOTAL_EPOCHS=20

# Xử lý tham số truyền vào: bash caer_s_cmaf.sh [CHECKPOINT_PATH] [EPOCHS]
if [ -n "$1" ]; then
  if [ -f "$1" ]; then
    echo "=> [Resume] Sử dụng checkpoint người dùng chỉ định: $1"
    RESUME_ARG=("--resume" "$1")
    if [ -n "$2" ]; then
      TOTAL_EPOCHS="$2"
      echo "=> [Epochs] Đặt lại tổng số epoch: ${TOTAL_EPOCHS}"
    fi
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    TOTAL_EPOCHS="$1"
    echo "=> [Epochs] Người dùng đặt tổng số epoch: ${TOTAL_EPOCHS}"
  else
    echo "=> [Cảnh báo] Checkpoint '$1' không tồn tại trên hệ thống!"
  fi
fi

if [ ${#RESUME_ARG[@]} -eq 0 ]; then
  LATEST_CKPT=""

  # 1. Quét tìm checkpoint trong outputs/ của session hiện tại (/kaggle/working hoặc local)
  SEARCH_OUTPUT_DIRS=("/kaggle/working/outputs" "outputs")
  for out_d in "${SEARCH_OUTPUT_DIRS[@]}"; do
    if [ -d "$out_d" ]; then
      FOUND_CKPT=$(find "$out_d" -type f -name "model.pth" 2>/dev/null | sort -V | tail -n 1)
      if [ -z "$FOUND_CKPT" ]; then
        FOUND_CKPT=$(find "$out_d" -type f -name "model_best.pth" 2>/dev/null | sort -V | tail -n 1)
      fi
      if [ -n "$FOUND_CKPT" ] && [ -f "$FOUND_CKPT" ]; then
        LATEST_CKPT="$FOUND_CKPT"
        break
      fi
    fi
  done

  # 2. Quét tìm tự động mọi checkpoint trên /kaggle/input (dù người dùng đặt tên dataset là gì)
  if [ -z "$LATEST_CKPT" ] && [ -d "/kaggle/input" ]; then
    # Tìm file có chứa caer hoặc model hoặc best
    LATEST_CKPT=$(find /kaggle/input -type f \( -name "*caer*.pth" -o -name "*model*.pth" -o -name "*best*.pth" \) 2>/dev/null | grep -iE "caer|model|checkpoint" | head -n 1)
  fi

  if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
    echo "=> [Auto-Resume] Tự động tìm thấy checkpoint: ${LATEST_CKPT}"
    RESUME_ARG=("--resume" "${LATEST_CKPT}")
  else
    echo "=> [Info] Không tìm thấy checkpoint cũ! Huấn luyện từ đầu (Train from scratch) với pretrained CLIP ViT-B/16!"
    if [ -d "/kaggle/input" ]; then
      echo "=> [Gợi ý] Danh sách các file .pth đang có trong /kaggle/input:"
      find /kaggle/input -type f -name "*.pth" 2>/dev/null || echo "   (Không có file .pth nào)"
    fi
  fi
fi

# 4. Khởi chạy huấn luyện RAPT-CLIP CAER-S Tối Ưu
python -u main.py \
  --mode train \
  --exper-name CAER-S-CMAF \
  --dataset CAER \
  --gpu 0 \
  --epochs ${TOTAL_EPOCHS} \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --print-freq 50 \
  --seed 42 \
  --workers 4 \
  --root-dir "${DATA_DIR}" \
  --train-annotation "${TRAIN_TXT}" \
  --val-annotation "${VAL_TXT}" \
  --test-annotation "${TEST_TXT}" \
  --clip-path ViT-B/16 \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  "${RESUME_ARG[@]}" \
  \
  --fusion-type cmaf \
  --fusion-stage frame \
  --temporal-layers 1 \
  --temporal-pooling attn \
  --use-context \
  --crop-body \
  --mask-context-face \
  \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  \
  --loss-type ldam \
  --ldam-s 1.0 \
  --ldam-max-m 0.5 \
  --use-weighted-sampler \
  --label-smoothing 0.05 \
  --mixup-alpha 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 3 \
  --mi-ramp 6 \
  --dc-warmup 3 \
  --dc-ramp 6 \
  --grad-clip 1.0 \
  \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --drop-path-rate 0.0 \
  --modality-dropout 0.0 \
  --use-amp
