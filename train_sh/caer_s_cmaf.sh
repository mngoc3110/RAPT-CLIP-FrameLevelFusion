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
if [ ! -f "${ANNOTATION_DIR}/train.txt" ] || [ ! -f "${ANNOTATION_DIR}/face.json" ]; then
  echo "=> [Preprocessor] Đang tự động chuyển đổi đường dẫn Colab sang Kaggle..."
  python tools/prepare_caers_dataset.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ANNOTATION_DIR}"
fi

TRAIN_TXT="${ANNOTATION_DIR}/train.txt"
VAL_TXT="${ANNOTATION_DIR}/validation.txt"
TEST_TXT="${ANNOTATION_DIR}/test.txt"
FACE_BOX="${ANNOTATION_DIR}/face.json"
BODY_BOX="${ANNOTATION_DIR}/body.json"

echo "=> File Train annotation: ${TRAIN_TXT}"
echo "=> File Val annotation:   ${VAL_TXT}"
echo "=> File Face BBox:        ${FACE_BOX}"
echo "=> File Body BBox:        ${BODY_BOX}"
echo "=> Thư mục CSVs đã đổi đường dẫn: ${ANNOTATION_DIR}/converted_csvs/"

# 3. Khởi chạy huấn luyện RAPT-CLIP
python -u main.py \
  --mode train \
  --exper-name CAER-S-CMAF \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 16 \
  --grad-accum-steps 2 \
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
  \
  --fusion-type cmaf \
  --fusion-stage frame \
  --temporal-layers 1 \
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
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --use-amp
