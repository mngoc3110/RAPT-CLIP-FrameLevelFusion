#!/bin/bash
# ============================================================================
# CAER-S Evaluation Script — Context-Aware Emotion Recognition (Static Images)
# ============================================================================
export PYTHONUNBUFFERED=1
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
  DETECTED_DIR=$(find /kaggle/input -maxdepth 4 -type f -name "*bbox_*.csv" 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  else
    DATA_DIR="/kaggle/input/datasets/lcngtr/caer-s"
  fi
fi

echo "=> Using CAER-S dataset root: ${DATA_DIR}"

ANNOTATION_DIR="/kaggle/working/caers_data"
if [ ! -f "${ANNOTATION_DIR}/test.txt" ] || [ ! -f "${ANNOTATION_DIR}/face.json" ]; then
  echo "=> [Preprocessor] Chuẩn bị annotation CAER-S..."
  python tools/prepare_caers_dataset.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ANNOTATION_DIR}"
fi

TEST_TXT="${ANNOTATION_DIR}/test.txt"
FACE_BOX="${ANNOTATION_DIR}/face.json"
BODY_BOX="${ANNOTATION_DIR}/body.json"

# 2. Nhận diện Checkpoint để đánh giá
CHECKPOINT="/kaggle/input/models/mngochocsupham/caers/tensorflow2/default/1/CAER-S-model.pth"

if [ -n "$1" ]; then
  CHECKPOINT="$1"
elif [ ! -f "$CHECKPOINT" ]; then
  for cp in "outputs/CAER-S/CAER-S-model.pth" "outputs/CAER-S/model_best.pth" "/kaggle/input/caers/tensorflow2/default/1/CAER-S-model.pth" "/kaggle/working/outputs/CAER-S-CMAF/model_best.pth"; do
    if [ -f "$cp" ]; then
      CHECKPOINT="$cp"
      break
    fi
  done
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "=> [Error] Không tìm thấy checkpoint tại: ${CHECKPOINT}"
  exit 1
fi

echo "=> Evaluating Checkpoint: ${CHECKPOINT}"

# 3. Chạy đánh giá
python -u main.py \
  --mode eval \
  --exper-name Eval-CAER-S-CMAF \
  --dataset CAER \
  --gpu 0 \
  --batch-size 32 \
  --workers 4 \
  --root-dir "${DATA_DIR}" \
  --test-annotation "${TEST_TXT}" \
  --clip-path ViT-B/16 \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  --eval-checkpoint "${CHECKPOINT}" \
  \
  --fusion-type cmaf \
  --fusion-stage frame \
  --temporal-layers 1 \
  --use-context \
  --crop-body \
  --mask-context-face \
  \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --use-amp
