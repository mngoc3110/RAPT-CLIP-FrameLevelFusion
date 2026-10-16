#!/bin/bash
# ============================================================================
# CAER Video Training Script — Context-Aware Emotion Recognition + CMAF
# ============================================================================
# CAER video dataset:
# - 7 emotion classes (0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise)
# - Video clips (.avi / .mp4, ~60-100 frames, num_segments=16, duration=1)
# - Multi-stream architecture: Face (MediaPipe/BBox), Body (BBox), Context (masked face)
# - Fusion: Tridirectional Frame-Level CMAF -> Temporal Transformer
# - Loss: Cross-Entropy with Label Smoothing (0.1) for SOTA WAR benchmark
# ============================================================================

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OPENCV_LOG_LEVEL=OFF
export OPENCV_FFMPEG_LOGLEVEL=0

# Check CUDA availability
python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
  echo ""
  echo "============================================================================"
  echo "⚠️  [CẢNH BÁO / WARNING] CUDA KHÔNG KHẢ DỤNG (CUDA is NOT available)!"
  echo "    Bạn đang chạy trên CPU! Huấn luyện video trên CPU sẽ rất chậm."
  echo "    👉 Trên Kaggle: Bấm menu bên phải (Notebook Options) -> Accelerator"
  echo "       chọn 'GPU T4 x2' hoặc 'GPU P100' rồi chạy lại!"
  echo "============================================================================"
  echo ""
fi

# Parse optional arguments (--subset, --balance, or pass-through)
SPLIT_CHOICE="${TRAIN_SPLIT:-train.txt}"
EXTRA_ARGS=()
for arg in "$@"; do
  case $arg in
    --subset)
      SPLIT_CHOICE="train_subset.txt"
      ;;
    --balance)
      SPLIT_CHOICE="train_balance.txt"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

# 1. Tự động nhận diện thư mục Video CAER (Kaggle / Colab / Local)
POSSIBLE_DATA_DIRS=(
  "/kaggle/input/datasets/bearmn/caer-video-dataset/CAER"
  "/kaggle/input/datasets/bearmn/caer-video-dataset"
  "/kaggle/input/caer-video-dataset/CAER"
  "/kaggle/input/caer-video-dataset"
  "/kaggle/input/caer-dataset/CAER"
  "/kaggle/input/caer-dataset"
  "/kaggle/input/caer-video/CAER"
  "/kaggle/input/caer-video"
  "/kaggle/input/caer/CAER"
  "/kaggle/input/caer"
  "/kaggle/input/datasets/bearmn/caers-dataset-rapt-clip"
  "/Users/macbook/Downloads/RAPT-CLIP/CAER_Video"
  "/Users/macbook/Downloads/RAPT-CLIP/CAER"
)

DATA_DIR=""
for d in "${POSSIBLE_DATA_DIRS[@]}"; do
  if [ -d "$d" ]; then
    DATA_DIR="$d"
    break
  fi
done

if [ -z "$DATA_DIR" ] && [ -d "/kaggle/input" ]; then
  # Fallback: Tự động tìm thư mục chứa 'train/Anger' hoặc 'test/Anger'
  DETECTED_DIR=$(find /kaggle/input -maxdepth 6 -type d \( -name "Anger" -o -name "anger" \) 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  fi
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/kaggle/input/datasets/bearmn/caer-video-dataset/CAER"
fi

# Nếu DATA_DIR trỏ tới thư mục cha mà bên trong có folder CAER/
if [ -d "${DATA_DIR}/CAER/train" ]; then
  DATA_DIR="${DATA_DIR}/CAER"
fi

echo "============================================================================"
echo "=> [1/4] Thư mục CAER Video root: ${DATA_DIR}"

# 2. Tự động nhận diện thư mục Annotation & Bounding Boxes
# Ưu tiên dataset Kaggle: 'caer-video-annotation' của user 'traincollab'
POSSIBLE_ANNOT_DIRS=(
  "/kaggle/input/datasets/traincollab/caer-video-annotation"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/CAER-video-annotation"
  "/kaggle/input/caer-video-annotation"
  "/kaggle/input/caer-video-annotation/CAER-video-annotation"
  "/kaggle/input/caer-video-annotations"
  "/kaggle/input/caer-video-annotations/CAER-video-annotations"
  "/kaggle/input/CAER-video-annotation"
  "/kaggle/input/caer-annotation"
  "/kaggle/input/caer-annotations"
  "${DATA_DIR}"
  "annotations/caer_video"
  "CAER_Video"
  "/Users/macbook/Downloads/RAPT-CLIP/CAER_Video"
  "/Users/macbook/Downloads/RAPT-CLIP/annotations/caer_video"
)

# Thêm tất cả đường dẫn khớp wildcard nếu có
if [ -d "/kaggle/input/datasets" ]; then
  for d in /kaggle/input/datasets/*/caer-video-annotation*; do
    if [ -d "$d" ]; then
      POSSIBLE_ANNOT_DIRS=("$d" "${POSSIBLE_ANNOT_DIRS[@]}")
    fi
  done
fi

ANNOTATION_DIR="${ANNOTATION_DIR:-${ANNOT_DIR:-}}"
if [ -z "$ANNOTATION_DIR" ]; then
  for d in "${POSSIBLE_ANNOT_DIRS[@]}"; do
    if [ -d "$d" ] && { [ -f "$d/train.txt" ] || [ -f "$d/body_boxes.json" ] || [ -f "$d/face_boxes_mediapipe.json" ] || [ -f "$d/face_boxes.json" ]; }; then
      ANNOTATION_DIR="$d"
      break
    fi
  done
fi

if [ -z "$ANNOTATION_DIR" ] && [ -d "/kaggle/input" ]; then
  DETECTED_ANNOT=$(find /kaggle/input -maxdepth 8 \( -name "body_boxes.json" -o -name "face_boxes*.json" \) 2>/dev/null | head -n 1 | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_ANNOT" ] && [ -d "$DETECTED_ANNOT" ]; then
    ANNOTATION_DIR="$DETECTED_ANNOT"
  fi
fi

if [ -z "$ANNOTATION_DIR" ]; then
  ANNOTATION_DIR="annotations/caer_video"
fi

echo "=> [2/4] Thư mục Annotation root: ${ANNOTATION_DIR}"

# 3. Chọn file Annotations (Train / Val / Test)
TRAIN_TXT="${ANNOTATION_DIR}/${SPLIT_CHOICE}"
if [ ! -f "$TRAIN_TXT" ]; then
  if [ -f "${ANNOTATION_DIR}/train.txt" ]; then
    TRAIN_TXT="${ANNOTATION_DIR}/train.txt"
  elif [ -f "${ANNOTATION_DIR}/train_balance.txt" ]; then
    TRAIN_TXT="${ANNOTATION_DIR}/train_balance.txt"
  elif [ -f "annotations/caer_video/${SPLIT_CHOICE}" ]; then
    TRAIN_TXT="annotations/caer_video/${SPLIT_CHOICE}"
  elif [ -f "annotations/caer_video/train.txt" ]; then
    TRAIN_TXT="annotations/caer_video/train.txt"
  elif [ -f "${DATA_DIR}/train.txt" ]; then
    TRAIN_TXT="${DATA_DIR}/train.txt"
  fi
fi

VAL_TXT="${ANNOTATION_DIR}/validation.txt"
if [ ! -f "$VAL_TXT" ]; then
  if [ -f "annotations/caer_video/validation.txt" ]; then
    VAL_TXT="annotations/caer_video/validation.txt"
  elif [ -f "${DATA_DIR}/validation.txt" ]; then
    VAL_TXT="${DATA_DIR}/validation.txt"
  fi
fi

TEST_TXT="${ANNOTATION_DIR}/test.txt"
if [ ! -f "$TEST_TXT" ]; then
  if [ -f "annotations/caer_video/test.txt" ]; then
    TEST_TXT="annotations/caer_video/test.txt"
  elif [ -f "${DATA_DIR}/test.txt" ]; then
    TEST_TXT="${DATA_DIR}/test.txt"
  fi
fi

echo "=> [3/4] File Danh Sách Dữ Liệu:"
echo "   - Train: $TRAIN_TXT"
echo "   - Val:   $VAL_TXT"
echo "   - Test:  $TEST_TXT"

# 4. Tự động nhận diện Bounding Boxes (Face & Body)
# Face: Ưu tiên face_boxes_mediapipe.json (100% video clips & mọi frame) -> face_boxes.json -> face_boxes_optimized.json
FACE_BOX=""
CANDIDATE_FACE_BOXES=(
  "${ANNOTATION_DIR}/face_boxes_mediapipe.json"
  "${ANNOTATION_DIR}/face_boxes.json"
  "${ANNOTATION_DIR}/face_boxes_optimized.json"
  "${ANNOTATION_DIR}/face.json"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/face_boxes_mediapipe.json"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/face_boxes.json"
  "${DATA_DIR}/face_boxes_mediapipe.json"
  "${DATA_DIR}/face_boxes.json"
  "${DATA_DIR}/face.json"
  "CAER_Video/face_boxes_mediapipe.json"
  "annotations/caer_video/face_boxes.json"
)

for fb in "${CANDIDATE_FACE_BOXES[@]}"; do
  if [ -f "$fb" ]; then
    FACE_BOX="$fb"
    break
  fi
done

# Body: Ưu tiên body_boxes.json (100% video clips) -> body.json
BODY_BOX=""
CANDIDATE_BODY_BOXES=(
  "${ANNOTATION_DIR}/body_boxes.json"
  "${ANNOTATION_DIR}/body.json"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/body_boxes.json"
  "${DATA_DIR}/body_boxes.json"
  "${DATA_DIR}/body.json"
  "CAER_Video/body_boxes.json"
  "annotations/caer_video/body_boxes.json"
)

for bb in "${CANDIDATE_BODY_BOXES[@]}"; do
  if [ -f "$bb" ]; then
    BODY_BOX="$bb"
    break
  fi
done

echo "=> [4/4] Bounding Box Dữ Liệu:"
echo "   - Face BBox: $FACE_BOX"
echo "   - Body BBox: $BODY_BOX"
echo "============================================================================"

python -u main.py \
  --mode train \
  --exper-name CAER-Video-CMAF \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
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
  --lr-image-encoder 1e-5 \
  --lr-prompt-learner 3e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  \
  --loss-type ce \
  --label-smoothing 0.1 \
  --mixup-alpha 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --grad-clip 1.0 \
  \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --use-amp \
  "${EXTRA_ARGS[@]}"
