#!/bin/bash
# ============================================================================
# DAiSEE Training Script — Engagement Recognition (Video) + CMAF
# ============================================================================
# DAiSEE dataset:
# - E-learning engagement recognition (4 affective states, focus on Engagement)
# - SOTA setup: 3-Class Merged (Very Low + Low -> Low Engagement) or 4-Class
# - Multi-stream architecture:
#     * Face stream: Face crop focusing on micro-expressions, eye gaze, blinking
#     * Body stream: Upper-body posture crop focusing on shoulders, slouching, head tilt
#     * Context stream: Full webcam view (face masked to prevent shortcut leakage)
# - Fusion: Tridirectional Frame-Level CMAF -> Temporal Transformer
# - Loss: LDAM Loss with s=1.0, max_m=0.5, class-weights, label smoothing (0.05)
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
  echo "    Bạn đang chạy trên CPU! Huấn luyện video DAiSEE trên CPU sẽ rất chậm."
  echo "    👉 Trên Kaggle: Bấm menu bên phải (Notebook Options) -> Accelerator"
  echo "       chọn 'GPU T4 x2' hoặc 'GPU P100' rồi chạy lại!"
  echo "============================================================================"
  echo ""
fi

# Parse optional arguments
MERGE_LOW_FLAG="--daisee-merge-low"
EXTRA_ARGS=()
for arg in "$@"; do
  case $arg in
    --4class|--no-merge)
      MERGE_LOW_FLAG=""
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

# 1. Tự động nhận diện thư mục dữ liệu DAiSEE (Kaggle / Colab / Local)
POSSIBLE_DATA_DIRS=(
  "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
  "/kaggle/input/datasets/mngochocsupham/daisee"
  "/kaggle/input/daisee/DAiSEE_data"
  "/kaggle/input/daisee"
  "/kaggle/input/DAiSEE_data"
  "/kaggle/input/DAiSEE"
  "DAiSEE/DAiSEE_data"
  "DAiSEE"
  "/Users/macbook/Downloads/RAPT-CLIP/DAiSEE"
)

# Thêm đường dẫn wildcard nếu có
if [ -d "/kaggle/input/datasets" ]; then
  for d in /kaggle/input/datasets/*/*daisee*/DAiSEE_data /kaggle/input/datasets/*/*daisee*; do
    if [ -d "$d" ]; then
      POSSIBLE_DATA_DIRS=("$d" "${POSSIBLE_DATA_DIRS[@]}")
    fi
  done
fi

DATA_DIR=""
for d in "${POSSIBLE_DATA_DIRS[@]}"; do
  if [ -f "$d/daisee_train.txt" ] || [ -f "$d/train.txt" ]; then
    DATA_DIR="$d"
    break
  fi
done

if [ -z "$DATA_DIR" ] && [ -d "/kaggle/input" ]; then
  DETECTED_FILE=$(find /kaggle/input -maxdepth 8 \( -iname "daisee_train.txt" -o -iname "daisee_test.txt" \) 2>/dev/null | head -n 1)
  if [ -n "$DETECTED_FILE" ]; then
    DATA_DIR=$(dirname "$DETECTED_FILE")
  fi
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
fi

echo "============================================================================"
echo "=> [1/4] Thư mục DAiSEE root: ${DATA_DIR}"

# 2. Nhận diện file Annotation (Train, Val, Test)
TRAIN_TXT="$DATA_DIR/daisee_train.txt"
[ ! -f "$TRAIN_TXT" ] && [ -f "$DATA_DIR/train.txt" ] && TRAIN_TXT="$DATA_DIR/train.txt"

VAL_TXT="$DATA_DIR/daisee_val.txt"
[ ! -f "$VAL_TXT" ] && [ -f "$DATA_DIR/val.txt" ] && VAL_TXT="$DATA_DIR/val.txt"
[ ! -f "$VAL_TXT" ] && [ -f "$DATA_DIR/validation.txt" ] && VAL_TXT="$DATA_DIR/validation.txt"

TEST_TXT="$DATA_DIR/daisee_test.txt"
[ ! -f "$TEST_TXT" ] && [ -f "$DATA_DIR/test.txt" ] && TEST_TXT="$DATA_DIR/test.txt"

echo "=> [2/4] File Danh Sách Dữ Liệu:"
echo "   - Train: $TRAIN_TXT"
echo "   - Val:   $VAL_TXT"
echo "   - Test:  $TEST_TXT"

# 3. Tự động tìm hoặc sinh Face Bounding Box (YuNet / MTCNN)
FACE_BBOX_PATH="none"
POSSIBLE_BBOX=(
  "$DATA_DIR/daisee_face_bbox.json"
  "$DATA_DIR/face_bbox.json"
  "/kaggle/working/daisee_face_bbox.json"
  "DAiSEE/daisee_face_bbox.json"
  "daisee_face_bbox.json"
)

for f in "${POSSIBLE_BBOX[@]}"; do
  if [ -f "$f" ]; then
    SIZE=$(wc -c < "$f" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 500 ]; then
      FACE_BBOX_PATH="$f"
      break
    fi
  fi
done

if [ "$FACE_BBOX_PATH" = "none" ] && [ -d "/kaggle/input" ]; then
  DETECTED_BBOX=$(find /kaggle/input -maxdepth 8 -name "*daisee*face*.json" 2>/dev/null | head -n 1)
  if [ -n "$DETECTED_BBOX" ]; then
    FACE_BBOX_PATH="$DETECTED_BBOX"
  fi
fi

if [ "$FACE_BBOX_PATH" = "none" ] && [ -f "$TRAIN_TXT" ]; then
  echo "=> Notice: daisee_face_bbox.json not found. Generating high-precision Face Boxes via YuNet..."
  OUT_BBOX="/kaggle/working/daisee_face_bbox.json"
  [ ! -d "/kaggle/working" ] && OUT_BBOX="DAiSEE/daisee_face_bbox.json"
  python -u tools/generate_daisee_face_boxes.py \
    --root-dir "$DATA_DIR" \
    --annotations "$TRAIN_TXT" "$VAL_TXT" "$TEST_TXT" \
    --output "$OUT_BBOX" \
    --samples-per-video 5 \
    --device auto
  if [ -f "$OUT_BBOX" ] && [ $(wc -c < "$OUT_BBOX" 2>/dev/null || echo 0) -gt 500 ]; then
    FACE_BBOX_PATH="$OUT_BBOX"
  fi
fi

echo "=> [3/4] Face Bounding Box: $FACE_BBOX_PATH"
echo "=> [4/4] Chế độ phân loại: ${MERGE_LOW_FLAG:-"4-Class Standard (0: Very Low, 1: Low, 2: High, 3: Very High)"}"
echo "============================================================================"

python -u main.py \
  --mode train \
  --exper-name DAiSEE-CMAF \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --print-freq 50 \
  --seed 42 \
  --workers 4 \
  --early-stopping-patience 0 \
  --root-dir "$DATA_DIR" \
  --train-annotation "$TRAIN_TXT" \
  --val-annotation "$VAL_TXT" \
  --test-annotation "$TEST_TXT" \
  --clip-path ViT-B/16 \
  --bounding-box-face "$FACE_BBOX_PATH" \
  --bounding-box-body "none" \
  \
  --fusion-type cmaf \
  --temporal-layers 1 \
  --fusion-stage frame \
  --use-context \
  --crop-body \
  --mask-context-face \
  ${MERGE_LOW_FLAG} \
  \
  --optimizer AdamW \
  --lr 3e-5 \
  --lr-image-encoder 5e-7 \
  --lr-prompt-learner 1e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  \
  --loss-type ldam \
  --ldam-s 1.0 \
  --ldam-max-m 0.5 \
  --use-class-weight \
  --label-smoothing 0.05 \
  --mixup-alpha 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 3 \
  --mi-ramp 5 \
  --dc-warmup 3 \
  --dc-ramp 5 \
  --grad-clip 1.0 \
  \
  --text-type prompt_ensemble \
  --ctx-init "a video of a student in a classroom" \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --drop-path-rate 0.1 \
  --use-amp \
  "${EXTRA_ARGS[@]}"
