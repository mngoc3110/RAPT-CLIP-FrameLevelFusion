#!/bin/bash
# ============================================================================
# CAER Video Training Script — Context-Aware Emotion Recognition + CMAF
# ============================================================================
# CAER video dataset:
# - 7 emotion classes (0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise)
# - Video clips (.avi / .mp4, ~60-100 frames)
# - Multi-stream architecture: Face (MediaPipe/BBox), Body (BBox), Context (masked face)
# - Fusion: Tridirectional Frame-Level CMAF -> Temporal Transformer
# - Loss: LDAM Loss with s=1.5, max_m=0.5 and class-weights (resolves Neutral collapse)
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

normalize_path() {
  local p="$1"
  if [ -n "$p" ]; then
    if [[ "$p" == content/* ]]; then
      p="/$p"
    elif [[ "$p" == drive/* ]]; then
      p="/content/$p"
    fi
  fi
  echo "$p"
}

# Parse optional arguments
SPLIT_CHOICE="${TRAIN_SPLIT:-train.txt}"
LOSS_CHOICE="${LOSS_TYPE:-ldam}"
NUM_SEGS="${NUM_SEGMENTS:-16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-4}"
LR="${LR:-0.0001}"
LR_IMAGE="${LR_IMAGE:-1e-05}"
LR_PROMPT="${LR_PROMPT:-0.0005}"
LR_ADAPTER="${LR_ADAPTER:-0.0002}"
SCHEDULER="${SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
EXTRA_ARGS=()

for arg in "$@"; do
  case $arg in
    --subset)
      SPLIT_CHOICE="train_subset.txt"
      ;;
    --balance)
      SPLIT_CHOICE="train_balance.txt"
      ;;
    --ce)
      LOSS_CHOICE="ce"
      ;;
    --focal)
      LOSS_CHOICE="focal"
      ;;
    --ldam)
      LOSS_CHOICE="ldam"
      ;;
    --16seg|--segments-16)
      NUM_SEGS="16"
      ;;
    --8seg|--segments-8)
      NUM_SEGS="8"
      ;;
    --batch-size=*|-b=*)
      BATCH_SIZE="${arg#*=}"
      ;;
    --workers=*|-w=*)
      WORKERS="${arg#*=}"
      ;;
    --lr=*)
      LR="${arg#*=}"
      ;;
    --lr-image=*|--lr-image-encoder=*)
      LR_IMAGE="${arg#*=}"
      ;;
    --lr-prompt=*|--lr-prompt-learner=*)
      LR_PROMPT="${arg#*=}"
      ;;
    --lr-adapter=*)
      LR_ADAPTER="${arg#*=}"
      ;;
    --cosine)
      SCHEDULER="cosine"
      ;;
    --multistep)
      SCHEDULER="multistep"
      ;;
    --scheduler=*)
      SCHEDULER="${arg#*=}"
      ;;
    --warmup=*|--warmup-epochs=*)
      WARMUP_EPOCHS="${arg#*=}"
      ;;
    --data-dir=*)
      DATA_DIR="$(normalize_path "${arg#*=}")"
      ;;
    --annot-dir=*)
      ANNOTATION_DIR="$(normalize_path "${arg#*=}")"
      ;;
    --train-txt=*)
      TRAIN_TXT="$(normalize_path "${arg#*=}")"
      ;;
    --val-txt=*)
      VAL_TXT="$(normalize_path "${arg#*=}")"
      ;;
    --test-txt=*)
      TEST_TXT="$(normalize_path "${arg#*=}")"
      ;;
    --face-box=*)
      FACE_BOX="$(normalize_path "${arg#*=}")"
      ;;
    --ldam-s=*)
      LDAM_S="${arg#*=}"
      ;;
    --body-box=*)
      BODY_BOX="$(normalize_path "${arg#*=}")"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

LDAM_S="${LDAM_S:-1.5}"
DATA_DIR="$(normalize_path "${DATA_DIR:-}")"
ANNOTATION_DIR="$(normalize_path "${ANNOTATION_DIR:-${ANNOT_DIR:-}}")"
TRAIN_TXT="$(normalize_path "${TRAIN_TXT:-}")"
VAL_TXT="$(normalize_path "${VAL_TXT:-}")"
TEST_TXT="$(normalize_path "${TEST_TXT:-}")"
FACE_BOX="$(normalize_path "${FACE_BOX:-}")"
BODY_BOX="$(normalize_path "${BODY_BOX:-}")"

# If TRAIN_TXT is already provided, derive ANNOTATION_DIR if empty
if [ -n "$TRAIN_TXT" ] && [ -f "$TRAIN_TXT" ]; then
  if [ -z "$ANNOTATION_DIR" ]; then
    ANNOTATION_DIR="$(dirname "$TRAIN_TXT")"
  fi
fi

if [ "$LOSS_CHOICE" = "ldam" ]; then
  LOSS_ARGS=(--loss-type ldam --ldam-s "${LDAM_S}" --ldam-max-m 0.5 --use-weighted-sampler --label-smoothing 0.05)
elif [ "$LOSS_CHOICE" = "focal" ]; then
  LOSS_ARGS=(--loss-type focal --focal-gamma 2.0 --use-weighted-sampler --use-class-weight)
else
  LOSS_ARGS=(--loss-type ce --label-smoothing 0.05 --use-weighted-sampler)
fi

# 0. Ưu tiên trực tiếp đường dẫn Google Drive CAER Video nếu tồn tại
COLAB_TARGET_DIR="/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
if [ -d "$COLAB_TARGET_DIR" ]; then
  DATA_DIR="${DATA_DIR:-$COLAB_TARGET_DIR}"
  ANNOTATION_DIR="${ANNOTATION_DIR:-$COLAB_TARGET_DIR}"
fi

# 1. Tự động nhận diện thư mục Video CAER (Colab / Kaggle / Local)
POSSIBLE_DATA_DIRS=(
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video"
  "/content/drive/MyDrive/CAER_Video/CAER_Video"
  "/content/drive/MyDrive/CAER_Video"
  "/content/CAER_Video/CAER_Video"
  "/content/CAER_Video"
  "/content/CAER"
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

DATA_DIR="${DATA_DIR:-}"
if [ -z "$DATA_DIR" ]; then
  for d in "${POSSIBLE_DATA_DIRS[@]}"; do
    if [ -d "$d" ]; then
      DATA_DIR="$d"
      break
    fi
  done
fi

if [ -z "$DATA_DIR" ] && [ -d "/content/drive/MyDrive" ]; then
  DETECTED_DIR=$(find "/content/drive/MyDrive" -maxdepth 8 -type d \( -name "Anger" -o -name "anger" \) 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  fi
fi

if [ -z "$DATA_DIR" ] && [ -d "/kaggle/input" ]; then
  DETECTED_DIR=$(find /kaggle/input -maxdepth 6 -type d \( -name "Anger" -o -name "anger" \) 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  fi
fi

if [ -z "$DATA_DIR" ]; then
  if [ -d "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video" ]; then
    DATA_DIR="/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  else
    DATA_DIR="/kaggle/input/datasets/bearmn/caer-video-dataset/CAER"
  fi
fi

if [ -d "${DATA_DIR}/CAER/train" ]; then
  DATA_DIR="${DATA_DIR}/CAER"
fi

echo "============================================================================"
echo "=> [1/4] Thư mục CAER Video root: ${DATA_DIR}"

# 2. Tự động nhận diện thư mục Annotation & Bounding Boxes
POSSIBLE_ANNOT_DIRS=(
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/annotations"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/annotations"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video"
  "/content/CAER_Video/CAER_Video"
  "/content/CAER_Video"
  "/content/CAER"
  "/content/drive/MyDrive/CAER_Video/CAER_Video"
  "/content/drive/MyDrive/CAER_Video"
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

if [ -z "$ANNOTATION_DIR" ] && [ -d "/content/drive/MyDrive" ]; then
  DETECTED_ANNOT=$(find "/content/drive/MyDrive" -maxdepth 8 \( -name "body_boxes.json" -o -name "face_boxes*.json" -o -name "train.txt" \) 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_ANNOT" ] && [ -d "$DETECTED_ANNOT" ]; then
    ANNOTATION_DIR="$DETECTED_ANNOT"
  fi
fi

if [ -z "$ANNOTATION_DIR" ] && [ -d "/kaggle/input" ]; then
  DETECTED_ANNOT=$(find /kaggle/input -maxdepth 8 \( -name "body_boxes.json" -o -name "face_boxes*.json" \) 2>/dev/null | head -n 1 | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_ANNOT" ] && [ -d "$DETECTED_ANNOT" ]; then
    ANNOTATION_DIR="$DETECTED_ANNOT"
  fi
fi

if [ -z "$ANNOTATION_DIR" ]; then
  if [ -d "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video" ]; then
    ANNOTATION_DIR="/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  else
    ANNOTATION_DIR="annotations/caer_video"
  fi
fi

echo "=> [2/4] Thư mục Annotation root: ${ANNOTATION_DIR}"

# 3. Chọn file Annotations (Train / Val / Test)
if [ -z "$TRAIN_TXT" ]; then
  TRAIN_TXT="${ANNOTATION_DIR}/${SPLIT_CHOICE}"
  if [ ! -f "$TRAIN_TXT" ]; then
    if [ -f "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/train.txt" ]; then
      TRAIN_TXT="/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/train.txt"
    elif [ -f "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/train.txt" ]; then
      TRAIN_TXT="content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/train.txt"
    elif [ -f "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/train.txt" ]; then
      TRAIN_TXT="/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/train.txt"
    elif [ -f "${ANNOTATION_DIR}/train.txt" ]; then
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
fi

if [ -z "$VAL_TXT" ] || [ ! -f "$VAL_TXT" ]; then
  for v in "${ANNOTATION_DIR}/validation.txt" "${ANNOTATION_DIR}/val.txt" \
           "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/validation.txt" \
           "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/val.txt" \
           "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/validation.txt" \
           "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/val.txt" \
           "annotations/caer_video/validation.txt" "annotations/caer_video/val.txt" \
           "${DATA_DIR}/validation.txt" "${DATA_DIR}/val.txt"; do
    if [ -f "$v" ]; then
      VAL_TXT="$v"
      break
    fi
  done
fi

if [ -z "$TEST_TXT" ] || [ ! -f "$TEST_TXT" ]; then
  for t in "${ANNOTATION_DIR}/test.txt" \
           "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/test.txt" \
           "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/test.txt" \
           "annotations/caer_video/test.txt" \
           "${DATA_DIR}/test.txt"; do
    if [ -f "$t" ]; then
      TEST_TXT="$t"
      break
    fi
  done
fi

echo "=> [3/4] File Danh Sách Dữ Liệu:"
echo "   - Train: $TRAIN_TXT"
echo "   - Val:   $VAL_TXT"
echo "   - Test:  $TEST_TXT"

# 4. Tự động nhận diện Bounding Boxes (Face & Body)
CANDIDATE_FACE_BOXES=(
  "${ANNOTATION_DIR}/face_boxes_mediapipe.json"
  "${ANNOTATION_DIR}/face_boxes.json"
  "${ANNOTATION_DIR}/face_boxes_optimized.json"
  "${ANNOTATION_DIR}/face.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/face_boxes_mediapipe.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/face_boxes.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/face.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/face_boxes_mediapipe.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/face_boxes.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/face.json"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/face_boxes.json"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/face_boxes.json"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/face_boxes_mediapipe.json"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/face_boxes.json"
  "${DATA_DIR}/face_boxes_mediapipe.json"
  "${DATA_DIR}/face_boxes.json"
  "${DATA_DIR}/face.json"
  "CAER_Video/face_boxes_mediapipe.json"
  "annotations/caer_video/face_boxes.json"
)

if [ -z "$FACE_BOX" ] || [ ! -f "$FACE_BOX" ]; then
  for fb in "${CANDIDATE_FACE_BOXES[@]}"; do
    if [ -f "$fb" ]; then
      FACE_BOX="$fb"
      break
    fi
  done
fi

CANDIDATE_BODY_BOXES=(
  "${ANNOTATION_DIR}/body_boxes.json"
  "${ANNOTATION_DIR}/body.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/body_boxes.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/body.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/body_boxes.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/body.json"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/body_boxes.json"
  "content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/body_boxes.json"
  "/kaggle/input/datasets/traincollab/caer-video-annotation/body_boxes.json"
  "${DATA_DIR}/body_boxes.json"
  "${DATA_DIR}/body.json"
  "CAER_Video/body_boxes.json"
  "annotations/caer_video/body_boxes.json"
)

if [ -z "$BODY_BOX" ] || [ ! -f "$BODY_BOX" ]; then
  for bb in "${CANDIDATE_BODY_BOXES[@]}"; do
    if [ -f "$bb" ]; then
      BODY_BOX="$bb"
      break
    fi
  done
fi

echo "=> [4/4] Cấu hình Huấn luyện:"
echo "   - Face BBox:  $FACE_BOX"
echo "   - Body BBox:  $BODY_BOX"
echo "   - Loss Type:  $LOSS_CHOICE (${LOSS_ARGS[*]})"
echo "   - Batch Size: $BATCH_SIZE"
echo "   - Workers:    $WORKERS"
echo "   - LR Main:    $LR (ViT Image: $LR_IMAGE, Prompt: $LR_PROMPT, Adapter: $LR_ADAPTER)"
echo "   - Scheduler:  $SCHEDULER (Warmup: $WARMUP_EPOCHS epochs)"
echo "   - Segments:   $NUM_SEGS"
echo "============================================================================"

python -u main.py \
  --mode train \
  --exper-name CAER-Video-CMAF \
  --dataset CAER \
  --gpu 0 \
  --workers "${WORKERS}" \
  --seed 42 \
  --root-dir "${DATA_DIR}" \
  --train-annotation "${TRAIN_TXT}" \
  --val-annotation "${VAL_TXT}" \
  --test-annotation "${TEST_TXT}" \
  --clip-path ViT-B/16 \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  --epochs 20 \
  --batch-size "${BATCH_SIZE}" \
  --print-freq 20 \
  --use-amp \
  --grad-clip 1.0 \
  --optimizer AdamW \
  --lr "${LR}" \
  --lr-image-encoder "${LR_IMAGE}" \
  --lr-prompt-learner "${LR_PROMPT}" \
  --lr-adapter "${LR_ADAPTER}" \
  --scheduler "${SCHEDULER}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  --weight-decay 0.005 \
  --momentum 0.9 \
  --milestones 10 15 \
  --gamma 0.1 \
  "${LOSS_ARGS[@]}" \
  --mixup-alpha 0.0 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --mi-ramp-type ramp_up \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --text-type prompt_ensemble \
  --temporal-layers 1 \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments "${NUM_SEGS}" \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body \
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --fusion-type cmaf \
  --fusion-stage frame \
  --use-context \
  --mask-context-face \
  "${EXTRA_ARGS[@]}"
