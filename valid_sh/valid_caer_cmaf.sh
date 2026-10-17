#!/bin/bash
# ============================================================================
# CAER Video Evaluation Script — Context-Aware Emotion Recognition (CMAF)
# ============================================================================
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

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

# 0. Ưu tiên trực tiếp đường dẫn Google Drive CAER Video nếu tồn tại
COLAB_TARGET_DIR="/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
if [ -d "$COLAB_TARGET_DIR" ]; then
  DATA_DIR="${DATA_DIR:-$COLAB_TARGET_DIR}"
  ANNOTATION_DIR="${ANNOTATION_DIR:-$COLAB_TARGET_DIR}"
fi

# 1. Tự động nhận diện thư mục dataset trên Colab / Kaggle / Local
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
  "/Users/macbook/Downloads/RAPT-CLIP/CAER_Video"
  "/Users/macbook/Downloads/RAPT-CLIP/CAER"
)

DATA_DIR="$(normalize_path "${DATA_DIR:-}")"
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

echo "=> [1/4] Thư mục CAER Video root: ${DATA_DIR}"

# 2. Nhận diện Thư mục Annotations
POSSIBLE_ANNOT_DIRS=(
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video"
  "${DATA_DIR}"
  "annotations/caer_video"
  "/kaggle/input/datasets/traincollab/caer-video-annotation"
  "/kaggle/input/caer-video-annotation"
)

ANNOTATION_DIR="$(normalize_path "${ANNOTATION_DIR:-}")"
if [ -z "$ANNOTATION_DIR" ]; then
  for d in "${POSSIBLE_ANNOT_DIRS[@]}"; do
    if [ -d "$d" ] && { [ -f "$d/test.txt" ] || [ -f "$d/train.txt" ]; }; then
      ANNOTATION_DIR="$d"
      break
    fi
  done
fi

if [ -z "$ANNOTATION_DIR" ]; then
  ANNOTATION_DIR="${DATA_DIR}"
fi

echo "=> [2/4] Thư mục Annotation root: ${ANNOTATION_DIR}"

# 3. File test.txt
TEST_TXT="$(normalize_path "${TEST_TXT:-}")"
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

echo "=> [3/4] Test annotation: ${TEST_TXT}"

# 4. Bounding Boxes
CANDIDATE_FACE_BOXES=(
  "${ANNOTATION_DIR}/face_boxes_mediapipe.json"
  "${ANNOTATION_DIR}/face_boxes.json"
  "${ANNOTATION_DIR}/face_boxes_optimized.json"
  "${ANNOTATION_DIR}/face.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/face_boxes_mediapipe.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/CAER_Video/face_boxes.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/face_boxes_mediapipe.json"
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/face_boxes.json"
  "${DATA_DIR}/face_boxes_mediapipe.json"
  "${DATA_DIR}/face_boxes.json"
  "annotations/caer_video/face_boxes.json"
)

FACE_BOX="$(normalize_path "${FACE_BOX:-}")"
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
  "/content/drive/MyDrive/Nghiên cứu/0.Dataset and Source code/CAER_Video/body_boxes.json"
  "${DATA_DIR}/body_boxes.json"
  "annotations/caer_video/body_boxes.json"
)

BODY_BOX="$(normalize_path "${BODY_BOX:-}")"
if [ -z "$BODY_BOX" ] || [ ! -f "$BODY_BOX" ]; then
  for bb in "${CANDIDATE_BODY_BOXES[@]}"; do
    if [ -f "$bb" ]; then
      BODY_BOX="$bb"
      break
    fi
  done
fi

# 5. Checkpoint
CHECKPOINT="${1:-${CHECKPOINT:-}}"
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  for cp in "outputs/CAER-Video-CMAF/model_best.pth" \
           $(find outputs -type f -name "model_best.pth" 2>/dev/null | grep -i "caer" | head -n 1) \
           "/content/outputs/CAER-Video-CMAF/model_best.pth" \
           "/kaggle/working/outputs/CAER-Video-CMAF/model_best.pth"; do
    if [ -n "$cp" ] && [ -f "$cp" ]; then
      CHECKPOINT="$cp"
      break
    fi
  done
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "=> [Error] Không tìm thấy checkpoint để đánh giá! Vui lòng truyền đường dẫn: bash valid_sh/valid_caer_cmaf.sh <path_to_model_best.pth>"
  exit 1
fi

echo "=> [4/4] Đang đánh giá Checkpoint: ${CHECKPOINT}"
echo "============================================================================"

python -u main.py \
  --mode eval \
  --exper-name Eval-CAER-Video-CMAF \
  --dataset CAER \
  --gpu 0 \
  --batch-size 8 \
  --workers 2 \
  --root-dir "${DATA_DIR}" \
  --test-annotation "${TEST_TXT}" \
  --clip-path ViT-B/16 \
  --bounding-box-face "${FACE_BOX}" \
  --bounding-box-body "${BODY_BOX}" \
  --eval-checkpoint "${CHECKPOINT}" \
  --fusion-type cmaf \
  --fusion-stage frame \
  --temporal-layers 1 \
  --use-context \
  --crop-body \
  --mask-context-face \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --use-amp
