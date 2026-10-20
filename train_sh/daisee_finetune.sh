#!/bin/bash
# ============================================================================
# DAiSEE Fine-Tuning Script — Warm-Start from model_daisee.zip
# ============================================================================
# Mục đích:
#   1. Khắc phục sụp đổ Recall lớp thiểu số (Low Engagement từ 2.27% -> 50%+)
#   2. Nâng UAR từ 40.11% lên 65% - 70%+
#   3. Áp dụng LDAM Loss (s=1.0, max_m=0.5) + Class Weighting để cân bằng gradient
#   4. Tự động sinh hoặc nạp Face Bounding Boxes chuẩn từ YuNet
# ============================================================================

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OPENCV_LOG_LEVEL=OFF
export OPENCV_FFMPEG_LOGLEVEL=0

CHECKPOINT_PATH=""
EPOCHS=8
GEN_BBOX="${GEN_BBOX:-0}"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --gen-bbox)
      GEN_BBOX=1
      ;;
    --epochs=*)
      EPOCHS="${arg#*=}"
      ;;
    --checkpoint=*)
      CHECKPOINT_PATH="${arg#*=}"
      ;;
    --*)
      EXTRA_ARGS+=("$arg")
      ;;
    *)
      if [ -z "$CHECKPOINT_PATH" ] && ([ -f "$arg" ] || [[ "$arg" == *.pth ]] || [[ "$arg" == *.zip ]]); then
        CHECKPOINT_PATH="$arg"
      elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        EPOCHS="$arg"
      elif [ -z "$CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH="$arg"
      else
        EXTRA_ARGS+=("$arg")
      fi
      ;;
  esac
done

# Tự động tìm checkpoint nếu chưa có
if [ -z "$CHECKPOINT_PATH" ]; then
  KNOWN_CKPTS=(
    "/kaggle/input/models/bearmn/daisy-model/tensorflow2/default/1/model_daisee.pth"
    "model_daisee.zip"
    "model_clean.pth"
  )
  for ckpt in "${KNOWN_CKPTS[@]}"; do
    if [ -f "$ckpt" ]; then
      CHECKPOINT_PATH="$ckpt"
      break
    fi
  done
  if [ -z "$CHECKPOINT_PATH" ] && [ -d "/kaggle/input" ]; then
    FOUND_C=$(find /kaggle/input -type f \( -name "*daisee*.pth" -o -name "*daisy*.pth" -o -name "*model_daisee*.zip" \) 2>/dev/null | head -n 1)
    if [ -n "$FOUND_C" ]; then
      CHECKPOINT_PATH="$FOUND_C"
    fi
  fi
  if [ -z "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH="model_daisee.zip"
  fi
fi

echo "============================================================================"
echo "🎯 RAPT-CLIP DAiSEE TARGETED FINE-TUNING"
echo "• Base Checkpoint: ${CHECKPOINT_PATH}"
echo "• Target Epochs:   ${EPOCHS}"
echo "============================================================================"

# 1. Nhận diện thư mục dữ liệu DAiSEE
POSSIBLE_DATA_DIRS=(
  "DAiSEE"
  "DAiSEE/DataSet"
  "/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE"
  "/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE/DataSet"
  "/kaggle/input/datasets/olgaparfenova/daisee"
  "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
  "/kaggle/input/datasets/mngochocsupham/daisee"
  "/kaggle/input/daisee/DAiSEE_data"
  "/kaggle/input/daisee/DAiSEE"
  "/kaggle/input/daisee"
  "/kaggle/input/DAiSEE_data"
  "/kaggle/input/DAiSEE"
)

DATA_DIR=""
for d in "${POSSIBLE_DATA_DIRS[@]}"; do
  if [ -f "$d/DataSet/Validation.txt" ] || [ -f "$d/Validation.txt" ] || [ -f "$d/daisee_train.txt" ] || [ -f "$d/train.txt" ] || [ -f "$d/Labels/TrainLabels.csv" ]; then
    DATA_DIR="$d"
    break
  fi
done

if [ -z "$DATA_DIR" ] && [ -d "/kaggle/input" ]; then
  DETECTED_FILE=$(find /kaggle/input -maxdepth 8 \( -iname "Validation.txt" -o -iname "daisee_train.txt" -o -iname "TrainLabels.csv" \) 2>/dev/null | head -n 1)
  if [ -n "$DETECTED_FILE" ]; then
    DIRNAME=$(dirname "$DETECTED_FILE")
    if [ "$(basename "$DIRNAME")" = "DataSet" ] || [ "$(basename "$DIRNAME")" = "Labels" ]; then
      DATA_DIR=$(dirname "$DIRNAME")
    else
      DATA_DIR="$DIRNAME"
    fi
  fi
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="DAiSEE"
fi

echo "=> [1/4] Thư mục DAiSEE: ${DATA_DIR}"

# 2. File danh sách dữ liệu (Train / Val / Test)
TRAIN_TXT=""
for t in "$DATA_DIR/daisee_train.txt" "$DATA_DIR/DataSet/Train.txt" "$DATA_DIR/Train.txt" "$DATA_DIR/Labels/TrainLabels.csv"; do
  if [ -f "$t" ]; then TRAIN_TXT="$t"; break; fi
done

VAL_TXT=""
for v in "$DATA_DIR/daisee_val.txt" "$DATA_DIR/DataSet/Validation.txt" "$DATA_DIR/Validation.txt" "$DATA_DIR/Labels/ValidationLabels.csv"; do
  if [ -f "$v" ]; then VAL_TXT="$v"; break; fi
done

TEST_TXT=""
for s in "$DATA_DIR/daisee_test.txt" "$DATA_DIR/DataSet/Test.txt" "$DATA_DIR/Test.txt" "$DATA_DIR/Labels/TestLabels.csv"; do
  if [ -f "$s" ]; then TEST_TXT="$s"; break; fi
done

echo "=> [2/4] Annotations:"
echo "   • Train: $TRAIN_TXT"
echo "   • Val:   $VAL_TXT"
echo "   • Test:  $TEST_TXT"

# 3. Face Bounding Box (YuNet detector)
FACE_BBOX_PATH="none"
for f in "$DATA_DIR/daisee_face_bbox.json" "$DATA_DIR/face_bbox.json" "/kaggle/working/daisee_face_bbox.json" "DAiSEE/daisee_face_bbox.json"; do
  if [ -f "$f" ] && [ $(wc -c < "$f" 2>/dev/null || echo 0) -gt 500 ]; then
    FACE_BBOX_PATH="$f"
    break
  fi
done

if [ "$FACE_BBOX_PATH" = "none" ] && [ "$GEN_BBOX" = "1" ] && [ -f "tools/generate_daisee_face_boxes.py" ]; then
  echo "=> [YuNet] Đang sinh Face Bounding Box bằng YuNet detector..."
  OUT_BBOX="DAiSEE/daisee_face_bbox.json"
  [ -d "/kaggle/working" ] && OUT_BBOX="/kaggle/working/daisee_face_bbox.json"
  python -u tools/generate_daisee_face_boxes.py \
    --root-dir "$DATA_DIR" \
    --annotations "$TRAIN_TXT" "$VAL_TXT" "$TEST_TXT" \
    --output "$OUT_BBOX" \
    --samples-per-video 4 \
    --device auto
  if [ -f "$OUT_BBOX" ] && [ $(wc -c < "$OUT_BBOX" 2>/dev/null || echo 0) -gt 500 ]; then
    FACE_BBOX_PATH="$OUT_BBOX"
  fi
elif [ "$FACE_BBOX_PATH" = "none" ]; then
  echo "=> [Thông báo] Sử dụng chuẩn Center-Crop Fallback (đồng bộ 100% với checkpoint gốc, bắt đầu train ngay lập tức)."
  echo "   (Nếu muốn sinh lại Face Bounding Box chi tiết bằng YuNet, truyền thêm cờ --gen-bbox)"
fi

echo "=> [3/4] Face Bounding Box: ${FACE_BBOX_PATH}"
echo "=> [4/4] Bắt đầu quá trình Targeted Fine-Tuning với LDAM Loss..."
echo "============================================================================"

# Chọn thiết bị (cuda / mps / cpu)
DEVICE_ARG="0"
python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
  DEVICE_ARG="mps"
fi

python -u main.py \
  --mode train \
  --exper-name DAiSEE-Finetune \
  --dataset DAiSEE \
  --gpu "$DEVICE_ARG" \
  --epochs "$EPOCHS" \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --print-freq 10 \
  --seed 42 \
  --workers 2 \
  --early-stopping-patience 0 \
  --root-dir "$DATA_DIR" \
  --train-annotation "$TRAIN_TXT" \
  --val-annotation "$VAL_TXT" \
  --test-annotation "$TEST_TXT" \
  --clip-path ViT-B/16 \
  --bounding-box-face "$FACE_BBOX_PATH" \
  --bounding-box-body "none" \
  --finetune-from "$CHECKPOINT_PATH" \
  \
  --fusion-type cmaf \
  --temporal-layers 1 \
  --fusion-stage frame \
  --daisee-merge-low \
  \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 5e-7 \
  --lr-prompt-learner 1e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.01 \
  --scheduler cosine \
  --warmup-epochs 1 \
  \
  --loss-type ldam \
  --ldam-s 1.0 \
  --ldam-max-m 0.5 \
  --use-class-weight \
  --label-smoothing 0.05 \
  \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --drop-path-rate 0.1 \
  "${EXTRA_ARGS[@]}"
