#!/bin/bash
# ============================================================================
# CAER-S Fine-Tuning & TTA Script (Warm-Start from Pretrained Checkpoint)
# ============================================================================
# Verified SOTA Milestone (Epoch 7):
#   • Train WAR: 96.55% | Train UAR: 96.55% (Loss: 0.5419)
#   • Valid WAR: 90.40% | Valid UAR: 90.40% (Loss: 0.6787)
#   • Per-class Recall: Fear (99.7%), Disgust (98.1%), Sad (93.4%), Anger (92.3%),
#                       Surprise (92.2%), Happy (83.4%), Neutral (73.7%)
# ============================================================================
# Mục đích:
# - Cứu và nâng cấp checkpoint cũ (từ 82.70% -> 90.40%+)
# - Chuyển sang Cross-Entropy (CE + Label Smoothing 0.1) để mở khóa biên độ gradient
# - Bỏ qua Temporal Transformer (--no-temporal) để tập trung 100% vào Frame CMAF
# - Chạy 5-8 epoch với Cosine Annealing (Warmup 1 epoch)
# - Tự động kích hoạt TTA (Test-Time Augmentation) tại bước đánh giá cuối cùng
# ============================================================================
# Cách chạy:
#   bash train_sh/caer_s_finetune.sh                                # Tự động tìm checkpoint, fine-tune 8 epochs
#   bash train_sh/caer_s_finetune.sh /path/to/model_best.pth        # Fine-tune checkpoint chỉ định 8 epochs
#   bash train_sh/caer_s_finetune.sh /path/to/model_best.pth 10     # Fine-tune checkpoint chỉ định 10 epochs
# ============================================================================

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 1. Tự động nhận diện thư mục dataset trên Kaggle / Colab
POSSIBLE_DIRS=(
  "/kaggle/input/datasets/lcngtr/caer-s"
  "/kaggle/input/caer-s"
  "/kaggle/input/datasets/bearmn/caers-dataset-rapt-clip"
  "/content/CAER-S"
  "/content/drive/MyDrive/CAER-S"
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

echo "============================================================================"
echo "=> [1/4] Thư mục CAER-S dataset root: ${DATA_DIR}"

# 2. Tự động chuẩn bị annotation và bounding box
ANNOTATION_DIR="/kaggle/working/caers_data"
if [ "$REGEN" = "1" ] || [ ! -f "${ANNOTATION_DIR}/train.txt" ] || [ ! -f "${ANNOTATION_DIR}/face.json" ] || [ ! -f "${ANNOTATION_DIR}/validation.txt" ]; then
  echo "=> [Preprocessor] Đang tự động kiểm tra/chuyển đổi annotation..."
  python tools/prepare_caers_dataset.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${ANNOTATION_DIR}"
fi

TRAIN_TXT="${ANNOTATION_DIR}/train.txt"
VAL_TXT="${ANNOTATION_DIR}/validation.txt"
TEST_TXT="${ANNOTATION_DIR}/test.txt"

if [ ! -f "${VAL_TXT}" ] && [ -f "${TEST_TXT}" ]; then
  VAL_TXT="${TEST_TXT}"
fi

FACE_BOX="${ANNOTATION_DIR}/face.json"
BODY_BOX="${ANNOTATION_DIR}/body.json"

echo "=> [2/4] File Danh Sách Dữ Liệu:"
echo "   - Train: $TRAIN_TXT"
echo "   - Val:   $VAL_TXT"
echo "   - Test:  $TEST_TXT"
echo "   - Face:  $FACE_BOX"
echo "   - Body:  $BODY_BOX"

# 3. Xử lý Checkpoint Fine-tune & Số Epochs
TOTAL_EPOCHS=8
FINETUNE_CKPT=""

if [ -n "$1" ]; then
  if [ -f "$1" ]; then
    FINETUNE_CKPT="$1"
    if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
      TOTAL_EPOCHS="$2"
    fi
  elif [[ "$1" =~ ^[0-9]+$ ]]; then
    TOTAL_EPOCHS="$1"
  fi
fi

if [ -z "$FINETUNE_CKPT" ]; then
  # Ưu tiên trực tiếp đường dẫn checkpoint người dùng cung cấp
  KNOWN_CKPTS=(
    "/kaggle/input/models/bearmn/caer-finetune-end/tensorflow2/default/1/Caers-finetune.pth"
    "/kaggle/input/caer-finetune-end/tensorflow2/default/1/Caers-finetune.pth"
    "/kaggle/input/caer-finetune-end/Caers-finetune.pth"
    "/kaggle/input/models/bearmn/caerv2-finetune/tensorflow2/default/1/model_best.pth"
    "/kaggle/input/models/bearmn/caerv2-finetune/tensorflow2/default/1/model.pth"
    "/kaggle/input/models/traincollab/caerv2-funing/tensorflow2/default/1/model_best.pth"
    "/kaggle/input/models/traincollab/caerv2-funing/tensorflow2/default/1/model.pth"
  )
  for ckpt in "${KNOWN_CKPTS[@]}"; do
    if [ -f "$ckpt" ]; then
      FINETUNE_CKPT="$ckpt"
      echo "=> [Auto-Detect] Tìm thấy checkpoint mục tiêu: ${FINETUNE_CKPT}"
      break
    fi
  done
fi

if [ -z "$FINETUNE_CKPT" ]; then
  # Quét tìm trong outputs/ hiện tại
  SEARCH_OUTPUT_DIRS=("/kaggle/working/outputs" "outputs")
  for out_d in "${SEARCH_OUTPUT_DIRS[@]}"; do
    if [ -d "$out_d" ]; then
      FOUND_CKPT=$(find "$out_d" -type f -name "model_best.pth" 2>/dev/null | sort -V | tail -n 1)
      if [ -z "$FOUND_CKPT" ]; then
        FOUND_CKPT=$(find "$out_d" -type f -name "model.pth" 2>/dev/null | sort -V | tail -n 1)
      fi
      if [ -n "$FOUND_CKPT" ] && [ -f "$FOUND_CKPT" ]; then
        FINETUNE_CKPT="$FOUND_CKPT"
        break
      fi
    fi
  done
fi

if [ -z "$FINETUNE_CKPT" ] && [ -d "/kaggle/input" ]; then
  # Quét tìm trên các dataset checkpoint người dùng import vào Kaggle
  FINETUNE_CKPT=$(find /kaggle/input -type f \( -iname "*caer*.pth" -o -iname "*model*.pth" -o -iname "*best*.pth" \) 2>/dev/null | grep -iE "caer|model|best" | head -n 1)
fi

echo "=> [3/4] Cấu hình Fine-tuning:"
if [ -n "$FINETUNE_CKPT" ] && [ -f "$FINETUNE_CKPT" ]; then
  echo "   - Checkpoint nguồn (Warm-Start): ${FINETUNE_CKPT}"
  FINETUNE_ARG=("--finetune-from" "${FINETUNE_CKPT}")
else
  echo "   ⚠️ [Cảnh báo] Không tìm thấy checkpoint cũ! Sẽ khởi tạo từ ViT-B/16 chuẩn."
  FINETUNE_ARG=()
fi
echo "   - Số epochs fine-tune:  ${TOTAL_EPOCHS}"
echo "   - TTA (Test-Time Aug):  BẬT (Horizontal Flip on all 3 streams)"
echo "   - Hàm Loss:             Cross-Entropy (Label Smoothing = 0.1)"
echo "   - Tốc độ học (LR):      Image Encoder = 5e-6 | Main = 3e-5 | Prompt = 2e-4"
echo "============================================================================"

# 4. Khởi chạy Fine-tuning RAPT-CLIP CAER-S + TTA
python -u main.py \
  --mode train \
  --exper-name CAER-S-Finetune-TTA \
  --dataset CAER-S \
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
  "${FINETUNE_ARG[@]}" \
  \
  --fusion-type cmaf \
  --fusion-stage frame \
  --no-temporal \
  --use-context \
  --crop-body \
  --mask-context-face \
  \
  --optimizer AdamW \
  --lr 3e-5 \
  --lr-image-encoder 5e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.005 \
  --scheduler cosine \
  --warmup-epochs 1 \
  \
  --loss-type ce \
  --label-smoothing 0.1 \
  --mixup-alpha 0.0 \
  --lambda_mi 0.05 \
  --lambda_dc 0.05 \
  --mi-warmup 1 \
  --mi-ramp 3 \
  --dc-warmup 1 \
  --dc-ramp 3 \
  --grad-clip 1.0 \
  --early-stopping-patience 3 \
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
  --modality-dropout 0.1 \
  --use-tta \
  --use-amp
