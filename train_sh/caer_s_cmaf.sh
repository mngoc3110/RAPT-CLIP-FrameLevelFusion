#!/bin/bash
# ============================================================================
# CAER-S Training Script — Context-Aware Emotion Recognition (Static) + CMAF
# ============================================================================
# CAER-S dataset:
# - 7 emotion classes (0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise)
# - Static images (num_segments=1, duration=1, no-temporal)
# - Triple-stream architecture: Face (from caer_s_faces.json), Body (torso estimation), Context (full scene)
# - Fusion: Tridirectional CMAF (Cross-Modal Attention Fusion)
# - Loss: LDAM Loss (s=30.0, max_m=0.5)
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Auto-detect or specify Kaggle Dataset directory
DATA_DIR="/kaggle/input/caer-s-dataset-rapt-clip"
if [ ! -d "$DATA_DIR" ]; then
  # Fallback search if mounted under a custom slug/username path on Kaggle
  DETECTED_DIR=$(find /kaggle/input -maxdepth 3 -type f -name "train.txt" 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  fi
fi
echo "=> Using CAER-S dataset path: $DATA_DIR"

python main.py \
  --mode train \
  --exper-name CAER-CMAF \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --print-freq 50 \
  --seed 42 \
  --workers 4 \
  --root-dir "${DATA_DIR}" \
  --train-annotation "${DATA_DIR}/train.txt" \
  --val-annotation "${DATA_DIR}/validation.txt" \
  --test-annotation "${DATA_DIR}/test.txt" \
  --clip-path ViT-B/16 \
  --bounding-box-face "${DATA_DIR}/face.json" \
  --bounding-box-body "${DATA_DIR}/body.json" \
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
  --ldam-s 30.0 \
  --ldam-max-m 0.5 \
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
  --num-segments 8 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --drop-path-rate 0.1 \
  --use-amp
