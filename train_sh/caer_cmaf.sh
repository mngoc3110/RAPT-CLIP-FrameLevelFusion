#!/bin/bash
# ============================================================================
# CAER Video Training Script — Context-Aware Emotion Recognition + CMAF
# ============================================================================
# CAER video dataset:
# - 7 emotion classes (0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise)
# - Video clips (.avi, ~60-100 frames, num_segments=8, duration=1)
# - Multi-stream architecture: Face (from face.json), Body (torso estimation), Context (full scene with face masked)
# - Fusion: Tridirectional Frame-Level CMAF -> Temporal Transformer
# - Loss: LDAM Loss (s=30.0, max_m=0.5)
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Exact Kaggle Dataset directory for CAER
DATA_DIR="/kaggle/input/datasets/bearmn/caers-dataset-rapt-clip"
if [ ! -d "$DATA_DIR" ]; then
  # Fallback search if mounted under a custom slug/username path on Kaggle
  DETECTED_DIR=$(find /kaggle/input -maxdepth 4 -type f -name "train.txt" 2>/dev/null | grep -i "caer" | head -n 1 | xargs dirname 2>/dev/null)
  if [ -n "$DETECTED_DIR" ] && [ -d "$DETECTED_DIR" ]; then
    DATA_DIR="$DETECTED_DIR"
  fi
fi
# Preprocess body bounding boxes if dataset body.json is empty ({})
BODY_BOX="${DATA_DIR}/body.json"
if [ ! -f "$BODY_BOX" ] || [ $(wc -c < "$BODY_BOX" 2>/dev/null || echo 0) -lt 100 ]; then
  GENERATED_BODY="/kaggle/working/caer_body_boxes.json"
  if [ ! -f "$GENERATED_BODY" ]; then
    echo "=> [Preprocessing] Dataset body.json is empty ({}) or missing."
    echo "=> Generating high-precision temporally-smoothed Body Bounding Boxes..."
    python tools/generate_caer_body_boxes.py \
      --face-json "${DATA_DIR}/face.json" \
      --output "$GENERATED_BODY"
  fi
  BODY_BOX="$GENERATED_BODY"
fi
echo "=> Using Body bounding box path: $BODY_BOX"

python main.py \
  --mode train \
  --exper-name CAER-CMAF \
  --dataset CAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 2 \
  --print-freq 50 \
  --seed 42 \
  --workers 2 \
  --root-dir "${DATA_DIR}" \
  --train-annotation "${DATA_DIR}/train.txt" \
  --val-annotation "${DATA_DIR}/validation.txt" \
  --test-annotation "${DATA_DIR}/test.txt" \
  --clip-path ViT-B/16 \
  --bounding-box-face "${DATA_DIR}/face.json" \
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
