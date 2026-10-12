#!/bin/bash
# ============================================================================
# DAiSEE Training Script — Engagement Recognition (Video) + CMAF
# ============================================================================
# DAiSEE dataset:
# - 4 engagement levels (0: Very Low, 1: Low, 2: High, 3: Very High)
# - Webcam video clips (~300 frames, sampled 8 segments)
# - Differentiated Multi-Stream Preprocessing:
#     * Face stream: Center 50% crop focusing on facial micro-cues and eye gaze
#     * Body stream: Upper 70% posture crop focusing on shoulders, slouching, head tilt
#     * Context stream: Full webcam view
# - Fusion: CMAF (Cross-Modal Attention Fusion) -> Unified Temporal Transformer
# - Loss: LDAM Loss with s=1.0 / 2.0 (avoiding double-scaling trap with CLIP temp 0.07)
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --mode train \
  --exper-name DAiSEE-CMAF \
  --dataset DAiSEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 8 \
  --print-freq 50 \
  --seed 42 \
  --workers 2 \
  --root-dir DAiSEE \
  --train-annotation DAiSEE/daisee_train.txt \
  --val-annotation DAiSEE/daisee_val.txt \
  --test-annotation DAiSEE/daisee_test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face DAiSEE/face_bbox_dummy.json \
  --bounding-box-body DAiSEE/body_bbox_dummy.json \
  \
  --fusion-type cmaf \
  --temporal-layers 1 \
  --fusion-stage frame \
  --daisee-merge-low \
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
  --label-smoothing 0.05 \
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
