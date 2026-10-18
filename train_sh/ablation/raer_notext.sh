#!/bin/bash
# ============================================================================
# RAER Ablation: VISUAL ONLY (No Text Alignment / Standard Classifier)
# ============================================================================
# Multimodal Alignment Ablation:
#   Ablates CLIP text embeddings and language-guided alignment.
#   Uses --ablation-no-text: video features are directly projected into a standard
#   linear classifier head rather than computing cosine similarities with prompt vectors.
#   Disables MI and DC losses as text representations are absent.
#   Tests the fundamental benefit of Vision-Language alignment vs Vision-Only learning.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Auto-detect Bounding Box paths (Local vs Kaggle)
FACE_BOX="RAER/bounding_box/face.json"
[ ! -f "$FACE_BOX" ] && FACE_BOX="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/face.json"

BODY_BOX="RAER/bounding_box/body.json"
[ ! -f "$BODY_BOX" ] && BODY_BOX="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/body.json"

python main.py \
  --mode train \
  --exper-name Ablation-RAER-Visual-Only \
  --ablation-no-text \
  --dataset RAER \
  --gpu 0 \
  --workers 2 \
  --seed 42 \
  --root-dir /kaggle/input/datasets/bearmn/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face "$FACE_BOX" \
  --bounding-box-body "$BODY_BOX" \
  --epochs 20 \
  --early-stopping-patience 3 \
  --batch-size 4 \
  --print-freq 50 \
  --use-amp \
  --grad-clip 1.0 \
  --optimizer AdamW \
  --lr 2e-05 \
  --lr-image-encoder 1e-06 \
  --lr-adapter 0.0001 \
  --weight-decay 0.005 \
  --momentum 0.9 \
  --milestones 10 15 \
  --gamma 0.1 \
  --loss-type ldam \
  --lambda_mi 0.0 \
  --lambda_dc 0.0 \
  --use-weighted-sampler \
  --label-smoothing 0.05 \
  --ldam-max-m 0.5 \
  --ldam-s 30.0 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --crop-body \
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --fusion-type cmaf \
  --use-context
