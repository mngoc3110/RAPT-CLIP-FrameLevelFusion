#!/bin/bash
# ============================================================================
# RAER Evaluation Script — Ablation No LDAM
# ============================================================================
export CUDA_VISIBLE_DEVICES=0

CKPT="${1:-outputs/Ablation-RAER-NoLDAM/model_best.pth}"

python main.py \
  --mode eval \
  --exper-name Eval-RAER-NoLDAM \
  --dataset RAER \
  --gpu 0 \
  --batch-size 4 \
  --workers 2 \
  --image-size 224 \
  --root-dir /kaggle/input/datasets/bearmn/raer-video-emotion-dataset \
  --test-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
  --text-type prompt_ensemble \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --crop-body \
  --use-context \
  --fusion-type cmaf \
  --eval-checkpoint "$CKPT"
