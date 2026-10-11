#!/bin/bash
# ============================================================================
# RAER Training Script — FrameLevelFusion
# ============================================================================
# Auto-generated to match the exact configuration from the provided log.
# Run this on Kaggle to resume or retrain the exact experiment.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python main.py \
  --mode train \
  --exper-name RAER-FrameLevelFusion \
  --dataset RAER \
  --gpu 0 \
  --workers 2 \
  --seed 42 \
  --root-dir /kaggle/input/datasets/bearmn/raer-video-emotion-dataset \
  --train-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt \
  --val-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --test-annotation /kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt \
  --clip-path ViT-B/16 \
  --bounding-box-face RAER/bounding_box/face.json \
  --bounding-box-body RAER/bounding_box/body.json \
  --epochs 20 \
  --batch-size 2 \
  --print-freq 50 \
  --use-amp \
  --grad-clip 1.0 \
  --optimizer AdamW \
  --lr 2e-05 \
  --lr-image-encoder 1e-06 \
  --lr-prompt-learner 0.0003 \
  --lr-adapter 0.0001 \
  --weight-decay 0.005 \
  --momentum 0.9 \
  --milestones 10 15 \
  --gamma 0.1 \
  --loss-type ldam \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --mi-ramp-type ramp_up \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --use-weighted-sampler \
  --label-smoothing 0.05 \
  --ldam-max-m 0.5 \
  --ldam-s 30.0 \
  --text-type prompt_ensemble \
  --temporal-layers 1 \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body \
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --fusion-type cmaf \
  --use-context \

