#!/bin/bash
# ==============================================================================
# RAPT-CLIP EMOTIC Training Script (Preprocessed NumPy Arrays)
# ==============================================================================
# Multi-label Context Emotion Recognition on 26 Discrete Categories (Standard RAER Workflow)
# Evaluated on Macro mAP across all 26 classes.
#
# Key Features:
#   1. Memory-Mapped NumPy DataLoader (instant loading, zero RAM bloat)
#   2. Asymmetric Loss (ASL) calibrated for extreme long-tail class imbalance
#   3. Full unmasked scene context + Tri-stream CMAF
#   4. Regularization: DropPath (0.1), Weight Decay (0.02), Mixup (0.4)
# ==============================================================================

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Configurable data path (local default, can be overridden for Kaggle/Colab)
NPY_DIR=${1:-"emotic_dataset/emotic_pre"}

python main.py \
  --mode train \
  --exper-name EMOTIC-RAPT-CLIP-NPY \
  --dataset EMOTIC \
  --use-npy \
  --npy-dir "$NPY_DIR" \
  --clip-path ViT-B/16 \
  --gpu 0 \
  --workers 2 \
  --seed 42 \
  --epochs 20 \
  --batch-size 8 \
  --print-freq 50 \
  --use-amp \
  --grad-clip 1.0 \
  --optimizer AdamW \
  --lr 2e-05 \
  --lr-image-encoder 0 \
  --freeze-image-encoder \
  --lr-prompt-learner 5e-06 \
  --lr-adapter 2e-05 \
  --weight-decay 0.02 \
  --milestones 10 15 \
  --gamma 0.1 \
  --loss-type asl \
  --asl-gamma-neg 2.0 \
  --asl-gamma-pos 0.0 \
  --asl-clip 0.05 \
  --mixup-alpha 0.4 \
  --text-type prompt_ensemble \
  --no-temporal \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --num-segments 1 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body \
  --drop-path-rate 0.1 \
  --modality-dropout 0.1 \
  --fusion-type cmaf \
  --fusion-stage frame \
  --use-context
