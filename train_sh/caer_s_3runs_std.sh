#!/bin/bash
# ==============================================================================
# Script tự động chạy 3 lần độc lập (3 Seeds) cho CAER-S để tính Mean & Std
# Phù hợp chạy qua đêm trên GPU (Kaggle / Colab / Server)
# ==============================================================================

SEEDS=(42 123 999)
RESULTS_LOG="outputs/CAER-S/multi_run_results.txt"
mkdir -p outputs/CAER-S

echo "==================================================" > "${RESULTS_LOG}"
echo "  CAER-S 3-RUN STATISTICAL VALIDATION (SEEDS: 42, 123, 999)" >> "${RESULTS_LOG}"
echo "==================================================" >> "${RESULTS_LOG}"

for SEED in "${SEEDS[@]}"; do
  echo "" | tee -a "${RESULTS_LOG}"
  echo ">>> [STARTING RUN WITH SEED: ${SEED}] <<<" | tee -a "${RESULTS_LOG}"
  
  RUN_EXP="CAER-S-Seed-${SEED}"
  
  python -u main.py \
    --mode train \
    --exper-name "${RUN_EXP}" \
    --dataset CAER \
    --gpu 0 \
    --seed "${SEED}" \
    --epochs 8 \
    --batch-size 32 \
    --workers 4 \
    --lr 3e-5 \
    --clip-path ViT-B/16 \
    --fusion-type cmaf \
    --fusion-stage frame \
    --use-context \
    --crop-body \
    --mask-context-face \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --use-amp 2>&1 | tee "outputs/CAER-S/${RUN_EXP}.log"

  echo ">>> [FINISHED RUN WITH SEED: ${SEED}] <<<" | tee -a "${RESULTS_LOG}"
done

echo "" | tee -a "${RESULTS_LOG}"
echo "=> Toàn bộ 3 runs đã hoàn tất. Sử dụng tools/plot_caers_accuracy_cm.py để tổng kết Mean +- Std." | tee -a "${RESULTS_LOG}"
