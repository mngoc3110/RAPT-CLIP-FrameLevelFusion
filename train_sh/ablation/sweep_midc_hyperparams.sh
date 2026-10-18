#!/bin/bash
# ============================================================================
# RAPT-CLIP: Hyperparameter Sensitivity Sweep cho Lambda MI/DC & Scheduling
# ============================================================================
# Mục đích:
#   Kiểm chứng thực nghiệm trên RAER để tìm ra bộ siêu tham số tối ưu:
#   1. Quét biên độ: lambda = 0.01, 0.05, 0.1 (chuẩn), 0.2
#   2. Quét cơ chế: Constant vs Ramp-Up vs Ramp-Down
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-help}"

run_sweep_trial() {
    local exp_name="$1"
    local l_mi="$2"
    local l_dc="$3"
    local warmup="$4"
    local ramp="$5"
    local ramp_type="$6"

    echo "============================================================================"
    echo "  BẮT ĐẦU SWEEP TRIAL: ${exp_name}"
    echo "  lambda_mi=${l_mi}, lambda_dc=${l_dc}, warmup=${warmup}, ramp=${ramp}, type=${ramp_type}"
    echo "============================================================================"

    python main.py \
      --mode train \
      --exper-name "Sweep-${exp_name}" \
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
  --early-stopping-patience 3 \
      --batch-size 4 \
      --print-freq 50 \
      --use-amp \
      --grad-clip 1.0 \
      --optimizer AdamW \
      --lr 2e-05 \
      --lr-image-encoder 1e-06 \
      --lr-prompt-learner 0.0003 \
      --lr-adapter 0.0001 \
      --weight-decay 0.005 \
      --loss-type ldam \
      --lambda_mi "${l_mi}" \
      --lambda_dc "${l_dc}" \
      --mi-warmup "${warmup}" \
      --mi-ramp "${ramp}" \
      --mi-ramp-type "${ramp_type}" \
      --dc-warmup "${warmup}" \
      --dc-ramp "${ramp}" \
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
      --use-context
}

case "${MODE}" in
    lambda_sweep)
        echo "🔥 Chạy sweep kiểm chứng các mức lambda (0.01, 0.05, 0.1, 0.2)..."
        run_sweep_trial "lambda-0.01" 0.01 0.01 5 10 "ramp_up"
        run_sweep_trial "lambda-0.05" 0.05 0.05 5 10 "ramp_up"
        run_sweep_trial "lambda-0.10" 0.10 0.10 5 10 "ramp_up"
        run_sweep_trial "lambda-0.20" 0.20 0.20 5 10 "ramp_up"
        ;;
    schedule_sweep)
        echo "🔥 Chạy sweep so sánh Constant vs Ramp-Down vs Ramp-Up..."
        run_sweep_trial "sched-constant" 0.10 0.10 0 0 "ramp_up"
        run_sweep_trial "sched-rampdown" 0.10 0.10 0 15 "ramp_down"
        run_sweep_trial "sched-rampup"   0.10 0.10 5 10 "ramp_up"
        ;;
    *)
        echo "Cách dùng:"
        echo "  bash train_sh/ablation/sweep_midc_hyperparams.sh lambda_sweep    # Quét độ lớn lambda"
        echo "  bash train_sh/ablation/sweep_midc_hyperparams.sh schedule_sweep  # Quét Constant vs Ramp-Up vs Ramp-Down"
        exit 0
        ;;
esac
