#!/bin/bash
# ============================================================================
# RAPT-CLIP: Master Runner cho Toàn Bộ Thí Nghiệm Ablation Cần Train Từ Đầu
# (Training Objective Ablations trên tập dữ liệu RAER)
# ============================================================================
# Danh mục các thử nghiệm bắt buộc phải train lại (chi phối gradient 20 epochs):
#   1. no_prompt_tuning    : Cắt Prompt Tuning (Fixed hand-crafted prompts, lambda_mi=0, lambda_dc=0)
#   2. no_midc             : Cắt hàm phạt Regularization (lambda_mi=0, lambda_dc=0, vẫn giữ learnable prompt)
#   3. no_ldam             : Cắt LDAM loss (dùng standard Cross-Entropy loss)
#   4. no_sampler          : Cắt WeightedRandomSampler (dùng uniform random sampling)
#   5. freeze_encoder      : Đóng băng Visual Backbone ViT-B/16 (lr_image_encoder=0)
#   6. no_modality_dropout : Cắt Modality Dropout (p=0.0 thay vì 0.3)
#   7. no_drop_path        : Cắt Stochastic Depth DropPath (p=0.0 thay vì 0.1)
#   8. all                 : Chạy tuần tự tất cả các thí nghiệm trên
#
# Cách sử dụng:
#   bash train_sh/ablation/run_training_ablations.sh no_prompt_tuning
#   bash train_sh/ablation/run_training_ablations.sh all
#   bash train_sh/ablation/run_training_ablations.sh no_midc --resume path/to/ckpt.pth
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

TARGET="${1:-help}"
shift || true
EXTRA_ARGS=("$@")

print_banner() {
    echo "============================================================================"
    echo "  RAPT-CLIP RAER TRAINING ABLATION RUNNER"
    echo "  Thời gian: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Mục tiêu:  $1"
    echo "============================================================================"
}

run_task() {
    local task_name="$1"
    local script_file="$2"
    local desc="$3"

    print_banner "${task_name} — ${desc}"
    
    if [ ! -f "${script_file}" ]; then
        echo "❌ LỖI: Không tìm thấy file script: ${script_file}"
        exit 1
    fi

    chmod +x "${script_file}"
    START_TIME=$(date +%s)
    
    echo "🚀 Bắt đầu thực thi: bash ${script_file} ${EXTRA_ARGS[*]}"
    bash "${script_file}" "${EXTRA_ARGS[@]}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "✅ HOÀN THÀNH: ${task_name} trong $((ELAPSED / 60)) phút $((ELAPSED % 60)) giây."
    echo "----------------------------------------------------------------------------"
}

case "${TARGET}" in
    no_prompt_tuning)
        run_task "No-Prompt-Tuning" "${SCRIPT_DIR}/raer_no_prompt_tuning.sh" "Fixed Text Prompts (Tắt CoOp, lr_prompt=0, lambda_mi/dc=0)"
        ;;
    no_midc)
        run_task "No-MI-DC" "${SCRIPT_DIR}/raer_no_midc.sh" "Tắt Mutual Information & Decorrelation Losses"
        ;;
    no_ldam)
        run_task "No-LDAM" "${SCRIPT_DIR}/raer_no_ldam.sh" "Dùng Cross-Entropy chuẩn thay thế LDAM"
        ;;
    no_sampler)
        run_task "No-Sampler" "${SCRIPT_DIR}/raer_no_sampler.sh" "Dùng Uniform Random Shuffling thay thế Weighted Sampler"
        ;;
    freeze_encoder)
        run_task "Freeze-Image-Encoder" "${SCRIPT_DIR}/raer_freeze_encoder.sh" "Khóa cứng toàn bộ CLIP ViT-B/16 visual encoder"
        ;;
    no_modality_dropout)
        run_task "No-Modality-Dropout" "${SCRIPT_DIR}/raer_no_modality_dropout.sh" "Modality Dropout = 0.0 (Tắt che ngẫu nhiên các luồng)"
        ;;
    no_drop_path)
        run_task "No-DropPath" "${SCRIPT_DIR}/raer_drop_path.sh" "Drop Path Rate = 0.0 (Tắt Stochastic Depth)"
        ;;
    all)
        echo "🔥 BẮT ĐẦU CHẠY TOÀN BỘ 6 THÍ NGHIỆM TRAINING OBJECTIVE ABLATION CHO RAER..."
        run_task "1/6: No-Prompt-Tuning" "${SCRIPT_DIR}/raer_no_prompt_tuning.sh" "Fixed Text Prompts"
        run_task "2/6: No-MI-DC" "${SCRIPT_DIR}/raer_no_midc.sh" "Tắt MI & DC Regularizer"
        run_task "3/6: No-LDAM" "${SCRIPT_DIR}/raer_no_ldam.sh" "Thay thế bằng Standard Cross-Entropy"
        run_task "4/6: No-Sampler" "${SCRIPT_DIR}/raer_no_sampler.sh" "Thay thế bằng Uniform Sampler"
        run_task "5/6: Freeze-Image-Encoder" "${SCRIPT_DIR}/raer_freeze_encoder.sh" "Khóa Visual Backbone"
        run_task "6/6: No-Modality-Dropout" "${SCRIPT_DIR}/raer_no_modality_dropout.sh" "Modality Dropout = 0.0"
        echo "🎉 TẤT CẢ CÁC THÍ NGHIỆM TRAINING ABLATION ĐÃ HOÀN TẤT THÀNH CÔNG!"
        ;;
    *)
        echo "============================================================================"
        echo "  HƯỚNG DẪN SỬ DỤNG: run_training_ablations.sh"
        echo "============================================================================"
        echo "Cú pháp:"
        echo "  bash train_sh/ablation/run_training_ablations.sh <TARGET> [EXTRA_ARGS...]"
        echo ""
        echo "Danh sách <TARGET> khả dụng:"
        echo "  1. no_prompt_tuning    : Cắt Prompt Tuning (Text prompt cố định, không học context)"
        echo "  2. no_midc             : Cắt MI & DC loss (Vẫn giữ learnable prompt)"
        echo "  3. no_ldam             : Cắt LDAM margin loss (Dùng Standard Cross-Entropy)"
        echo "  4. no_sampler          : Cắt WeightedRandomSampler (Dùng Uniform Sampler)"
        echo "  5. freeze_encoder      : Khóa cứng CLIP Image Backbone (lr = 0)"
        echo "  6. no_modality_dropout : Cắt Modality Dropout (p = 0.0)"
        echo "  7. no_drop_path        : Cắt Stochastic Depth DropPath (rate = 0.0)"
        echo "  8. all                 : Chạy tuần tự cả 6 thí nghiệm trên từ 1 đến 6"
        echo ""
        echo "Ví dụ:"
        echo "  # Chạy riêng thử nghiệm cắt Prompt Tuning:"
        echo "  bash train_sh/ablation/run_training_ablations.sh no_prompt_tuning"
        echo ""
        echo "  # Chạy thử nghiệm cắt MI/DC kèm cờ resume:"
        echo "  bash train_sh/ablation/run_training_ablations.sh no_midc --resume outputs/Ablation-RAER-NoMIDC/model_best.pth"
        echo ""
        echo "  # Chạy toàn bộ 6 thử nghiệm lần lượt:"
        echo "  bash train_sh/ablation/run_training_ablations.sh all"
        echo "============================================================================"
        exit 0
        ;;
esac
