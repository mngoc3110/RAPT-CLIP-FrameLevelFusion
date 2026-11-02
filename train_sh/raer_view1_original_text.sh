#!/bin/bash
# ==============================================================================
# raer_view1_original_text.sh: Thí nghiệm Đối chứng View 1 dùng Prompt gốc của RAER
# ==============================================================================
# Mục đích khoa học:
#   - Khóa chặt Text View 1 bằng 100% prompt gốc của bài báo ICCV 2025 (CLIP-CAER):
#     "Relaxed mouth, open eyes, neutral eyebrows, ... or natural body posture."
#   - Chứng minh khoảng cách vượt trội so với CLIP-CAER (68.00% UAR) hoàn toàn
#     đến từ các khối thị giác đề xuất (Frame-level CMAF, EAA, Temporal Transformer)
#     và cơ chế canh chỉnh View 2 (MI/DC Loss), KHÔNG PHẢI do Prompt Ensemble.
#
# Cách dùng trên Kaggle / Server:
#   1. Chạy bản đầy đủ (Tự động nhận diện checkpoint /kaggle/input/.../origin_text.pth nếu có để resume):
#      bash train_sh/raer_view1_original_text.sh
#      hoặc:
#      bash train_sh/raer_view1_original_text.sh with_view2
#
#   2. Resume với đường dẫn checkpoint bất kỳ:
#      bash train_sh/raer_view1_original_text.sh with_view2 0 4 42 /path/to/checkpoint.pth
#
#   3. Bỏ qua resume, chạy mới từ Epoch 0:
#      bash train_sh/raer_view1_original_text.sh with_view2 0 4 42 none
#
#   4. Chạy bản bóc tách riêng Khối thị giác (Tắt View 2):
#      bash train_sh/raer_view1_original_text.sh no_view2
# ==============================================================================

VIEW2_MODE=${1:-"with_view2"}
GPU_ID=${2:-0}
BATCH_SIZE=${3:-4}
SEED=${4:-42}
RESUME_INPUT=${5:-""}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ------------------------------------------------------------------------------
# 1. Tự động nhận diện đường dẫn dữ liệu (Kaggle vs Local)
# ------------------------------------------------------------------------------
if [ -d "/kaggle/input/datasets/bearmn/raer-video-emotion-dataset" ]; then
    ROOT_DIR="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset"
    TRAIN_ANNOT="/kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt"
    VAL_ANNOT="/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"
    TEST_ANNOT="/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"
elif [ -d "/kaggle/input/raer-video-emotion-dataset" ]; then
    ROOT_DIR="/kaggle/input/raer-video-emotion-dataset"
    TRAIN_ANNOT="/kaggle/input/raer-annot/annotation/train.txt"
    VAL_ANNOT="/kaggle/input/raer-annot/annotation/test.txt"
    TEST_ANNOT="/kaggle/input/raer-annot/annotation/test.txt"
else
    ROOT_DIR="./"
    TRAIN_ANNOT="RAER/annotation/train.txt"
    VAL_ANNOT="RAER/annotation/test.txt"
    TEST_ANNOT="RAER/annotation/test.txt"
fi

BBOX_FACE="RAER/bounding_box/face.json"
BBOX_BODY="RAER/bounding_box/body.json"

# ------------------------------------------------------------------------------
# 2. Tự động nhận diện và xử lý Checkpoint Resume
# ------------------------------------------------------------------------------
DEFAULT_KAGGLE_RESUME="/kaggle/input/models/bearmn/origin-text/tensorflow2/default/1/origin_text.pth"
ALT_KAGGLE_RESUME="/kaggle/input/origin-text/origin_text.pth"

RESUME_TARGET=""
if [ -n "$RESUME_INPUT" ] && [ "$RESUME_INPUT" != "none" ]; then
    RESUME_TARGET="$RESUME_INPUT"
elif [ -e "$DEFAULT_KAGGLE_RESUME" ]; then
    RESUME_TARGET="$DEFAULT_KAGGLE_RESUME"
elif [ -e "$ALT_KAGGLE_RESUME" ]; then
    RESUME_TARGET="$ALT_KAGGLE_RESUME"
fi

RESUME_PTH=""
if [ -n "$RESUME_TARGET" ] && [ -e "$RESUME_TARGET" ]; then
    echo "=============================================================================="
    echo "🔄 Phát hiện checkpoint Resume: $RESUME_TARGET"
    if [ -d "$RESUME_TARGET" ]; then
        echo "📦 Checkpoint ở dạng thư mục giải nén. Tự động đóng gói lại chuẩn PyTorch..."
        python3 -c "
import os, zipfile
src = '$RESUME_TARGET'
if os.path.isdir(os.path.join(src, 'model')):
    src = os.path.join(src, 'model')
out = '/kaggle/working/origin_text_repack.pth'
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as zipf:
    for root, dirs, files in os.walk(src):
        for f in files:
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, src)
            zipf.write(full_p, arcname=os.path.join('archive', rel_p))
print(f'✅ Đã đóng gói thành công: {out} ({os.path.getsize(out)/(1024**2):.1f} MB)')
"
        RESUME_PTH="/kaggle/working/origin_text_repack.pth"
    else
        echo "📂 Checkpoint là file hợp lệ."
        RESUME_PTH="$RESUME_TARGET"
    fi
    echo "=============================================================================="
fi

# ------------------------------------------------------------------------------
# 2. Cấu hình cờ View 2 (MI Loss / DC Loss)
# ------------------------------------------------------------------------------
if [ "$VIEW2_MODE" == "no_view2" ]; then
    EXP_NAME="RAER_OriginalText_NoView2_Seed${SEED}"
    LAMBDA_MI=0.0
    LAMBDA_DC=0.0
    echo "=============================================================================="
    echo "🔬 Chế độ: BÓC TÁCH RIÊNG KHỐI THỊ GIÁC (Tắt View 2 MI/DC)"
    echo "   • View 1: 100% Text gốc của CLIP-CAER (class_descriptor_5)"
    echo "   • View 2: TẮT (lambda_mi=0.0, lambda_dc=0.0)"
    echo "   • Thị giác: Frame-level CMAF Tri-stream + EAA + Unified Temporal Net"
    echo "=============================================================================="
else
    EXP_NAME="RAER_OriginalText_WithView2_Seed${SEED}"
    LAMBDA_MI=0.1
    LAMBDA_DC=0.1
    echo "=============================================================================="
    echo "🚀 Chế độ: ĐẦY ĐỦ THỊ GIÁC + CANH CHỈNH VIEW 2 (Khuyên dùng)"
    echo "   • View 1: 100% Text gốc của CLIP-CAER (class_descriptor_5)"
    echo "   • View 2: BẬT Mỏ neo ngữ nghĩa (lambda_mi=0.1, lambda_dc=0.1)"
    echo "   • Thị giác: Frame-level CMAF Tri-stream + EAA + Unified Temporal Net"
    echo "=============================================================================="
fi

# ------------------------------------------------------------------------------
# 3. Khởi chạy huấn luyện
# ------------------------------------------------------------------------------
CMD=(
  python main.py
  --mode train
  --exper-name "${EXP_NAME}"
  --dataset RAER
  --gpu 0
  --workers 2
  --seed "${SEED}"
  --root-dir "${ROOT_DIR}"
  --train-annotation "${TRAIN_ANNOT}"
  --val-annotation "${VAL_ANNOT}"
  --test-annotation "${TEST_ANNOT}"
  --clip-path ViT-B/16
  --bounding-box-face "${BBOX_FACE}"
  --bounding-box-body "${BBOX_BODY}"
  --epochs 20
  --early-stopping-patience 3
  --batch-size "${BATCH_SIZE}"
  --print-freq 50
  --use-amp
  --grad-clip 1.0
  --optimizer AdamW
  --lr 2e-05
  --lr-image-encoder 1e-06
  --lr-prompt-learner 0.0003
  --lr-adapter 0.0001
  --weight-decay 0.005
  --momentum 0.9
  --milestones 10 15
  --gamma 0.1
  --loss-type ldam
  --ldam-s 30.0
  --ldam-max-m 0.5
  --mixup-alpha 0.0
  --lambda_mi "${LAMBDA_MI}"
  --lambda_dc "${LAMBDA_DC}"
  --mi-warmup 5
  --mi-ramp 10
  --mi-ramp-type ramp_up
  --dc-warmup 5
  --dc-ramp 10
  --use-weighted-sampler
  --label-smoothing 0.05
  --text-type class_descriptor
  --temporal-layers 1
  --contexts-number 8
  --class-token-position end
  --class-specific-contexts True
  --load_and_tune_prompt_learner True
  --num-segments 16
  --duration 1
  --image-size 224
  --temperature 0.07
  --crop-body
  --drop-path-rate 0.1
  --modality-dropout 0.3
  --fusion-type cmaf
  --use-context
)

if [ -n "$RESUME_PTH" ]; then
    CMD+=(--resume "$RESUME_PTH")
fi

"${CMD[@]}"

echo "=============================================================================="
echo "✅ Huấn luyện hoàn tất! Kết quả và checkpoint lưu tại: outputs/${EXP_NAME}"
echo "=============================================================================="

