#!/bin/bash
# ============================================================================
# RAER Ablation: NO PROMPT TUNING (Static Hand-Crafted Text Prompts)
# ============================================================================
# PHÂN ĐỊNH BẢN CHẤT KHOA HỌC:
#   1. "No Prompt Tuning" (Script này) vs. "No MI & DC" (raer_no_midc.sh):
#      - Ở script này: TẮT HOÀN TOÀN việc học Prompt (lr_prompt_learner=0.0,
#        requires_grad=False). Mô hình dùng văn bản cố định do con người viết
#        (Handcrafted text template: "a video of a student showing [CLASS]").
#        Vì text đóng băng, lambda_mi và lambda_dc tự động = 0.
#      - Ở raer_no_midc.sh: VẪN BẬT Prompt Tuning (lr=3e-4), nhưng KHÔNG áp hàm phạt
#        MI và DC, dẫn tới nguy cơ trôi dạt ngữ nghĩa (Semantic Drift / Prompt Collapse).
#
#   2. ẨN DỤ TRỰC QUAN:
#      - No Prompt Tuning: Học sinh dùng câu văn mẫu từ điển có sẵn, cấm sửa từ nào.
#      - No MI & DC: Học sinh được tự do viết câu mới nhưng không ai giám thị,
#        dẫn đến việc câu tả Mặt và câu tả Cảnh chép y hệt nhau.
#      - Full RAPT-CLIP: Tự do sáng tạo có định hướng (MI giữ nghĩa gốc, DC chống trùng lặp).
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Auto-detect Bounding Box paths (Local vs Kaggle)
FACE_BOX="RAER/bounding_box/face.json"
[ ! -f "$FACE_BOX" ] && FACE_BOX="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/face.json"

BODY_BOX="RAER/bounding_box/body.json"
[ ! -f "$BODY_BOX" ] && BODY_BOX="/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER/bounding_box/body.json"

python main.py \
  --mode train \
  --exper-name Ablation-RAER-NoPromptTuning \
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
  --lr-prompt-learner 0.0 \
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
  --text-type prompt_ensemble \
  --temporal-layers 1 \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner False \
  --ctx-init "a video of a student showing" \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --temperature 0.07 \
  --crop-body \
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --fusion-type cmaf \
  --use-context \
  "$@"
