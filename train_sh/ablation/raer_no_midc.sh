#!/bin/bash
# ============================================================================
# RAER Ablation: NO MI & DC REGULARIZATION (Unconstrained Prompt Tuning)
# ============================================================================
# PHÂN ĐỊNH BẢN CHẤT KHOA HỌC:
#   1. "No MI & DC" (Script này) vs. "No Prompt Tuning" (raer_no_prompt_tuning.sh):
#      - Ở script này: VẪN BẬT Prompt Tuning (lr_prompt_learner=0.0003), cho phép
#        mô hình tự cập nhật các context vectors. Tuy nhiên, TẮT HOÀN TOÀN cả 2 hàm
#        ràng buộc (lambda_mi=0.0, lambda_dc=0.0).
#      - Ở raer_no_prompt_tuning.sh: TẮT HOÀN TOÀN việc học Prompt (lr=0), dùng câu
#        chữ viết tay cố định (Handcrafted descriptors).
#
#   2. CÂU HỎI KHOA HỌC KIỂM CHỨNG:
#      - "Khi cho mô hình tự học prompt, nếu không có hàm phạt MI (giữ nghĩa gốc)
#        và DC (chống trùng lặp giữa Mặt và Cảnh) thì có bị Semantic Drift hay không?"
#
#   3. ẨN DỤ TRỰC QUAN:
#      - No MI & DC: Học sinh được tự do viết câu văn mới nhưng không có giám thị chấm,
#        dẫn đến việc câu tả Mặt và câu tả Cảnh chép y hệt nhau hoặc lạc đề.
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
  --exper-name Ablation-RAER-NoMIDC \
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
  --lr-prompt-learner 0.0003 \
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
  "$@"
