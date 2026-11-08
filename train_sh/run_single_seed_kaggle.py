#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Training Runner: Chạy 1 Seed Độc Lập Với Văn Bản Gốc (Original Text / class_descriptor)
- Tự động hiển thị log thời gian thực (Real-time Streaming, không bị buffer/treo màn hình trên Kaggle).
- Chỉ cần đổi duy nhất biến TARGET_SEED ở dòng đầu tiên.
- Khớp 100% siêu tham số và đường dẫn trên Kaggle của bạn.
"""

import os
import sys
import subprocess

# ==============================================================================
# 👉 BẠN CHỈ CẦN THAY ĐỔI DUY NHẤT DÒNG NÀY (ví dụ: 7, 123, 456, 789)
# ==============================================================================
TARGET_SEED = 7

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN KAGGLE CỦA BẠN
# ==============================================================================
RESUME_CKPT = "/kaggle/input/models/bearmn/origin-full/tensorflow2/default/1/origin_text.pth"
ROOT_DIR = "/kaggle/input/datasets/bearmn/raer-video-emotion-dataset"
TRAIN_ANNOT = "/kaggle/input/datasets/bearmn/raer-annot/annotation/train.txt"
VAL_ANNOT = "/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"
TEST_ANNOT = "/kaggle/input/datasets/bearmn/raer-annot/annotation/test.txt"
BBOX_FACE = "RAER/bounding_box/face.json"
BBOX_BODY = "RAER/bounding_box/body.json"

EXP_NAME = f"RAER_OriginalText_WithView2_Seed{TARGET_SEED}"

# Kiểm tra đường dẫn resume
if not os.path.exists(RESUME_CKPT):
    # Dự phòng nếu đường dẫn nằm ở dataset khác trên Kaggle
    alt_resumes = [
        "/kaggle/input/origin-text/origin_text.pth",
        "/kaggle/input/origin-full/origin_text.pth",
        "origin_text.pth"
    ]
    for alt in alt_resumes:
        if os.path.exists(alt):
            RESUME_CKPT = alt
            break

cmd = [
    sys.executable, "-u", "main.py",
    "--mode", "train",
    "--exper-name", EXP_NAME,
    "--dataset", "RAER",
    "--gpu", "0",
    "--workers", "2",
    "--seed", str(TARGET_SEED),
    "--root-dir", ROOT_DIR,
    "--train-annotation", TRAIN_ANNOT,
    "--val-annotation", VAL_ANNOT,
    "--test-annotation", TEST_ANNOT,
    "--clip-path", "ViT-B/16",
    "--bounding-box-face", BBOX_FACE,
    "--bounding-box-body", BBOX_BODY,
    "--epochs", "20",
    "--early-stopping-patience", "3",
    "--batch-size", "4",
    "--print-freq", "50",
    "--use-amp",
    "--grad-clip", "1.0",
    "--optimizer", "AdamW",
    "--lr", "2e-05",
    "--lr-image-encoder", "1e-06",
    "--lr-prompt-learner", "0.0003",
    "--lr-adapter", "0.0001",
    "--weight-decay", "0.005",
    "--momentum", "0.9",
    "--milestones", "10", "15",
    "--gamma", "0.1",
    "--loss-type", "ldam",
    "--ldam-s", "30.0",
    "--ldam-max-m", "0.5",
    "--mixup-alpha", "0.0",
    "--lambda_mi", "0.1",
    "--lambda_dc", "0.1",
    "--mi-warmup", "5",
    "--mi-ramp", "10",
    "--mi-ramp-type", "ramp_up",
    "--dc-warmup", "5",
    "--dc-ramp", "10",
    "--use-weighted-sampler",
    "--label-smoothing", "0.05",
    "--text-type", "class_descriptor",     # 100% Văn bản gốc CLIP-CAER
    "--temporal-layers", "1",
    "--contexts-number", "8",
    "--class-token-position", "end",
    "--class-specific-contexts", "True",
    "--load_and_tune_prompt_learner", "True",
    "--num-segments", "16",
    "--duration", "1",
    "--image-size", "224",
    "--temperature", "0.07",
    "--crop-body",
    "--drop-path-rate", "0.0",
    "--modality-dropout", "0.3",
    "--fusion-type", "cmaf",
    "--use-context"
]

if os.path.exists(RESUME_CKPT):
    print(f"📦 Resume từ Checkpoint: {RESUME_CKPT}")
    cmd.extend(["--resume", RESUME_CKPT])
else:
    print(f"⚠️ Không tìm thấy checkpoint resume tại {RESUME_CKPT}. Huấn luyện mới từ Epoch 0.")

print("=" * 70)
print(f"🚀 KHỞI ĐỘNG HUẤN LUYỆN SEED {TARGET_SEED} (VĂN BẢN GỐC / CLASS_DESCRIPTOR)")
print(f"   • Tên thí nghiệm: {EXP_NAME}")
print("=" * 70)

# Cấu hình môi trường chạy unbuffered để Kaggle hiển thị log ngay lập tức
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1,
    env=env
)

for line in process.stdout:
    print(line, end="", flush=True)

process.wait()
if process.returncode != 0:
    print(f"\n❌ Huấn luyện kết thúc với mã lỗi: {process.returncode}")
else:
    print(f"\n✅ Hoàn tất huấn luyện Seed {TARGET_SEED} thành công!")

