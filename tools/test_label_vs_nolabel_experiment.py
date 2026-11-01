#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/test_label_vs_nolabel_experiment.py
──────────────────────────────────────────
Thực nghiệm khoa học: Đánh giá tác động của việc CÓ hay KHÔNG CÓ TÊN NHÃN trong câu mô tả.
Huấn luyện và đánh giá trên 2,121 video train và 528 video test RAER.

So sánh 2 kịch bản:
- Kịch bản A (Có tên nhãn - Gốc):
  "neutral face", "looks happy", "looks confused", "looks tired", "looks distracted"
- Kịch bản B (HOÀN TOÀN KHÔNG CÓ TÊN NHÃN - Pure Visual Cues):
  Loại bỏ toàn bộ từ khóa nhãn, chỉ giữ lại miêu tả sinh trắc học cử chỉ mắt, chân mày, khóe miệng.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from models.Prompt_Learner import PromptLearner, TextEncoder
from models.Text import prompt_ensemble_5, class_descriptor_5_only_face
from tools.fast_sweep_midc_raer import run_single_midc_trial

RAER_CLASSES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

# 1. CÂU MÔ TẢ GỐC (CÓ TÊN NHÃN)
DESCRIPTOR_WITH_LABEL = class_descriptor_5_only_face

# 2. CÂU MÔ TẢ BỎ HOÀN TOÀN TÊN NHÃN (PURE OBSERVABLE VISUAL CUES)
DESCRIPTOR_NO_LABEL = [
    "A student with relaxed mouth, normally open eyes, smooth forehead, and calm facial muscles.",
    "A student with a slight smile, upturned lip corners, bright crinkled eyes, and raised cheeks.",
    "A student with furrowed lowered eyebrows, squinted perplexed eyes, and slightly parted mouth.",
    "A student with heavy drooping eyelids, slow blinking, frequent yawning, and slack jaw.",
    "A student with unfocused wandering gaze, head turned sideways, looking away from learning materials."
]

def run_experiment(name, descriptors, train_tensors, test_tensors, clip_model, device):
    print(f"\n🚀 Đang chạy huấn luyện kiểm chứng: [{name}]...")
    # Encode descriptors for MI Loss
    tokens = torch.cat([clip.tokenize(p) for p in descriptors]).to(device)
    with torch.no_grad():
        anchor_feats = clip_model.encode_text(tokens).float()
        anchor_feats = F.normalize(anchor_feats, dim=-1)
        
    res = run_single_midc_trial(
        trial_name=name,
        lambda_mi=0.10,
        lambda_dc=0.10,
        train_tensors=train_tensors,
        test_tensors=test_tensors,
        clip_model=clip_model,
        hand_crafted_text_features=anchor_feats,
        device=device,
        epochs=12,
        warmup=2,
        ramp=4,
        batch_size=64,
        lr_head=2e-4,
        lr_prompt=3e-4,
        seed=42
    )
    return res['uar'], res['war'], res['recalls'], res['cm']

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"⚡ Thiết bị chạy thực nghiệm: {device}")
    
    train_path = os.path.join(ROOT_DIR, "cached_features", "raer", "raer_train_features.pt")
    test_path = os.path.join(ROOT_DIR, "cached_features", "raer", "raer_test_features.pt")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Không tìm thấy cache trong cached_features/")
        return
        
    print(f"📦 Đang nạp cached visual features (2,121 train / 528 test)...")
    train_d = torch.load(train_path, map_location='cpu', weights_only=False)
    test_d = torch.load(test_path, map_location='cpu', weights_only=False)
    
    train_tensors = (train_d['face_feats'], train_d['body_feats'], train_d['context_feats'], train_d['labels'])
    test_tensors = (test_d['face_feats'], test_d['body_feats'], test_d['context_feats'], test_d['labels'])
    
    print(f"🧠 Đang nạp CLIP ViT-B/16 text encoder...")
    clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_model = clip_model.float()
    
    # 1. Chạy với Prompt gốc (Có tên nhãn)
    uar_a, war_a, recs_a, cm_a = run_experiment(
        "Kịch bản A: CÓ TÊN NHÃN (Gốc)", 
        DESCRIPTOR_WITH_LABEL, 
        train_tensors, test_tensors, clip_model, device
    )
    
    # 2. Chạy với Prompt BỎ NHÃN (Pure Visual Cues)
    uar_b, war_b, recs_b, cm_b = run_experiment(
        "Kịch bản B: BỎ HOÀN TOÀN TÊN NHÃN (Pure Visual Cues)", 
        DESCRIPTOR_NO_LABEL, 
        train_tensors, test_tensors, clip_model, device
    )
    
    print("\n" + "="*80)
    print(" 📊 BẢNG SO SÁNH ĐỐI CHỨNG TRỰC DIỆN: CÓ TÊN NHÃN vs BỎ TÊN NHÃN")
    print("="*80)
    print(f"{'Kịch Bản Thử Nghiệm':<48} | {'WAR (%)':<8} | {'UAR (%)':<8} | {'Chênh lệch UAR'}")
    print("-" * 80)
    print(f"{'Kịch bản A: CÓ TÊN NHÃN (Gốc)':<48} | {war_a:6.2f}%  | {uar_a:6.2f}%  | Baseline")
    delta_uar = uar_b - uar_a
    sign = "+" if delta_uar >= 0 else ""
    print(f"{'Kịch bản B: BỎ TÊN NHÃN (Pure Visual Cues)':<48} | {war_b:6.2f}%  | {uar_b:6.2f}%  | {sign}{delta_uar:.2f}%")
    print("-" * 80)
    print(" Chi tiết Recall từng lớp:")
    print(f"   • Kịch bản A (Có nhãn) : " + " | ".join([f"{RAER_CLASSES[i][:4]}: {recs_a[i]:.1f}%" for i in range(5)]))
    print(f"   • Kịch bản B (Bỏ nhãn) : " + " | ".join([f"{RAER_CLASSES[i][:4]}: {recs_b[i]:.1f}%" for i in range(5)]))
    print("="*80)

if __name__ == "__main__":
    main()
