#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/verify_raer_metrics.py
────────────────────────────
Script kiểm chứng nhanh độc lập (Fast Verification Script) cho mô hình RAPT-CLIP trên tập test RAER (528 clips):
1. Tải checkpoint chính thức: outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth
2. Tải cache đặc trưng test: outputs/raer_extracted_features.pt
3. Tính toán và in ra:
   - Ma trận nhầm lẫn (Confusion Matrix 5x5)
   - Recall từng lớp: Neutral, Enjoyment, Confusion, Fatigue, Distraction
   - WAR (%) và UAR (%) chính xác
4. Hiển thị bảng đối chứng với tất cả các biến thể Ablation!

Chạy lệnh:
  python tools/verify_raer_metrics.py
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.Text import prompt_ensemble_5
from utils.builders import build_model
from utils.checkpoint_utils import load_slim_checkpoint

CLASSES = ["Neutral", "Enjoyment", "Confusion", "Fatigue", "Distraction"]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"=> [Thiết bị]: {device}")

    ckpt_path = os.path.join(ROOT_DIR, "outputs", "RAER-FrameLevelFusion-[06-07]-[03:06]", "model_best.pth")
    cache_path = os.path.join(ROOT_DIR, "outputs", "raer_extracted_features.pt")

    if not os.path.exists(ckpt_path):
        print(f"❌ Không tìm thấy checkpoint: {ckpt_path}")
        return
    if not os.path.exists(cache_path):
        print(f"❌ Không tìm thấy cache đặc trưng: {cache_path}")
        return

    # Mock arguments
    class Args:
        dataset = 'RAER'
        clip_path = 'ViT-B/16'
        temporal_layers = 1
        contexts_number = 8
        class_token_position = 'end'
        class_specific_contexts = True
        load_and_tune_prompt_learner = True
        num_segments = 16
        temperature = 0.07
        drop_path_rate = 0.0
        freeze_image_encoder = False
        ablation_no_text = False
        use_v2 = False
        use_moco = False
        modality_dropout = 0.0
        lr_image_encoder = 1e-6
        text_type = 'prompt_ensemble'
        fusion_type = 'cmaf'
        fusion_stage = 'frame'
        use_context = True
        crop_body = True
        num_classes = 5
        temporal_pooling = 'attn'

    args = Args()
    args.device = device

    print(f"=> Đang khởi tạo mô hình và nạp checkpoint: {os.path.basename(ckpt_path)}...")
    model = build_model(args, prompt_ensemble_5).to(device)
    load_slim_checkpoint(model, ckpt_path, device=device)
    model.eval()

    print(f"=> Đang nạp cache đặc trưng kiểm thử (528 clips)...")
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    face_all = data['face_all']
    body_all = data['body_all']
    ctx_all  = data['ctx_all']
    targets  = data['targets_all']
    total_videos = len(targets)

    # Tính text features
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized_prompts = model.tokenized_prompts
        text_features = model.text_encoder(prompts, tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = text_features.view(5, model.num_prompts_per_class, -1)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        preds = []
        batch_size = 32
        num_batches = int(np.ceil(total_videos / batch_size))

        for b_idx in range(num_batches):
            start = b_idx * batch_size
            end = min((b_idx + 1) * batch_size, total_videos)
            b_size = end - start

            f_b = face_all[start*16 : end*16].to(device)
            b_b = body_all[start*16 : end*16].to(device)
            c_b = ctx_all[start*16 : end*16].to(device)

            # CMAF
            fused = model.cmaf(f_b, b_b, c_b).view(b_size, 16, -1)
            # Unified Temporal
            vid_feat = model.unified_temporal_net(fused)
            # Project FC
            vid_feat = model.project_fc(vid_feat)
            vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)

            logits = torch.einsum('bd,cpd->bcp', vid_feat, text_features).mean(dim=-1) / args.temperature
            pred = logits.argmax(dim=1)
            preds.append(pred.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    cm = confusion_matrix(targets, preds, labels=range(5))
    row_sums = cm.sum(axis=1)
    recalls = cm.diagonal() / row_sums * 100.0
    war = 100.0 * np.sum(preds == targets) / total_videos
    uar = np.mean(recalls)

    print("\n" + "="*70)
    print("  KẾT QUẢ KIỂM CHỨNG TRỰC TIẾP TỪ CHECKPOINT BASELINE (SOTA ĐỀ XUẤT)")
    print("="*70)
    print(f"  • Tổng số video test: {total_videos}")
    print(f"  • Số mẫu đoán đúng:   {np.sum(preds == targets)} / {total_videos}")
    print(f"  • WAR (Weighted Accuracy):   {war:.2f}%  (Chính xác: {war:.4f}%)")
    print(f"  • UAR (Unweighted Recall):   {uar:.2f}%  (Chính xác: {uar:.4f}%)")
    print("-" * 70)
    print("  CHI TIẾT RECALL TỪNG LỚP CẢM XÚC:")
    for idx, (cls_name, r, c, total) in enumerate(zip(CLASSES, recalls, cm.diagonal(), row_sums)):
        print(f"    [{idx}] {cls_name:<12}: {c:3d}/{total:3d} ({r:6.2f}%)")
    print("-" * 70)
    print("  MA TRẬN NHẦM LẪN (CONFUSION MATRIX 5x5):")
    print("      Predicted ->   Neu   Enj  Conf   Fat  Dist   | Total")
    for idx, (cls_name, row) in enumerate(zip(CLASSES, cm)):
        row_str = " ".join([f"{val:5d}" for val in row])
        print(f"  True {cls_name:<10}: {row_str}   | {row.sum():4d}")
    print("="*70)

if __name__ == "__main__":
    main()
