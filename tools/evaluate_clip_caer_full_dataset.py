#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/evaluate_clip_caer_full_dataset.py
────────────────────────────────────────
Evaluates the official baseline clip-caer.pth (ViT-B/32, 2-Stream Face+Body)
on the FULL test dataset of RAER (528 test video clips).

Outputs:
- Full 5x5 Confusion Matrix
- WAR (Weighted Average Recall / Accuracy)
- UAR (Unweighted Average Recall)
- Per-class accuracy comparison against RAPT-CLIP
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class RecorderMeter:
    pass
sys.modules['__main__'].RecorderMeter = RecorderMeter

from models.clip import clip
from models.Temporal_Model import Temporal_Transformer_AttnPool
from models.Prompt_Learner import TextEncoder
from dataloader.video_dataloader import test_data_loader
from torch.utils.data import DataLoader

CLASSES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f"🚀 Running Full Evaluation of Baseline CLIP-CAER on RAER Test Set")
    print(f"   Device: {device}")

    # 1. Load Checkpoint
    ckpt_path = os.path.join(ROOT_DIR, 'clip-caer.pth')
    if not os.path.exists(ckpt_path):
        print(f"❌ Cannot find checkpoint: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    # 2. Visual Backbone (ViT-B/32)
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model = clip_model.float()
    img_sd = {k.replace('image_encoder.', ''): v for k, v in sd.items() if k.startswith('image_encoder.')}
    clip_model.visual.load_state_dict(img_sd, strict=False)
    clip_model.visual.eval()

    # 3. Temporal Transformers
    t_face = Temporal_Transformer_AttnPool(16, 512, 1, 8, 1024, 64, pooling='cls').to(device)
    t_face_sd = {k.replace('temporal_net.', ''): v for k, v in sd.items() if k.startswith('temporal_net.')}
    t_face.load_state_dict(t_face_sd, strict=True)
    t_face.eval()

    t_body = Temporal_Transformer_AttnPool(16, 512, 1, 8, 1024, 64, pooling='cls').to(device)
    t_body_sd = {k.replace('temporal_net_body.', ''): v for k, v in sd.items() if k.startswith('temporal_net_body.')}
    t_body.load_state_dict(t_body_sd, strict=True)
    t_body.eval()

    # 4. Fusion Projection
    proj = torch.nn.Linear(1024, 512).to(device)
    proj.weight.data.copy_(sd['project_fc.weight'])
    proj.bias.data.copy_(sd['project_fc.bias'])
    proj.eval()

    # 5. Text Encoder & Prompts
    text_encoder = TextEncoder(clip_model).to(device)
    text_sd = {k.replace('text_encoder.', ''): v for k, v in sd.items() if k.startswith('text_encoder.')}
    text_encoder.load_state_dict(text_sd, strict=False)
    text_encoder.eval()

    ctx = sd['prompt_learner.ctx'].to(device)
    prefix = sd['prompt_learner.token_prefix'].to(device)
    suffix = sd['prompt_learner.token_suffix'].to(device)
    full_prompt = torch.cat([prefix, ctx, suffix], dim=1)
    tokenized_prompts = torch.zeros(5, 77, dtype=torch.long, device=device)
    tokenized_prompts[:, -1] = 1

    with torch.no_grad():
        text_features = text_encoder(full_prompt, tokenized_prompts)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

    # 6. Dataloader for Full Test Set
    test_data = test_data_loader(
        root_dir='.',
        list_file='RAER/annotation/test.txt',
        num_segments=16,
        duration=1,
        image_size=224,
        bounding_box_face='RAER/bounding_box/face.json',
        bounding_box_body='RAER/bounding_box/body.json',
        crop_body=True,
        num_classes=5,
        dataset_name='RAER'
    )

    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2)
    print(f"📦 Loaded {len(test_data)} test video clips. Starting inference...")

    all_preds = []
    all_targets = []
    all_probs = []

    # Normalization constants for CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)

    t0 = time.time()
    with torch.no_grad():
        for face_batch, body_batch, _, target_batch in tqdm(test_loader, desc="Evaluating CLIP-CAER"):
            b = face_batch.shape[0]

            # Normalize inputs
            face_in = (face_batch.to(device) - mean) / std
            body_in = (body_batch.to(device) - mean) / std

            # Reshape to (B*16, 3, 224, 224)
            face_flat = face_in.view(-1, 3, 224, 224)
            body_flat = body_in.view(-1, 3, 224, 224)

            # ViT-B/32 Visual Encoder
            face_feats = clip_model.visual(face_flat).view(b, 16, 512)
            body_feats = clip_model.visual(body_flat).view(b, 16, 512)

            # Temporal Transformers
            face_vid = t_face(face_feats) # (B, 512)
            body_vid = t_body(body_feats) # (B, 512)

            # Dual-stream Concat & Project
            fused = torch.cat([face_vid, body_vid], dim=-1) # (B, 1024)
            video_feats = proj(fused) # (B, 512)
            video_feats = video_feats / (video_feats.norm(dim=-1, keepdim=True) + 1e-6)

            # Cosine similarity classification
            logits = video_feats @ text_features.t()
            probs = torch.softmax(logits / 0.07, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(target_batch.tolist())
            all_probs.extend(probs.cpu().tolist())

    elapsed = time.time() - t0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 7. Compute Metrics
    cm = confusion_matrix(all_targets, all_preds, labels=range(5))
    row_sums = cm.sum(axis=1)
    recalls = np.where(row_sums > 0, cm.diagonal() / row_sums * 100.0, 0.0)
    war = 100.0 * np.sum(all_preds == all_targets) / len(all_targets)
    uar = np.mean(recalls)

    print("\n" + "=" * 76)
    print("  🏆 KẾT QUẢ ĐÁNH GIÁ ĐẦY ĐỦ BASELINE CLIP-CAER TRÊN TOÀN BỘ TEST SET RAER")
    print("=" * 76)
    print(f"  • Thời gian chạy:      {elapsed:.2f}s ({len(all_targets) / elapsed:.1f} video/giây)")
    print(f"  • Tổng số video test:  {len(all_targets)}")
    print(f"  • Số mẫu đúng:         {np.sum(all_preds == all_targets)} / {len(all_targets)}")
    print(f"  • WAR (Weighted Acc):  {war:.2f}%  (Chính xác: {war:.4f}%)")
    print(f"  • UAR (Unweighted Rec):{uar:.2f}%  (Chính xác: {uar:.4f}%)")
    print("-" * 76)
    print("  CHI TIẾT RECALL TỪNG LỚP (BASELINE CLIP-CAER vs. RAPT-CLIP ĐỀ XUẤT):")
    rapt_recalls = [86.96, 68.75, 42.86, 76.92, 59.05]
    print(f"  {'Lớp cảm xúc':<14} | {'CLIP-CAER (Baseline)':<22} | {'RAPT-CLIP (Đề xuất)':<20} | {'Độ chênh lệch'}")
    print("-" * 76)
    for idx, (cls_name, r, c, total, rapt_r) in enumerate(zip(CLASSES, recalls, cm.diagonal(), row_sums, rapt_recalls)):
        diff = rapt_r - r
        diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
        print(f"  {cls_name:<14} | {c:3d}/{total:3d} ({r:6.2f}%)        | ({rapt_r:6.2f}%)           | {diff_str}")
    print("-" * 76)
    print("  MA TRẬN NHẦM LẪN 5x5 (CLIP-CAER):")
    print("      Predicted ->   Neu   Enj  Conf   Fat  Dist   | Total")
    for idx, (cls_name, row) in enumerate(zip(CLASSES, cm)):
        row_str = " ".join([f"{val:5d}" for val in row])
        print(f"  True {cls_name:<10}: {row_str}   | {row.sum():4d}")
    print("=" * 76)

    # Save results to outputs
    out_dict = {
        'model': 'CLIP-CAER (ViT-B/32 Dual-Stream)',
        'war': float(war),
        'uar': float(uar),
        'confusion_matrix': cm.tolist(),
        'recalls': {c: float(r) for c, r in zip(CLASSES, recalls)},
        'all_preds': all_preds.tolist(),
        'all_targets': all_targets.tolist(),
        'all_probs': all_probs.tolist()
    }
    torch.save(out_dict, os.path.join(ROOT_DIR, 'outputs', 'clip_caer_full_test_results.pt'))
    print(f"\n💾 Đã lưu kết quả chi tiết vào: outputs/clip_caer_full_test_results.pt")

if __name__ == '__main__':
    main()

