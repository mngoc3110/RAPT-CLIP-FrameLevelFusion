#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_emotic_modality_ablation.py
─────────────────────────────────────
Nghiên cứu cắt bỏ nhánh thị giác (Modality Ablation Study) trên bộ dữ liệu chuẩn EMOTIC (5,292 ảnh test).
Đánh giá chính xác vai trò và đóng góp của từng nhánh:
1. Full Tri-Stream (Face + Body + Context)
2. W/o Body (Face + Context)
3. W/o Context (Face + Body)
4. W/o Face (Body + Context)
5. Context-Only
6. Body-Only
7. Face-Only
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torchvision
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor, GroupNormalize
from dataloader.video_dataloader import EmoticNpyDataset

from main import get_class_info, build_model
from utils.checkpoint_utils import load_slim_checkpoint

EMOTIC_CLASSES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 
    'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = os.path.join(ROOT_DIR, 'outputs', 'EMOTIC')
    checkpoint_path = os.path.join(output_dir, 'model_best.zip')
    cache_path = os.path.join(output_dir, 'emotic_npy_ablation_features.pt')

    args = argparse.Namespace(
        dataset='EMOTIC',
        mode='eval',
        clip_path='ViT-B/16',
        use_npy=True,
        npy_dir='emotic_dataset/emotic_pre',
        batch_size=64,
        workers=2,
        seed=42,
        gpu='mps',
        device=device,
        image_size=224,
        text_type='prompt_ensemble',
        contexts_number=8,
        ctx_init='',
        class_token_position='end',
        class_specific_contexts=True,
        load_and_tune_prompt_learner=True,
        num_segments=1,
        duration=1,
        temperature=1.0,
        crop_body=True,
        fusion_type='cmaf',
        fusion_stage='frame',
        use_context=True,
        modality_dropout=0.0,
        drop_path_rate=0.0,
        freeze_image_encoder=False,
        lr_image_encoder=1e-5,
        no_temporal=False,
        use_temporal=True,
        temporal_layers=1,
        ablation_no_text=False,
        use_v2=False,
        mask_context_face=False,
        daisee_merge_low=False,
        eval_checkpoint=checkpoint_path,
        root_dir='./',
    )

    class_names, input_text = get_class_info(args)
    print(f"Loaded {len(class_names)} classes for EMOTIC.")

    model = build_model(args, input_text).to(device)
    print(f"=> Loading checkpoint from {checkpoint_path}...")
    load_slim_checkpoint(model, checkpoint_path, device=device)
    model.eval()

    # Step 1: Feature Extraction or Cache Loading
    if os.path.exists(cache_path):
        print(f"=> Loading pre-extracted features from {cache_path}...")
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        all_f = cache['f_feat']
        all_b = cache['b_feat']
        all_c = cache['c_feat']
        targets = cache['targets']
        print(f"Loaded {len(targets)} samples.")
    else:
        print("=> Building test dataset directly from EmoticNpyDataset...")
        test_transform = torchvision.transforms.Compose([
            GroupResize(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        test_data = EmoticNpyDataset(npy_dir=args.npy_dir, mode='test', transform=test_transform, num_classes=26)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        print(f"=> Extracting multi-stream features for EMOTIC test set ({len(test_data)} samples)...")
        f_list, b_list, c_list, target_list = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[Extracting EMOTIC NPY Features]"):
                face, body, context, target = [x.to(device) if x is not None else None for x in batch]
                # Pass through image_encoder
                f_feat = model.image_encoder(face.view(-1, 3, 224, 224).type(model.dtype))
                if hasattr(model, 'face_adapter'):
                    f_feat = model.face_adapter(f_feat)

                b_feat = model.image_encoder(body.view(-1, 3, 224, 224).type(model.dtype))
                c_feat = model.image_encoder(context.view(-1, 3, 224, 224).type(model.dtype))

                f_list.append(f_feat.cpu().float())
                b_list.append(b_feat.cpu().float())
                c_list.append(c_feat.cpu().float())
                target_list.append(target.cpu().float())

        all_f = torch.cat(f_list, dim=0)
        all_b = torch.cat(b_list, dim=0)
        all_c = torch.cat(c_list, dim=0)
        targets = torch.cat(target_list, dim=0).numpy()

        torch.save({
            'f_feat': all_f,
            'b_feat': all_b,
            'c_feat': all_c,
            'targets': targets
        }, cache_path)
        print(f"Saved extracted features to {cache_path} ({all_f.shape}).")

    # Step 2: Compute Text Ensemble Embeddings
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized_prompts = model.tokenized_prompts
        text_features = model.text_encoder(prompts, tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = text_features.view(len(class_names), model.num_prompts_per_class, -1)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

    # Step 3: Define evaluation helper
    def evaluate_combination(use_f=True, use_b=True, use_c=True):
        preds = []
        batch_size = 64
        total_samples = len(targets)
        num_batches = int(np.ceil(total_samples / batch_size))

        with torch.no_grad():
            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = min((b_idx + 1) * batch_size, total_samples)

                f_sub = all_f[start:end].to(device) if use_f else torch.zeros_like(all_f[start:end]).to(device)
                b_sub = all_b[start:end].to(device) if use_b else torch.zeros_like(all_b[start:end]).to(device)
                c_sub = all_c[start:end].to(device) if use_c else torch.zeros_like(all_c[start:end]).to(device)

                # CMAF
                fused = model.cmaf(f_sub, b_sub, c_sub) # (B, 1536)

                if model.use_temporal:
                    fused = fused.unsqueeze(1) # (B, 1, 1536)
                    vid_feat = model.unified_temporal_net(fused)
                else:
                    vid_feat = fused

                vid_feat = model.project_fc(vid_feat) # (B, 512)
                vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)

                # Similarity with Text Prompts
                # vid_feat: (B, 512), text_features: (26, 8, 512)
                logits = torch.einsum('bd,cpd->bcp', vid_feat, text_features)
                out = torch.mean(logits, dim=2) / args.temperature
                probs = torch.sigmoid(out).cpu().numpy()
                preds.append(probs)

        all_preds = np.concatenate(preds, axis=0)

        # Compute per-class AP and macro mAP
        ap_scores = []
        for c in range(targets.shape[1]):
            ap = average_precision_score(targets[:, c], all_preds[:, c])
            ap_scores.append(ap * 100.0)

        macro_map = np.mean(ap_scores)
        return macro_map, ap_scores

    # Step 4: Run all 7 variants
    experiments = [
        ("RAPT-CLIP (Full Tri-Stream)", True, True, True),
        ("W/o Body (Face + Context)", True, False, True),
        ("W/o Context (Face + Body)", True, True, False),
        ("W/o Face (Body + Context)", False, True, True),
        ("Context-Only", False, False, True),
        ("Body-Only", False, True, False),
        ("Face-Only", True, False, False),
    ]

    results = {}
    print("\n" + "="*80)
    print("MODALITY ABLATION STUDY RESULTS ON EMOTIC TEST SET (Macro mAP %)")
    print("="*80)
    print(f"{'Configuration':<32} | {'Face':<5} | {'Body':<5} | {'Context':<7} | {'mAP (%)':<8} | {'Δ (%)':<8}")
    print("-" * 80)

    base_map = None
    for name, uf, ub, uc in experiments:
        m_map, per_class = evaluate_combination(uf, ub, uc)
        if base_map is None:
            base_map = m_map
        delta = m_map - base_map
        results[name] = {
            'macro_map': float(m_map),
            'delta': float(delta),
            'use_face': uf,
            'use_body': ub,
            'use_context': uc,
            'per_class_ap': [float(x) for x in per_class]
        }
        f_s = "✅" if uf else "❌"
        b_s = "✅" if ub else "❌"
        c_s = "✅" if uc else "❌"
        delta_str = f"{delta:+.2f}%" if delta != 0 else "0.00% (SOTA)"
        print(f"{name:<32} | {f_s:<5} | {b_s:<5} | {c_s:<7} | {m_map:6.2f}% | {delta_str:<8}")

    print("="*80)

    # Save results json
    res_json_path = os.path.join(output_dir, 'emotic_modality_ablation_results.json')
    with open(res_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON results to {res_json_path}")

    # Generate Markdown Report
    md_path = os.path.join(ROOT_DIR, 'outputs', 'emotic_modality_ablation_report.md')
    with open(md_path, 'w') as f:
        f.write("# Báo Cáo Nghiên Cứu Cắt Bỏ Nhánh Thị Giác (Modality Ablation Study) Trên EMOTIC\n\n")
        f.write("> **Mô hình:** RAPT-CLIP (ViT-B/16 + EAA + CMAF Tri-stream + Unified Temporal Net)\n")
        f.write(f"> **Checkpoint:** `outputs/EMOTIC/model_best.zip` (Epoch 6 — Benchmark SOTA: **34.76% mAP**)\n")
        f.write(f"> **Tập kiểm thử:** EMOTIC Test Benchmark ({len(targets):,} mẫu, 26 nhãn cảm xúc đa nhãn)\n\n")
        f.write("---\n\n")
        f.write("### 1. Bảng Tổng Hợp Hiệu Năng Cắt Bỏ Nhánh (Modality Ablation Study)\n\n")
        f.write("| Cấu hình kiểm thử (Modality) | Face | Body | Context | Macro mAP (%) | Chênh lệch (Δ) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for name, uf, ub, uc in experiments:
            d = results[name]
            f_s = "✅" if uf else "❌"
            b_s = "✅" if ub else "❌"
            c_s = "✅" if uc else "❌"
            delta_str = f"**{d['delta']:+.2f}%**" if d['delta'] != 0 else "**0.00% (SOTA)**"
            f.write(f"| **{name}** | {f_s} | {b_s} | {c_s} | **{d['macro_map']:.2f}%** | {delta_str} |\n")
        f.write("\n---\n\n")
        f.write("### 2. Bảng Chi Tiết 26 Lớp Cảm Xúc (Per-Class AP %)\n\n")
        f.write("| STT | Lớp cảm xúc (Class) | Full Tri-Stream | W/o Body | W/o Context | W/o Face | Context-Only | Body-Only | Face-Only |\n")
        f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for idx, c_name in enumerate(EMOTIC_CLASSES):
            line = f"| {idx} | {c_name} "
            for exp_name, _, _, _ in experiments:
                ap = results[exp_name]['per_class_ap'][idx]
                line += f"| {ap:.2f}% "
            line += "|\n"
            f.write(line)
        f.write("\n---\n")

    print(f"Generated comprehensive report at {md_path}")

if __name__ == '__main__':
    main()
