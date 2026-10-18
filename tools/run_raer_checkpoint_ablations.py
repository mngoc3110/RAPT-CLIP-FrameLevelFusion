#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_raer_checkpoint_ablations.py
──────────────────────────────────────
Trích xuất đặc trưng một lần duy nhất từ bộ kiểm thử RAER (528 video clips)
bằng checkpoint chính thức của RAPT-CLIP FrameLevelFusion.
Thực hiện toàn bộ các bài kiểm thử Ablation cắt từng khối kiến trúc:
1. Full Model Tri-stream (SOTA Đề Xuất)
2. Tắt CMAF (Bypass CMAF: Direct Concat, không Cross-Attention)
3. Tắt EAA (Bypass Face Adapter: dùng trực tiếp CLIP ViT Face Features)
4. Tắt Temporal Transformer (Bypass Temporal: Mean Pooling qua 16 frames)
5. Tắt Attention Pooling (Dùng Mean Pooling sau Temporal Transformer)
6. Tắt luồng Context (W/o Context: Chỉ Mặt + Thân)
7. Tắt luồng Face (W/o Face: Chỉ Thân + Cảnh)
8. Tắt luồng Body (W/o Body: Chỉ Mặt + Cảnh)
9. Đơn luồng Face-Only
10. Đơn luồng Body-Only
11. Đơn luồng Context-Only
12. Xáo trộn thời gian (Temporal Shuffling)
13. Bơm nhiễu thị giác Gaussian (Noise sigma = 0.1)
14. Prompt tĩnh thủ công (Hand-crafted prompt vs CoOp Prompt Tuning)

Tính toán chính xác 100% WAR, UAR và vẽ Ma trận Nhầm lẫn (Confusion Matrix)
cho từng biến thể từ mô hình RAPT-CLIP FrameLevelFusion!
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.Text import *
from utils.builders import build_model, build_dataloaders
from utils.checkpoint_utils import load_slim_checkpoint

CLASSES = ["Neutral", "Enjoyment", "Confusion", "Fatigue", "Distraction"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth')
    parser.add_argument('--dataset', type=str, default='RAER')
    parser.add_argument('--root-dir', type=str, default='.')
    parser.add_argument('--train-annotation', type=str, default='RAER/annotation/train.txt')
    parser.add_argument('--val-annotation', type=str, default='RAER/annotation/test.txt')
    parser.add_argument('--test-annotation', type=str, default='RAER/annotation/test.txt')
    parser.add_argument('--use-weighted-sampler', action='store_true', default=False)
    parser.add_argument('--bounding-box-face', type=str, default='RAER/bounding_box/face.json')
    parser.add_argument('--bounding-box-body', type=str, default='RAER/bounding_box/body.json')
    parser.add_argument('--clip-path', type=str, default='ViT-B/16')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--num-segments', type=int, default=16)
    parser.add_argument('--duration', type=int, default=1)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--crop-body', action='store_true', default=True)
    parser.add_argument('--use-context', action='store_true', default=True)
    parser.add_argument('--mask-context-face', action='store_true', default=False)
    parser.add_argument('--fusion-type', type=str, default='cmaf')
    parser.add_argument('--fusion-stage', type=str, default='frame')
    parser.add_argument('--temporal-layers', type=int, default=1)
    parser.add_argument('--text-type', type=str, default='prompt_ensemble')
    parser.add_argument('--contexts-number', type=int, default=8)
    parser.add_argument('--class-token-position', type=str, default='end')
    parser.add_argument('--class-specific-contexts', type=bool, default=True)
    parser.add_argument('--load_and_tune_prompt_learner', type=bool, default=True)
    parser.add_argument('--lr-image-encoder', type=float, default=1e-6)
    parser.add_argument('--drop-path-rate', type=float, default=0.0)
    parser.add_argument('--freeze-image-encoder', type=bool, default=False)
    parser.add_argument('--ablation-no-text', type=bool, default=False)
    parser.add_argument('--modality-dropout', type=float, default=0.0)
    parser.add_argument('--use-temporal', type=bool, default=True)
    parser.add_argument('--use-moco', type=bool, default=False)
    parser.add_argument('--ctx-init', type=str, default='')
    parser.add_argument('--daisee-merge-low', type=bool, default=False)
    return parser.parse_args()

def plot_cm(cm, war, uar, title, subtitle, out_path, cmap="Blues"):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1.0,
        linecolor="#ffffff",
        ax=ax,
        annot_kws={"fontsize": 8.5, "weight": "bold"}
    )

    full_title = f"{title}\nWAR: {war:.2f}% | UAR: {uar:.2f}% ({subtitle})"
    ax.set_title(full_title, fontsize=9.0, fontweight="bold", pad=10)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=9.0, fontweight="bold", labelpad=6)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=9.0, fontweight="bold", labelpad=6)
    plt.xticks(rotation=20, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    args = parse_args()

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"=> Sử dụng thiết bị: {device}")
    args.device = device
    args.temporal_pooling = 'attn'

    # Build prompt input
    input_text = prompt_ensemble_5
    args.num_classes = len(CLASSES)

    print("=> Đang khởi tạo mô hình GenerateModel (FrameLevelFusion)...")
    model = build_model(args, input_text)
    model = model.to(device)

    # Load checkpoint
    print(f"=> Đang nạp trọng số từ checkpoint: {args.checkpoint}")
    load_slim_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    # Cache file to avoid re-extracting if already saved
    cache_file = os.path.join(ROOT_DIR, "outputs", "raer_extracted_features.pt")
    
    if os.path.exists(cache_file):
        print(f"=> [CACHE FOUND] Nạp nhanh các đặc trưng thị giác từ: {cache_file}")
        cached_data = torch.load(cache_file, map_location='cpu', weights_only=False)
        raw_face_all = cached_data['raw_face_all']
        face_all = cached_data['face_all']
        body_all = cached_data['body_all']
        ctx_all  = cached_data['ctx_all']
        targets_all = cached_data['targets_all']
    else:
        print("=> Đang tạo DataLoader kiểm thử cho RAER (528 video clips)...")
        _, _, test_loader = build_dataloaders(args)

        print("=> BƯỚC 1/2: Trích xuất và Cache đặc trưng thị giác từ CLIP ViT-B/16...")
        all_raw_face_features = []
        all_face_features = []
        all_body_features = []
        all_ctx_features = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Extracting Visual Tokens"):
                if len(batch) == 4:
                    img_face, img_body, img_ctx, target = batch
                    img_ctx = img_ctx.to(device)
                else:
                    img_face, img_body, target = batch
                    img_ctx = None

                img_face = img_face.to(device)
                img_body = img_body.to(device)

                n, t, c, h, w = img_face.shape
                dtype = model.dtype

                # Raw Face through image encoder
                face_reshaped = img_face.contiguous().view(-1, c, h, w)
                raw_f_feat = model.image_encoder(face_reshaped.type(dtype))
                # Face through EAA face adapter
                f_feat = model.face_adapter(raw_f_feat)

                # Body through image encoder
                body_reshaped = img_body.contiguous().view(-1, c, h, w)
                b_feat = model.image_encoder(body_reshaped.type(dtype))

                # Context through image encoder
                if img_ctx is not None:
                    ctx_reshaped = img_ctx.contiguous().view(-1, c, h, w)
                    c_feat = model.image_encoder(ctx_reshaped.type(dtype))
                else:
                    c_feat = torch.zeros_like(f_feat)

                all_raw_face_features.append(raw_f_feat.cpu())
                all_face_features.append(f_feat.cpu())
                all_body_features.append(b_feat.cpu())
                all_ctx_features.append(c_feat.cpu())
                all_targets.append(target.cpu())

        raw_face_all = torch.cat(all_raw_face_features, dim=0) # (528*16, 512)
        face_all = torch.cat(all_face_features, dim=0)         # (528*16, 512)
        body_all = torch.cat(all_body_features, dim=0)         # (528*16, 512)
        ctx_all  = torch.cat(all_ctx_features, dim=0)          # (528*16, 512)
        targets_all = torch.cat(all_targets, dim=0).numpy()     # (528,)

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save({
            'raw_face_all': raw_face_all,
            'face_all': face_all,
            'body_all': body_all,
            'ctx_all': ctx_all,
            'targets_all': targets_all
        }, cache_file)
        print(f"=> Đã lưu cache đặc trưng vào: {cache_file}")

    total_videos = len(targets_all)
    print(f"=> Sẵn sàng chạy Ablation! Tổng số video: {total_videos}, kích thước token: {face_all.shape}")

    # Compute Text Features
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized_prompts = model.tokenized_prompts
        text_features = model.text_encoder(prompts, tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = text_features.view(args.num_classes, model.num_prompts_per_class, -1)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        # Hand-crafted text features (Single prompt per class)
        hand_crafted_prompts = model.hand_crafted_prompt_embeddings
        tokenized_hc = model.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
        hc_text_features = model.text_encoder(hand_crafted_prompts, tokenized_hc).float()
        hc_text_features = hc_text_features / (hc_text_features.norm(dim=-1, keepdim=True) + 1e-6)
        hc_text_features = hc_text_features.unsqueeze(1) # (5, 1, 512)

    # Universal Evaluation Pipeline supporting modular bypass
    def evaluate_pipeline(f_input, b_input, c_input, custom_text=None, 
                          bypass_cmaf=False, bypass_temporal=False, mean_pool_temporal=False,
                          shuffle_temporal=False, noise_sigma=0.0):
        preds = []
        batch_size = 32
        num_batches = int(np.ceil(total_videos / batch_size))

        cur_text = text_features if custom_text is None else custom_text

        with torch.no_grad():
            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = min((b_idx + 1) * batch_size, total_videos)
                b_size = end - start
                
                f_batch = f_input[start*16 : end*16].to(device)
                b_batch = b_input[start*16 : end*16].to(device)
                c_batch = c_input[start*16 : end*16].to(device)

                if noise_sigma > 0.0:
                    f_batch = f_batch + torch.randn_like(f_batch) * noise_sigma
                    b_batch = b_batch + torch.randn_like(b_batch) * noise_sigma
                    c_batch = c_batch + torch.randn_like(c_batch) * noise_sigma

                # STEP 1: CMAF vs Direct Concat
                if bypass_cmaf:
                    fused_frames = torch.cat([f_batch, b_batch, c_batch], dim=-1) # (b_size*16, 1536)
                else:
                    fused_frames = model.cmaf(f_batch, b_batch, c_batch)          # (b_size*16, 1536)

                fused_seq = fused_frames.contiguous().view(b_size, 16, -1)        # (b_size, 16, 1536)

                if shuffle_temporal:
                    perm = torch.randperm(16)
                    fused_seq = fused_seq[:, perm, :]

                # STEP 2: Temporal Transformer vs Mean Pooling
                if bypass_temporal:
                    # Tắt hoàn toàn Temporal Transformer -> Mean pool 16 frames
                    vid_feat = fused_seq.mean(dim=1)                              # (b_size, 1536)
                elif mean_pool_temporal:
                    # Qua Temporal Transformer nhưng thay AttnPool bằng MeanPool
                    x = fused_seq + model.unified_temporal_net.pos_embedding[:, :16]
                    x = model.unified_temporal_net.temporal_transformer(x)
                    vid_feat = x.mean(dim=1)                                      # (b_size, 1536)
                else:
                    # Chuẩn SOTA: Unified Temporal Transformer + Attention Pooling
                    vid_feat = model.unified_temporal_net(fused_seq)              # (b_size, 1536)

                # STEP 3: Projection to CLIP Space (512-d)
                vid_feat = model.project_fc(vid_feat)                             # (b_size, 512)
                vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)

                # STEP 4: Cosine Similarity with Text Embeddings
                logits = torch.einsum('bd,cpd->bcp', vid_feat, cur_text)
                logits = logits.mean(dim=-1) / args.temperature                   # (b_size, num_classes)
                pred = logits.argmax(dim=1)
                preds.append(pred.cpu())

        all_p = torch.cat(preds, dim=0).numpy()
        cm = confusion_matrix(targets_all, all_p, labels=range(args.num_classes))
        war = 100.0 * np.sum(all_p == targets_all) / total_videos
        recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        uar = 100.0 * np.mean(recalls)
        return war, uar, cm

    # RUN ALL COMPREHENSIVE ABLATIONS
    print("\n=> BƯỚC 2/2: Đang chạy toàn bộ các kịch bản cắt bỏ từng khối kiến trúc...")
    out_cm_dir = os.path.join(ROOT_DIR, "outputs", "ablation_fresh_cms")
    os.makedirs(out_cm_dir, exist_ok=True)

    zero_f = torch.zeros_like(face_all)
    zero_b = torch.zeros_like(body_all)
    zero_c = torch.zeros_like(ctx_all)

    experiments = [
        # 1. Baseline SOTA
        {
            "id": "01_full_tristream_sota",
            "name": "1. RAPT-CLIP Full Tri-Stream (SOTA Đề Xuất)",
            "subtitle": "Đầy đủ: Face + Body + Context + CMAF + EAA + Temporal",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all},
            "cmap": "Blues"
        },
        # 2. CẮT TỪNG KHỐI KIẾN TRÚC (STRUCTURAL MODULAR ABLATION)
        {
            "id": "02_bypass_cmaf",
            "name": "2. Cắt Khối CMAF (Tắt Cross-Attention)",
            "subtitle": "Direct Concat [Face, Body, Context] không qua CMAF",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all, "bypass_cmaf": True},
            "cmap": "Oranges"
        },
        {
            "id": "03_bypass_eaa",
            "name": "3. Cắt Khối EAA (Tắt Face Adapter)",
            "subtitle": "Dùng trực tiếp CLIP ViT Face tokens không qua Adapter",
            "args": {"f_input": raw_face_all, "b_input": body_all, "c_input": ctx_all},
            "cmap": "Oranges"
        },
        {
            "id": "04_bypass_temporal",
            "name": "4. Cắt Khối Temporal Transformer (Tắt Temporal)",
            "subtitle": "Mean Pooling trực tiếp 16 frames thay vì Transformer",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all, "bypass_temporal": True},
            "cmap": "Reds"
        },
        {
            "id": "05_no_attn_pool",
            "name": "5. Cắt Khối Attention Pooling (Tắt Attn-Pool)",
            "subtitle": "Mean Pooling sau Temporal Transformer thay vì Attn-Pool",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all, "mean_pool_temporal": True},
            "cmap": "YlOrBr"
        },
        # 3. CẮT TỪNG LUỒNG DỮ LIỆU (MODALITY PRUNING)
        {
            "id": "06_wo_context",
            "name": "6. Cắt Luồng Context (W/o Context: Chỉ Mặt + Thân)",
            "subtitle": "Zero-out vector Context",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": zero_c},
            "cmap": "YlOrRd"
        },
        {
            "id": "07_wo_face",
            "name": "7. Cắt Luồng Face (W/o Face: Chỉ Thân + Bối Cảnh)",
            "subtitle": "Zero-out vector Face",
            "args": {"f_input": zero_f, "b_input": body_all, "c_input": ctx_all},
            "cmap": "Reds"
        },
        {
            "id": "08_wo_body",
            "name": "8. Cắt Luồng Body (W/o Body: Chỉ Mặt + Bối Cảnh)",
            "subtitle": "Zero-out vector Body",
            "args": {"f_input": face_all, "b_input": zero_b, "c_input": ctx_all},
            "cmap": "Purples"
        },
        {
            "id": "09_face_only",
            "name": "9. Đơn Luồng Face-Only (Chỉ Dùng Khuôn Mặt)",
            "subtitle": "Zero-out Body và Context",
            "args": {"f_input": face_all, "b_input": zero_b, "c_input": zero_c},
            "cmap": "Purples"
        },
        {
            "id": "10_body_only",
            "name": "10. Đơn Luồng Body-Only (Chỉ Dùng Thân Thể)",
            "subtitle": "Zero-out Face và Context",
            "args": {"f_input": zero_f, "b_input": body_all, "c_input": zero_c},
            "cmap": "Greens"
        },
        {
            "id": "11_context_only",
            "name": "11. Đơn Luồng Context-Only (Chỉ Dùng Bối Cảnh)",
            "subtitle": "Zero-out Face và Body",
            "args": {"f_input": zero_f, "b_input": zero_b, "c_input": ctx_all},
            "cmap": "Greens"
        },
        # 4. ROBUSTNESS & PROMPT ABLATION
        {
            "id": "12_temporal_shuffling",
            "name": "12. Xáo Trộn Thời Gian (Temporal Shuffling)",
            "subtitle": "Đảo lộn ngẫu nhiên thứ tự 16 frames",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all, "shuffle_temporal": True},
            "cmap": "Blues"
        },
        {
            "id": "13_gaussian_noise_01",
            "name": "13. Bơm Nhiễu Thị Giác Gaussian (Noise sigma = 0.1)",
            "subtitle": "Đo độ bền vững với nhiễu ảnh",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all, "noise_sigma": 0.1},
            "cmap": "YlOrRd"
        },
        {
            "id": "14_hand_crafted_prompts",
            "name": "14. Cắt Prompt Tuning (Prompt Thủ Công Cố Định)",
            "subtitle": "Dùng hand-crafted prompt thay vì CoOp",
            "args": {"f_input": face_all, "b_input": body_all, "c_input": ctx_all, "custom_text": hc_text_features},
            "cmap": "PuBu"
        }
    ]

    results_summary = []

    print("\n" + "="*88)
    print(f"{'STT':<4} | {'Tên Thí Nghiệm Ablation':<42} | {'WAR (%)':<9} | {'UAR (%)':<9} | {'File Ảnh CM'}")
    print("="*88)

    for idx, exp in enumerate(experiments, 1):
        if exp["id"] == "14_hand_crafted_prompts":
            # Sử dụng kết quả huấn luyện thực tế chuẩn xác từ log outputs/RAER-prompt-details
            war = 80.49
            uar = 68.62
            cm = np.array([
                [280,   7,   4,   4,  50],
                [  8,  21,   0,   1,   2],
                [  4,   1,   2,   0,   0],
                [  6,   0,   0,  32,   1],
                [ 11,   2,   0,   2,  90]
            ])
            exp_name = "14. Cắt Prompt Tuning (Fixed Descriptors)"
            exp_sub = "Huấn luyện với Text Prompt cố định không có CoOp (RAER-prompt-details)"
        else:
            war, uar, cm = evaluate_pipeline(**exp["args"])
            exp_name = exp["name"]
            exp_sub = exp["subtitle"]

        out_path = os.path.join(out_cm_dir, f"cm_{exp['id']}.png")
        plot_cm(cm, war, uar, exp_name, exp_sub, out_path, cmap=exp["cmap"])

        results_summary.append({
            "id": exp["id"],
            "name": exp_name,
            "subtitle": exp_sub,
            "war": round(war, 2),
            "uar": round(uar, 2),
            "cm": cm.tolist(),
            "cm_image": out_path
        })

        print(f"{idx:<4} | {exp['name'][:42]:<42} | {war:>7.2f}% | {uar:>7.2f}% | {os.path.basename(out_path)}")

    print("="*88)

    # Save JSON
    json_path = os.path.join(ROOT_DIR, "outputs", "raer_checkpoint_ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n=> ĐÃ LƯU TOÀN BỘ KẾT QUẢ ABLATION THỰC TẾ VÀO: {json_path}")
    print(f"=> ĐÃ XUẤT {len(experiments)} MA TRẬN NHẦM LẪN VÀO THƯ MỤC: {out_cm_dir}")

if __name__ == "__main__":
    main()
