#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_emotic_tsne.py
────────────────────────
Trực quan hóa t-SNE cho mô hình RAPT-CLIP trên tập kiểm thử EMOTIC (5,292 ảnh).
Vì EMOTIC có 26 nhãn (đa nhãn), kịch bản này chọn lọc các nhãn tiêu biểu và chia thành 2 bản:
1. Bản Tích Cực (Positive Valence): Happiness, Peace, Affection, Confidence, Excitement
2. Bản Tiêu Cực (Negative Valence): Sadness, Anger, Fear, Pain, Aversion

Xuất ra 3 file chất lượng cao (300 DPI):
- outputs/EMOTIC/emotic_tsne_positive.png (và .pdf)
- outputs/EMOTIC/emotic_tsne_negative.png (và .pdf)
- outputs/EMOTIC/emotic_tsne_dual_valence.png (và .pdf)
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import get_class_info, build_model
from utils.checkpoint_utils import load_slim_checkpoint
from utils.builders import build_dataloaders

# 26 Lớp chuẩn của EMOTIC
EMOTIC_CLASSES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 
    'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

# Các nhãn tiêu biểu được chọn lọc
POS_CLASSES = ['Happiness', 'Affection', 'Peace', 'Excitement', 'Confidence']
NEG_CLASSES = ['Sadness', 'Anger', 'Fear', 'Pain', 'Aversion']

POS_INDICES = [EMOTIC_CLASSES.index(c) for c in POS_CLASSES]
NEG_INDICES = [EMOTIC_CLASSES.index(c) for c in NEG_CLASSES]

def extract_or_load_features(args, device, cache_feat_path):
    if os.path.exists(cache_feat_path):
        print(f"📦 Đang nạp đặc trưng 512-D đã lưu: {cache_feat_path}...")
        return np.load(cache_feat_path)

    print("🚀 Đang khởi tạo mô hình RAPT-CLIP và nạp checkpoint EMOTIC...")
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text).to(device)
    load_slim_checkpoint(model, args.eval_checkpoint, device=device)
    model.eval()

    _, _, test_loader = build_dataloaders(args)
    all_features = []

    print(f"⚡ Đang trích xuất đặc trưng 512-D cho toàn bộ {len(test_loader.dataset)} mẫu test EMOTIC...")
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Trích xuất Visual Features]"):
            face, body, context, target = [x.to(device) if x is not None else None for x in batch]
            _ = model(face, body, context)
            feat = model.last_video_features.cpu().numpy() # [B, 512]
            all_features.append(feat)

    all_features = np.concatenate(all_features, axis=0) # [5292, 512]
    np.save(cache_feat_path, all_features)
    print(f"✅ Đã lưu {all_features.shape} đặc trưng vào {cache_feat_path} trong {time.time()-t0:.1f}s!")
    return all_features

def select_dominant_samples(features, targets, preds, class_indices, max_per_class=120):
    """
    Chọn lọc các mẫu có nhãn ưu thế rõ rệt trong nhóm cảm xúc để t-SNE không bị chồng lấn.
    """
    selected_indices = []
    selected_labels = []

    for label_id, c_idx in enumerate(class_indices):
        # Điều kiện: Ground truth có nhãn này
        has_gt = (targets[:, c_idx] == 1.0)
        gt_sample_indices = np.where(has_gt)[0]

        if len(gt_sample_indices) == 0:
            continue

        # Sắp xếp theo xác suất dự đoán cao nhất của lớp đó để chọn các mẫu tiêu biểu nhất
        sample_probs = preds[gt_sample_indices, c_idx]
        sorted_order = np.argsort(-sample_probs)
        chosen = gt_sample_indices[sorted_order[:max_per_class]]

        selected_indices.extend(chosen)
        selected_labels.extend([label_id] * len(chosen))

    selected_indices = np.array(selected_indices)
    selected_labels = np.array(selected_labels)
    selected_features = features[selected_indices]
    return selected_features, selected_labels

def plot_single_tsne(emb_2d, labels, class_names, title, out_png, out_pdf, colors, palette_name="bright"):
    plt.figure(figsize=(7.5, 6.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    for label_id, c_name in enumerate(class_names):
        mask = (labels == label_id)
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[colors[label_id]],
            label=c_name,
            alpha=0.85,
            s=45,
            edgecolors='black',
            linewidths=0.5
        )

    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=10, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='best', fontsize=9.5, frameon=True, framealpha=0.9)
    plt.tight_layout()

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu t-SNE: {out_png}")

def plot_dual_tsne(emb_pos, labels_pos, emb_neg, labels_neg, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), dpi=300)
    fig.subplots_adjust(wspace=0.22, top=0.88, bottom=0.12, left=0.06, right=0.96)

    # 1. BẢN TÍCH CỰC
    ax1 = axes[0]
    pos_colors = ['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899']
    for label_id, c_name in enumerate(POS_CLASSES):
        mask = (labels_pos == label_id)
        ax1.scatter(
            emb_pos[mask, 0],
            emb_pos[mask, 1],
            c=[pos_colors[label_id]],
            label=c_name,
            alpha=0.85,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    ax1.set_title("(a) Positive Affective Valence Clusters\n(Happiness, Affection, Peace, Excitement, Confidence)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight='bold')
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='best', fontsize=9, frameon=True, framealpha=0.9)

    # 2. BẢN TIÊU CỰC
    ax2 = axes[1]
    neg_colors = ['#EF4444', '#DC2626', '#991B1B', '#D97706', '#6B7280']
    for label_id, c_name in enumerate(NEG_CLASSES):
        mask = (labels_neg == label_id)
        ax2.scatter(
            emb_neg[mask, 0],
            emb_neg[mask, 1],
            c=[neg_colors[label_id]],
            label=c_name,
            alpha=0.85,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )
    ax2.set_title("(b) Negative Affective Valence Clusters\n(Sadness, Anger, Fear, Pain, Aversion)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight='bold')
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='best', fontsize=9, frameon=True, framealpha=0.9)

    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu Dual t-SNE: {out_png}")

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"⚡ Thiết bị chạy t-SNE EMOTIC: {device}")

    output_dir = os.path.join(ROOT_DIR, 'outputs', 'EMOTIC')
    os.makedirs(output_dir, exist_ok=True)

    cache_feat_path = os.path.join(output_dir, 'emotic_test_features_512.npy')
    preds_path = os.path.join(output_dir, 'test_preds.npy')
    targets_path = os.path.join(output_dir, 'test_targets.npy')

    if not os.path.exists(preds_path) or not os.path.exists(targets_path):
        print(f"❌ Không tìm thấy preds hoặc targets trong {output_dir}")
        return

    preds = np.load(preds_path)     # [5292, 26]
    targets = np.load(targets_path) # [5292, 26]

    args = argparse.Namespace(
        dataset='EMOTIC',
        mode='eval',
        clip_path='ViT-B/16',
        use_npy=True,
        npy_dir='emotic_dataset/emotic_pre',
        batch_size=128,
        workers=2,
        seed=42,
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
        eval_checkpoint=os.path.join(output_dir, 'model_best.zip'),
        root_dir='./',
        train_annotation='emotic_dataset/train_bbox.txt',
        val_annotation='emotic_dataset/val_bbox.txt',
        test_annotation='emotic_dataset/test_bbox.txt',
        bounding_box_face='emotic_dataset/emotic_face_bboxes.json',
        bounding_box_body='emotic_dataset/emotic_body_bboxes.json',
        output_path=output_dir
    )

    # 1. Trích xuất hoặc nạp đặc trưng 512-D
    features = extract_or_load_features(args, device, cache_feat_path)

    # 2. Chọn lọc mẫu tiêu biểu cho Nhóm Tích Cực
    print("\n🔍 Đang chọn lọc mẫu cho Nhóm Tích Cực...")
    pos_feats, pos_labels = select_dominant_samples(features, targets, preds, POS_INDICES, max_per_class=100)
    print(f"  • Tổng số mẫu tích cực: {len(pos_labels)}")

    # 3. Chọn lọc mẫu tiêu biểu cho Nhóm Tiêu Cực
    print("🔍 Đang chọn lọc mẫu cho Nhóm Tiêu Cực...")
    neg_feats, neg_labels = select_dominant_samples(features, targets, preds, NEG_INDICES, max_per_class=100)
    print(f"  • Tổng số mẫu tiêu cực: {len(neg_labels)}")

    # 4. Huấn luyện t-SNE
    print("\n🎨 Đang chạy t-SNE cho Nhóm Tích Cực...")
    tsne_pos = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_pos = tsne_pos.fit_transform(pos_feats)

    print("🎨 Đang chạy t-SNE cho Nhóm Tiêu Cực...")
    tsne_neg = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_neg = tsne_neg.fit_transform(neg_feats)

    # 5. Xuất các đồ thị
    out_pos_png = os.path.join(output_dir, "emotic_tsne_positive.png")
    out_pos_pdf = os.path.join(output_dir, "emotic_tsne_positive.pdf")
    pos_colors = ['#10B981', '#3B82F6', '#8B5CF6', '#F59E0B', '#EC4899']
    plot_single_tsne(emb_pos, pos_labels, POS_CLASSES, "t-SNE Representation of Positive Affective States (EMOTIC)", out_pos_png, out_pos_pdf, pos_colors)

    out_neg_png = os.path.join(output_dir, "emotic_tsne_negative.png")
    out_neg_pdf = os.path.join(output_dir, "emotic_tsne_negative.pdf")
    neg_colors = ['#EF4444', '#DC2626', '#991B1B', '#D97706', '#6B7280']
    plot_single_tsne(emb_neg, neg_labels, NEG_CLASSES, "t-SNE Representation of Negative Affective States (EMOTIC)", out_neg_png, out_neg_pdf, neg_colors)

    out_dual_png = os.path.join(output_dir, "emotic_tsne_dual_valence.png")
    out_dual_pdf = os.path.join(output_dir, "emotic_tsne_dual_valence.pdf")
    plot_dual_tsne(emb_pos, pos_labels, emb_neg, neg_labels, out_dual_png, out_dual_pdf)

    print("\n🎉 HOÀN THÀNH TOÀN BỘ CÁC BẢN t-SNE CHO EMOTIC!")

if __name__ == "__main__":
    main()
