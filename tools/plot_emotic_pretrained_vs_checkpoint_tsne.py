#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_emotic_pretrained_vs_checkpoint_tsne.py
──────────────────────────────────────────────────
So sánh đối chứng trực diện t-SNE:
(a) Pre-trained Baseline CLIP (ViT-B/16 nguyên bản của OpenAI)
(b) RAPT-CLIP Fine-tuned Checkpoint (model_best.zip)

Cho toàn bộ 26 nhãn cảm xúc trên toàn bộ 5,292 mẫu test set EMOTIC.
Xuất các file ấn phẩm Q1 (300 DPI PNG và vector PDF).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'EMOTIC')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 26 Lớp chuẩn của EMOTIC
EMOTIC_CLASSES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 
    'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

# Bảng 26 màu phân biệt cao (Perceptually Distinct Categorical Palette)
COLORS_26 = [
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', 
    '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#FABEBE', 
    '#008080', '#2563EB', '#9A6324', '#D97706', '#800000', 
    '#AAFFC3', '#10B981', '#DC2626', '#8B5CF6', '#EC4899', 
    '#7F1D1D', '#6366F1', '#B91C1C', '#14B8A6', '#059669', '#6B7280'
]

def load_all_data():
    emb_pre_path = os.path.join(OUTPUT_DIR, 'emotic_pretrained_tsne_2d.npy')
    emb_ckpt_path = os.path.join(OUTPUT_DIR, 'emotic_full_tsne_2d.npy')
    targets_path = os.path.join(OUTPUT_DIR, 'test_targets.npy')
    preds_path = os.path.join(OUTPUT_DIR, 'test_preds.npy')

    emb_pre = np.load(emb_pre_path)
    emb_ckpt = np.load(emb_ckpt_path)
    targets = np.load(targets_path)
    preds = np.load(preds_path)

    # Gán nhãn chủ đạo theo Ground Truth và xác suất mô hình
    masked_preds = np.where(targets == 1.0, preds, -1e9)
    primary_labels = np.argmax(masked_preds, axis=-1)

    return emb_pre, emb_ckpt, primary_labels

def plot_comparison_scatter(emb_pre, emb_ckpt, labels):
    print("🎨 Đang vẽ đồ thị so sánh đối chứng 2 bên (Full 26 Classes Scatter)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=300)
    fig.subplots_adjust(wspace=0.15, top=0.88, bottom=0.18, left=0.05, right=0.97)

    counts = np.bincount(labels, minlength=len(EMOTIC_CLASSES))

    # 1. PANEL A: PRE-TRAINED CLIP
    ax1 = axes[0]
    for c_id in range(len(EMOTIC_CLASSES)):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        ax1.scatter(
            emb_pre[mask, 0],
            emb_pre[mask, 1],
            c=COLORS_26[c_id],
            alpha=0.68,
            s=18,
            edgecolors='black',
            linewidths=0.2,
            zorder=2
        )
    ax1.set_title("(a) Pre-trained Baseline CLIP (Untouched ViT-B/16)\nSeverely Entangled & Unstructured Affective Manifold",
                  fontsize=12.5, fontweight='bold', pad=12)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # 2. PANEL B: RAPT-CLIP CHECKPOINT
    ax2 = axes[1]
    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        ax2.scatter(
            emb_ckpt[mask, 0],
            emb_ckpt[mask, 1],
            c=COLORS_26[c_id],
            label=f"{c_name} (n={counts[c_id]})",
            alpha=0.72,
            s=18,
            edgecolors='black',
            linewidths=0.2,
            zorder=2
        )
    ax2.set_title("(b) RAPT-CLIP (Fine-tuned Checkpoint with CMAF & MI/DC Loss)\nDistinct Structural Separation of Valence & Arousal Dimensions",
                  fontsize=12.5, fontweight='bold', pad=12)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # Chung 1 Legend dưới chân 2 biểu đồ
    fig.legend(
        loc='lower center',
        bbox_to_anchor=(0.51, 0.02),
        ncol=7,
        fontsize=8.2,
        frameon=True,
        framealpha=0.95,
        title="EMOTIC 26 Affective Categories (Sample Count across 5,292 Test Images)",
        title_fontsize=9.5
    )

    out_png = os.path.join(OUTPUT_DIR, 'emotic_tsne_pretrained_vs_checkpoint_26_classes.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'emotic_tsne_pretrained_vs_checkpoint_26_classes.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu so sánh: {out_png}")

def plot_comparison_centroids(emb_pre, emb_ckpt, labels):
    print("🎨 Đang vẽ đồ thị so sánh Centroids (Tâm cụm 26 nhãn) đối chứng...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5), dpi=300)
    fig.subplots_adjust(wspace=0.15, top=0.88, bottom=0.10, left=0.05, right=0.97)

    # 1. PANEL A: PRE-TRAINED CLIP CENTROIDS
    ax1 = axes[0]
    # Vẽ điểm mờ phía sau
    for c_id in range(len(EMOTIC_CLASSES)):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        ax1.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=COLORS_26[c_id], alpha=0.25, s=12, edgecolors='none', zorder=2)

    # Vẽ centroid
    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        cx = np.median(emb_pre[mask, 0])
        cy = np.median(emb_pre[mask, 1])
        ax1.scatter(cx, cy, c=COLORS_26[c_id], s=130, edgecolors='black', linewidths=1.5, zorder=5)
        txt = ax1.text(cx, cy + 1.2, c_name, fontsize=8, fontweight='bold', color=COLORS_26[c_id],
                       ha='center', va='bottom', zorder=6)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    ax1.set_title("(a) Pre-trained Baseline CLIP Centroid Topology\nCollapsing Centroids with High Overlap & Indistinguishable Affect",
                  fontsize=12.5, fontweight='bold', pad=12)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # 2. PANEL B: RAPT-CLIP CHECKPOINT CENTROIDS
    ax2 = axes[1]
    for c_id in range(len(EMOTIC_CLASSES)):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        ax2.scatter(emb_ckpt[mask, 0], emb_ckpt[mask, 1], c=COLORS_26[c_id], alpha=0.25, s=12, edgecolors='none', zorder=2)

    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        cx = np.median(emb_ckpt[mask, 0])
        cy = np.median(emb_ckpt[mask, 1])
        ax2.scatter(cx, cy, c=COLORS_26[c_id], s=130, edgecolors='black', linewidths=1.5, zorder=5)
        txt = ax2.text(cx, cy + 1.2, c_name, fontsize=8, fontweight='bold', color=COLORS_26[c_id],
                       ha='center', va='bottom', zorder=6)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    ax2.set_title("(b) RAPT-CLIP Checkpoint Centroid Topology\nExpanded Affective Coordinate Manifold with Explicit Valence Polarization",
                  fontsize=12.5, fontweight='bold', pad=12)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5, zorder=1)

    out_png = os.path.join(OUTPUT_DIR, 'emotic_tsne_pretrained_vs_checkpoint_centroids.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'emotic_tsne_pretrained_vs_checkpoint_centroids.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu so sánh Centroids: {out_png}")

def main():
    print("=" * 80)
    print(" 🚀 TẠO BỘ ĐỒ THỊ SO SÁNH t-SNE: PRE-TRAINED CLIP vs RAPT-CLIP (FULL 26 NHÃN)")
    print("=" * 80)
    emb_pre, emb_ckpt, labels = load_all_data()

    plot_comparison_scatter(emb_pre, emb_ckpt, labels)
    plot_comparison_centroids(emb_pre, emb_ckpt, labels)
    print("\n🎉 HOÀN TẤT BỘ ĐỒ THỊ ĐỐI CHỨNG t-SNE CHUẨN Q1!")

if __name__ == '__main__':
    main()
