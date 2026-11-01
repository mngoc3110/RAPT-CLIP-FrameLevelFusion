#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/export_individual_emotic_tsne.py
──────────────────────────────────────
Xuất từng hình t-SNE ĐƠN LẺ (standalone individual figures) chuẩn LaTeX / IEEE:
1. emotic_tsne_pretrained_26_classes.png (.pdf)  : Bản phân tán Pre-trained CLIP (26 nhãn)
2. emotic_tsne_checkpoint_26_classes.png (.pdf)  : Bản phân tán RAPT-CLIP Checkpoint (26 nhãn)
3. emotic_tsne_pretrained_centroids.png (.pdf)   : Bản Centroids Pre-trained CLIP
4. emotic_tsne_checkpoint_centroids.png (.pdf)   : Bản Centroids RAPT-CLIP Checkpoint

Mỗi hình được thiết kế hoàn chỉnh với trục, tiêu đề, và chú thích riêng biệt (300 DPI).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

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

def load_data():
    emb_pre_path = os.path.join(OUTPUT_DIR, 'emotic_pretrained_tsne_2d.npy')
    emb_ckpt_path = os.path.join(OUTPUT_DIR, 'emotic_full_tsne_2d.npy')
    targets_path = os.path.join(OUTPUT_DIR, 'test_targets.npy')
    preds_path = os.path.join(OUTPUT_DIR, 'test_preds.npy')

    emb_pre = np.load(emb_pre_path)
    emb_ckpt = np.load(emb_ckpt_path)
    targets = np.load(targets_path)
    preds = np.load(preds_path)

    masked_preds = np.where(targets == 1.0, preds, -1e9)
    primary_labels = np.argmax(masked_preds, axis=-1)

    return emb_pre, emb_ckpt, primary_labels

def save_single_scatter(emb_2d, labels, title, filename_base):
    print(f"🎨 Đang xuất hình lẻ: {filename_base}...")
    plt.figure(figsize=(11, 9.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    counts = np.bincount(labels, minlength=len(EMOTIC_CLASSES))

    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=COLORS_26[c_id],
            label=f"{c_name} (n={counts[c_id]})",
            alpha=0.72,
            s=22,
            edgecolors='black',
            linewidths=0.25,
            zorder=2
        )

    plt.title(title, fontsize=12.5, fontweight='bold', pad=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, zorder=1)

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        title="EMOTIC 26 Affective Categories (Sample Count)",
        title_fontsize=9.5
    )

    out_png = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {out_png}")

def save_single_centroid(emb_2d, labels, title, filename_base):
    print(f"🎨 Đang xuất hình lẻ: {filename_base}...")
    plt.figure(figsize=(10.5, 9.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    # Vẽ điểm mờ làm nền
    for c_id in range(len(EMOTIC_CLASSES)):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=COLORS_26[c_id],
            alpha=0.25,
            s=14,
            edgecolors='none',
            zorder=2
        )

    # Vẽ các centroid
    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        cx = np.median(emb_2d[mask, 0])
        cy = np.median(emb_2d[mask, 1])

        plt.scatter(cx, cy, c=COLORS_26[c_id], s=140, edgecolors='black', linewidths=1.5, zorder=5)
        txt = plt.text(cx, cy + 1.2, c_name, fontsize=8.8, fontweight='bold', color=COLORS_26[c_id],
                       ha='center', va='bottom', zorder=6)
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=3.2, foreground='white')
        ])

    plt.title(title, fontsize=12.5, fontweight='bold', pad=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, zorder=1)

    out_png = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {out_png}")

def main():
    print("=" * 75)
    print(" 🚀 XUẤT CÁC HÌNH t-SNE ĐƠN LẺ (STANDALONE INDIVIDUAL FIGURES)")
    print("=" * 75)

    emb_pre, emb_ckpt, labels = load_data()

    # 1. Hình lẻ điểm tán xạ Pre-trained CLIP
    save_single_scatter(
        emb_pre, labels,
        "Pre-trained Baseline CLIP (Untouched ViT-B/16) - 26 Affective Classes\n(Severely Entangled & Unstructured Distribution across 5,292 Test Images)",
        "emotic_tsne_pretrained_26_classes"
    )

    # 2. Hình lẻ điểm tán xạ RAPT-CLIP Checkpoint
    save_single_scatter(
        emb_ckpt, labels,
        "RAPT-CLIP Fine-tuned Checkpoint - 26 Affective Classes\n(Structured Manifold with Cohesive Emotional Valence & Arousal Clustering)",
        "emotic_tsne_checkpoint_26_classes"
    )

    # 3. Hình lẻ Centroids Pre-trained CLIP
    save_single_centroid(
        emb_pre, labels,
        "Pre-trained Baseline CLIP Centroid Topology (26 Emotion Anchors)\n(Collapsing Centroids with Indistinguishable Emotional Valence)",
        "emotic_tsne_pretrained_centroids"
    )

    # 4. Hình lẻ Centroids RAPT-CLIP Checkpoint
    save_single_centroid(
        emb_ckpt, labels,
        "RAPT-CLIP Checkpoint Centroid Semantic Topology (26 Emotion Anchors)\n(Explicit Topological Polarization: Positive West vs Negative Northeast)",
        "emotic_tsne_checkpoint_centroids"
    )

    print("\n🎉 HOÀN TẤT XUẤT 4 HÌNH ĐƠN LẺ CHUẨN LATEX VÀO outputs/EMOTIC/!")

if __name__ == '__main__':
    main()
