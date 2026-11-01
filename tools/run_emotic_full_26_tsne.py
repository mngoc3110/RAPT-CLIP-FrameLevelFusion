#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_emotic_full_26_tsne.py
─────────────────────────────────
Trực quan hóa t-SNE TOÀN DIỆN CHO BỘ DỮ LIỆU EMOTIC:
- Đầy đủ 26 nhãn cảm xúc (Full 26 classes)
- Toàn bộ tập dữ liệu kiểm thử (5,292 samples test set)
- Xuất các biểu đồ chuẩn Q1 / IEEE với độ phân giải cao 300 DPI:
  1. outputs/EMOTIC/emotic_tsne_full_26_classes.png (.pdf)
  2. outputs/EMOTIC/emotic_tsne_full_26_annotated.png (.pdf)
  3. outputs/EMOTIC/emotic_tsne_affective_quadrants.png (.pdf)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.lines import Line2D
import seaborn as sns

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
    feat_path = os.path.join(OUTPUT_DIR, 'emotic_test_features_512.npy')
    emb_path = os.path.join(OUTPUT_DIR, 'emotic_full_tsne_2d.npy')
    targets_path = os.path.join(OUTPUT_DIR, 'test_targets.npy')
    preds_path = os.path.join(OUTPUT_DIR, 'test_preds.npy')

    feats = np.load(feat_path)
    emb_2d = np.load(emb_path)
    targets = np.load(targets_path)
    preds = np.load(preds_path)

    # Gán nhãn cảm xúc chủ đạo (Primary ground-truth active class with highest confidence)
    masked_preds = np.where(targets == 1.0, preds, -1e9)
    primary_labels = np.argmax(masked_preds, axis=-1)

    return feats, emb_2d, targets, preds, primary_labels

def plot_full_26_classes(emb_2d, labels):
    print("🎨 Đang vẽ đồ thị t-SNE Full 26 nhãn (5,292 mẫu)...")
    plt.figure(figsize=(15, 10.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    counts = np.bincount(labels, minlength=len(EMOTIC_CLASSES))

    # Vẽ từng lớp
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

    plt.title("Full 26-Class t-SNE Representation of RAPT-CLIP on EMOTIC Benchmark\n(Comprehensive 5,292 Test Set Sample Distribution)",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # Legend chia 3 cột đặt dưới hoặc bên cạnh
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

    out_png = os.path.join(OUTPUT_DIR, 'emotic_tsne_full_26_classes.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'emotic_tsne_full_26_classes.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {out_png}")

def plot_full_26_annotated(emb_2d, labels):
    print("🎨 Đang vẽ đồ thị t-SNE Full 26 nhãn có vị trí tâm cụm (Centroids)...")
    plt.figure(figsize=(15, 10.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    counts = np.bincount(labels, minlength=len(EMOTIC_CLASSES))

    # Vẽ điểm mẫu mờ phía sau
    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=COLORS_26[c_id],
            alpha=0.35,
            s=16,
            edgecolors='none',
            zorder=2
        )

    # Tính và vẽ tâm cụm (Centroid) cho 26 lớp
    centroids = []
    for c_id, c_name in enumerate(EMOTIC_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        cx = np.median(emb_2d[mask, 0])
        cy = np.median(emb_2d[mask, 1])
        centroids.append((c_id, cx, cy, c_name))

        # Điểm tâm lớn
        plt.scatter(cx, cy, c=COLORS_26[c_id], s=140, edgecolors='black', linewidths=1.5, zorder=5)

        # Nhãn text kèm viền trắng (shadow) để nổi bật
        txt = plt.text(cx, cy + 1.2, c_name, fontsize=8.5, fontweight='bold', color=COLORS_26[c_id],
                       ha='center', va='bottom', zorder=6)
        txt.set_path_effects([
            PathEffects.withStroke(linewidth=3, foreground='white')
        ])

    plt.title("Centroid Semantic Topology of 26 Affective Classes in RAPT-CLIP Latent Space\n(Class-Median Anchors with Shadow Outlines across 5,292 Test Samples)",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, zorder=1)

    out_png = os.path.join(OUTPUT_DIR, 'emotic_tsne_full_26_annotated.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'emotic_tsne_full_26_annotated.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {out_png}")

def plot_affective_quadrants(emb_2d, labels):
    print("🎨 Đang vẽ đồ thị t-SNE nhóm 5 vùng tâm lý học (Affective Super-Categories)...")

    # Phân nhóm theo mô hình tâm lý học cảm xúc EMOTIC (Kosti et al., TPAMI 2020)
    GROUPS = {
        'Positive / High Arousal (Happiness, Excitement, Pleasure, Anticipation, Confidence)': {
            'classes': ['Happiness', 'Excitement', 'Pleasure', 'Anticipation', 'Confidence'],
            'color': '#10B981' # Emerald Green
        },
        'Positive / Low Arousal (Affection, Esteem, Peace, Sympathy)': {
            'classes': ['Affection', 'Esteem', 'Peace', 'Sympathy'],
            'color': '#3B82F6' # Blue
        },
        'Negative / High Arousal (Anger, Annoyance, Fear, Disquietment, Surprise)': {
            'classes': ['Anger', 'Annoyance', 'Fear', 'Disquietment', 'Surprise'],
            'color': '#EF4444' # Red
        },
        'Negative / Low Arousal (Sadness, Suffering, Pain, Yearning, Embarrassment, Doubt/Confusion)': {
            'classes': ['Sadness', 'Suffering', 'Pain', 'Yearning', 'Embarrassment', 'Doubt/Confusion'],
            'color': '#8B5CF6' # Purple
        },
        'Behavioral & Engagement (Engagement, Disconnection, Disapproval, Aversion, Fatigue, Sensitivity)': {
            'classes': ['Engagement', 'Disconnection', 'Disapproval', 'Aversion', 'Fatigue', 'Sensitivity'],
            'color': '#F59E0B' # Amber
        }
    }

    plt.figure(figsize=(13.5, 9.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    for g_name, g_info in GROUPS.items():
        g_indices = [EMOTIC_CLASSES.index(c) for c in g_info['classes'] if c in EMOTIC_CLASSES]
        mask = np.isin(labels, g_indices)
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=g_info['color'],
            label=f"{g_name} (n={mask.sum()})",
            alpha=0.75,
            s=25,
            edgecolors='black',
            linewidths=0.25,
            zorder=2
        )

    plt.title("Affective Circumplex Cluster Decomposition of Full EMOTIC Test Set (5,292 Samples)\n(Mapped to Valence-Arousal Macro Dimensions: Positive, Negative, and Behavioral States)",
              fontsize=13.5, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5, zorder=1)

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=1,
        fontsize=9.2,
        frameon=True,
        framealpha=0.95
    )

    out_png = os.path.join(OUTPUT_DIR, 'emotic_tsne_affective_quadrants.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'emotic_tsne_affective_quadrants.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {out_png}")

def main():
    print("=" * 75)
    print(" 🚀 CHẠY TRỰC QUAN HÓA t-SNE EMOTIC FULL 26 NHÃN & TOÀN BỘ DATASET (5,292 MẪU)")
    print("=" * 75)

    feats, emb_2d, targets, preds, labels = load_data()
    print(f"📦 Dữ liệu nạp thành công: {feats.shape[0]} mẫu, {len(EMOTIC_CLASSES)} nhãn.")

    plot_full_26_classes(emb_2d, labels)
    plot_full_26_annotated(emb_2d, labels)
    plot_affective_quadrants(emb_2d, labels)

    print("\n🎉 HOÀN TẤT XUẤT CÁC BẢN t-SNE FULL 26 NHÃN CHO BÀI BÁO KHOA HỌC!")

if __name__ == '__main__':
    main()
