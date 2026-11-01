#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_emotic_contrasting_tsne.py
──────────────────────────────────────
Trực quan hóa t-SNE CÁC NHÃN TƯƠNG PHẢN (Contrasting Affective Pairs: Positive vs Negative):
So sánh đối chứng trực diện giữa Pre-trained Baseline CLIP và RAPT-CLIP Checkpoint.

5 Cặp cảm xúc đối kháng kinh điển trong tâm lý học:
1. Happiness (Hạnh phúc)    vs  Sadness (Buồn bã)
2. Peace (Bình yên)         vs  Anger (Giận dữ)
3. Affection (Yêu thương)   vs  Aversion (Xa lánh/Ghê sợ)
4. Confidence (Tự tin)      vs  Fear (Sợ hãi)
5. Excitement (Hào hứng)    vs  Pain (Đau đớn)

Xuất ra cả hình lẻ độc lập (standalone) và hình ghép đối chứng:
- emotic_tsne_contrast_pretrained.png (.pdf)
- emotic_tsne_contrast_checkpoint.png (.pdf)
- emotic_tsne_contrast_comparison.png (.pdf)
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

# Danh sách 5 lớp tích cực và 5 lớp tiêu cực đối kháng
CLASSES_POS = ['Happiness', 'Peace', 'Affection', 'Confidence', 'Excitement']
CLASSES_NEG = ['Sadness', 'Anger', 'Aversion', 'Fear', 'Pain']
ALL_CONTRAST_CLASSES = CLASSES_POS + CLASSES_NEG

# Màu sắc phân biệt rõ rệt cho 10 lớp cảm xúc tương phản:
# 5 Tích cực (Bright & Vibrant) vs 5 Tiêu cực (Distinct Cool/Warm Tones)
COLORS_POS = {
    'Happiness':  '#10B981',  # Xanh lục bảo
    'Peace':      '#3B82F6',  # Xanh lam
    'Affection':  '#EC4899',  # Hồng sen
    'Confidence': '#06B6D4',  # Xanh cyan / ngọc
    'Excitement': '#F59E0B'   # Vàng cam hổ phách
}

COLORS_NEG = {
    'Sadness':    '#64748B',  # Xám đá u buồn (Slate Grey)
    'Anger':      '#DC2626',  # Đỏ tươi
    'Aversion':   '#92400E',  # Nâu đồng
    'Fear':       '#7C3AED',  # Tím đậm
    'Pain':       '#881337'   # Đỏ mận thẫm (Wine/Burgundy)
}

def load_contrast_data():
    emb_pre = np.load(os.path.join(OUTPUT_DIR, 'contrast_tsne_pretrained.npy'))
    emb_ckpt = np.load(os.path.join(OUTPUT_DIR, 'contrast_tsne_checkpoint.npy'))
    labels = np.load(os.path.join(OUTPUT_DIR, 'contrast_labels.npy'))
    return emb_pre, emb_ckpt, labels

def plot_single_contrast(emb_2d, labels, title, filename_base, is_checkpoint=False):
    print(f"🎨 Đang vẽ hình lẻ: {filename_base}...")
    fig, ax = plt.subplots(figsize=(10.5, 9.5), dpi=300)
    plt.rcParams['font.family'] = 'sans-serif'

    # 1. Vẽ các điểm mẫu (TOÀN BỘ DÙNG HÌNH TRÒN 'o')
    for c_id, c_name in enumerate(ALL_CONTRAST_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue

        c = COLORS_POS[c_name] if c_name in CLASSES_POS else COLORS_NEG[c_name]

        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=c,
            marker='o',
            label=f"{c_name} (n={mask.sum()})",
            alpha=0.82,
            s=65,
            edgecolors='black',
            linewidths=0.6,
            zorder=3
        )

    # 2. Thêm Text Centroid trực tiếp để làm nổi bật vị trí cụm
    for c_id, c_name in enumerate(ALL_CONTRAST_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0:
            continue
        cx = np.median(emb_2d[mask, 0])
        cy = np.median(emb_2d[mask, 1])
        c = COLORS_POS[c_name] if c_name in CLASSES_POS else COLORS_NEG[c_name]

        # Centroid marker (Circle 'o')
        ax.scatter(cx, cy, c=c, marker='o', s=180, edgecolors='black', linewidths=1.8, zorder=6)

        # Centroid label with white halo
        offset_y = 1.6 if not is_checkpoint else (1.8 if c_name not in ['Affection', 'Peace'] else -2.2)
        txt = ax.text(cx, cy + offset_y, c_name, fontsize=9.5, fontweight='bold', color=c,
                      ha='center', va='bottom' if offset_y > 0 else 'top', zorder=7)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3.5, foreground='white')])

    ax.set_title(title, fontsize=12.5, fontweight='bold', pad=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # Legend chia 5 cột (Hàng 1: Positive, Hàng 2: Negative)
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.09),
        ncol=5,
        fontsize=8.8,
        frameon=True,
        framealpha=0.95,
        title="Contrasting Affective Classes (Top Row: Positive Valence | Bottom Row: Negative Valence)",
        title_fontsize=9.8
    )

    out_png = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {out_png}")

def plot_dual_contrast_comparison(emb_pre, emb_ckpt, labels):
    print("🎨 Đang vẽ đồ thị ghép 2 bên so sánh đối chứng (Dual Panel)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9.5), dpi=300)
    fig.subplots_adjust(wspace=0.16, top=0.88, bottom=0.17, left=0.05, right=0.97)

    # ─────────────────────────────────────────────────────────────────
    # PANEL A: PRE-TRAINED CLIP
    # ─────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    for c_id, c_name in enumerate(ALL_CONTRAST_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        c = COLORS_POS[c_name] if c_name in CLASSES_POS else COLORS_NEG[c_name]
        ax1.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=c, marker='o',
                    alpha=0.8, s=55, edgecolors='black', linewidths=0.5, zorder=3)

    # Centroids Panel A
    for c_id, c_name in enumerate(ALL_CONTRAST_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        cx = np.median(emb_pre[mask, 0])
        cy = np.median(emb_pre[mask, 1])
        c = COLORS_POS[c_name] if c_name in CLASSES_POS else COLORS_NEG[c_name]
        ax1.scatter(cx, cy, c=c, marker='o', s=160, edgecolors='black', linewidths=1.6, zorder=6)
        txt = ax1.text(cx, cy + 2.8, c_name, fontsize=8.8, fontweight='bold', color=c,
                       ha='center', va='bottom', zorder=7)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3.2, foreground='white')])

    ax1.set_title("(a) Pre-trained Baseline CLIP (Untouched ViT-B/16)\nEntangled Manifold: Positive & Negative Emotions Randomly Overlapping",
                  fontsize=12.5, fontweight='bold', pad=12)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # ─────────────────────────────────────────────────────────────────
    # PANEL B: RAPT-CLIP CHECKPOINT
    # ─────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    for c_id, c_name in enumerate(ALL_CONTRAST_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        c = COLORS_POS[c_name] if c_name in CLASSES_POS else COLORS_NEG[c_name]
        ax2.scatter(emb_ckpt[mask, 0], emb_ckpt[mask, 1], c=c, marker='o',
                    alpha=0.8, s=55, edgecolors='black', linewidths=0.5, zorder=3)

    # Centroids Panel B
    for c_id, c_name in enumerate(ALL_CONTRAST_CLASSES):
        mask = (labels == c_id)
        if mask.sum() == 0: continue
        cx = np.median(emb_ckpt[mask, 0])
        cy = np.median(emb_ckpt[mask, 1])
        c = COLORS_POS[c_name] if c_name in CLASSES_POS else COLORS_NEG[c_name]
        ax2.scatter(cx, cy, c=c, marker='o', s=160, edgecolors='black', linewidths=1.6, zorder=6)

        offset_y = 1.6 if c_name not in ['Affection', 'Peace'] else -2.2
        txt = ax2.text(cx, cy + offset_y, c_name, fontsize=8.8, fontweight='bold', color=c,
                       ha='center', va='bottom' if offset_y > 0 else 'top', zorder=7)
        txt.set_path_effects([PathEffects.withStroke(linewidth=3.2, foreground='white')])

    ax2.set_title("(b) RAPT-CLIP Fine-tuned Model (CMAF + MI/DC Regularization)\nPolarized Affective Space: Distinct Semantic Island Separation",
                  fontsize=12.5, fontweight='bold', pad=12)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5, zorder=1)

    # Legend chung đặt bên dưới (toàn bộ là hình tròn 'o')
    legend_elements_pos = [
        Line2D([0], [0], marker='o', color='w', label=f"{c}", markerfacecolor=COLORS_POS[c], markeredgecolor='black', markersize=9)
        for c in CLASSES_POS
    ]
    legend_elements_neg = [
        Line2D([0], [0], marker='o', color='w', label=f"{c}", markerfacecolor=COLORS_NEG[c], markeredgecolor='black', markersize=9)
        for c in CLASSES_NEG
    ]

    fig.legend(
        handles=legend_elements_pos + legend_elements_neg,
        loc='lower center',
        bbox_to_anchor=(0.51, 0.02),
        ncol=5,
        fontsize=8.8,
        frameon=True,
        framealpha=0.95,
        title="Contrasting Affective Classes (Top Row: Positive Valence | Bottom Row: Negative Valence)",
        title_fontsize=9.8
    )

    out_png = os.path.join(OUTPUT_DIR, 'emotic_tsne_contrast_comparison.png')
    out_pdf = os.path.join(OUTPUT_DIR, 'emotic_tsne_contrast_comparison.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu đồ thị ghép đối chứng: {out_png}")

def main():
    print("=" * 80)
    print(" 🚀 TẠO BỘ ĐỒ THỊ t-SNE CÁC NHÃN TƯƠNG PHẢN (CONTRASTING PAIRS: POSITIVE vs NEGATIVE)")
    print("=" * 80)

    emb_pre, emb_ckpt, labels = load_contrast_data()

    # 1. Hình lẻ Pre-trained CLIP
    plot_single_contrast(
        emb_pre, labels,
        "Pre-trained Baseline CLIP (Untouched ViT-B/16) - Contrasting Valence Pairs\n(Severe Intra-Space Overlap: Positive and Negative Affect Intertwined)",
        "emotic_tsne_contrast_pretrained",
        is_checkpoint=False
    )

    # 2. Hình lẻ RAPT-CLIP Checkpoint
    plot_single_contrast(
        emb_ckpt, labels,
        "RAPT-CLIP Checkpoint - Contrasting Valence Pairs\n(Explicit Valence Polarization & Clear Inter-Cluster Semantic Boundaries)",
        "emotic_tsne_contrast_checkpoint",
        is_checkpoint=True
    )

    # 3. Hình ghép 2 bên đối chứng
    plot_dual_contrast_comparison(emb_pre, emb_ckpt, labels)

    print("\n🎉 HOÀN TẤT TOÀN BỘ BỘ ĐỒ THỊ t-SNE CÁC NHÃN TƯƠNG PHẢN!")

if __name__ == '__main__':
    main()
