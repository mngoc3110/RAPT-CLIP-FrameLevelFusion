#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_proof_visualization.py
──────────────────────────────────────
Tạo hình ảnh và biểu đồ khoa học chuẩn Q1 (IEEE/ACM style) làm BẰNG CHỨNG THỰC NGHIỆM:
Chứng minh việc có hay không có tên nhãn trong câu mô tả không làm thay đổi
không gian biểu diễn ngữ nghĩa của CLIP và hiệu năng của RAPT-CLIP.

Xuất ra:
- outputs/fig_proof_semantic_robustness.png
- outputs/fig_proof_semantic_robustness.pdf
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from models.Text import class_descriptor_5_only_face

CLASSES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

# 1. Câu gốc có nhãn
WITH_LABEL = class_descriptor_5_only_face

# 2. Câu bỏ hoàn toàn tên nhãn
NO_LABEL = [
    "A student with relaxed mouth, normally open eyes, smooth forehead, and calm facial muscles.",
    "A student with a slight smile, upturned lip corners, bright crinkled eyes, and raised cheeks.",
    "A student with furrowed lowered eyebrows, squinted perplexed eyes, and slightly parted mouth.",
    "A student with heavy drooping eyelids, slow blinking, frequent yawning, and slack jaw.",
    "A student with unfocused wandering gaze, head turned sideways, looking away from learning materials."
]

# Kết quả thực nghiệm đã chạy từ test_label_vs_nolabel_experiment.py
RECALL_WITH_LABEL = [65.8, 68.8, 42.9, 89.7, 79.0]
RECALL_NO_LABEL   = [66.4, 68.8, 42.9, 89.7, 78.1]

def main():
    device = 'cpu'
    print("=> Đang nạp mô hình CLIP ViT-B/16...")
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()

    with torch.no_grad():
        tok_a = clip.tokenize(WITH_LABEL).to(device)
        tok_b = clip.tokenize(NO_LABEL).to(device)
        feat_a = F.normalize(model.encode_text(tok_a).float(), dim=-1).numpy()
        feat_b = F.normalize(model.encode_text(tok_b).float(), dim=-1).numpy()

    # Ma trận tương đồng chéo (Cross Similarity Matrix) 5x5
    cross_sim = np.matmul(feat_a, feat_b.T)

    # PCA 2D để trực quan hóa hình học không gian
    all_feats = np.concatenate([feat_a, feat_b], axis=0) # 10 x 512
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_feats)
    coords_a = coords[:5]
    coords_b = coords[5:]

    # Cấu hình phong cách đồ họa IEEE/ACM
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9.5
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 10

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=300)
    fig.subplots_adjust(wspace=0.32, top=0.86, bottom=0.14, left=0.06, right=0.96)

    # ─────────────────────────────────────────────────────────────────
    # SUBPLOT 1: CROSS-COSINE SIMILARITY HEATMAP
    # ─────────────────────────────────────────────────────────────────
    ax1 = axes[0]
    sns.heatmap(
        cross_sim,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        ax=ax1,
        vmin=0.70,
        vmax=1.00,
        annot_kws={"size": 9, "weight": "bold"}
    )
    ax1.set_title("(a) Cross-Cosine Similarity Matrix\n(With-Label vs. No-Label Descriptors)", pad=10)
    ax1.set_xlabel("No-Label Descriptors (Pure Visual Cues)", fontweight='bold')
    ax1.set_ylabel("With-Label Descriptors (Original)", fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)
    ax1.tick_params(axis='y', rotation=0)

    # ─────────────────────────────────────────────────────────────────
    # SUBPLOT 2: 2D LATENT SPACE PROJECTION (PCA)
    # ─────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    palette = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']
    
    for i in range(5):
        # Vẽ điểm With-Label
        ax2.scatter(coords_a[i, 0], coords_a[i, 1], color=palette[i], marker='o', s=130, edgecolors='black', linewidth=1.2, zorder=4)
        # Vẽ điểm No-Label
        ax2.scatter(coords_b[i, 0], coords_b[i, 1], color=palette[i], marker='^', s=140, edgecolors='black', linewidth=1.2, zorder=4)
        # Vẽ đường nối giữa cặp cùng lớp
        ax2.plot([coords_a[i, 0], coords_b[i, 0]], [coords_a[i, 1], coords_b[i, 1]], color=palette[i], linestyle='--', linewidth=1.8, alpha=0.8, zorder=3)
        # Thêm text nhãn
        mid_x = (coords_a[i, 0] + coords_b[i, 0]) / 2
        mid_y = (coords_a[i, 1] + coords_b[i, 1]) / 2
        # Tinh chinh vi tri text de khong de len marker
        offset_y = 10 if CLASSES[i] != 'Enjoyment' else -18
        offset_x = 0 if CLASSES[i] != 'Neutral' else -15
        ax2.annotate(CLASSES[i], (mid_x, mid_y), textcoords="offset points", xytext=(offset_x, offset_y), ha='center', fontsize=9, fontweight='bold', color=palette[i])

    # Legend giả lập cho Marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='With Category Label', markerfacecolor='#475569', markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='^', color='w', label='Pure Visual Cues (No Label)', markerfacecolor='#475569', markeredgecolor='black', markersize=9),
        Line2D([0], [0], color='gray', linestyle='--', label='Intra-Class Mapping (Cosine Sim > 90%)')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8, frameon=True, framealpha=0.9)
    ax2.set_title("(b) 2D Latent Geometry Projection\n(Collinear Embedding Alignment)", pad=10)
    ax2.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontweight='bold')
    ax2.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.5)

    # ─────────────────────────────────────────────────────────────────
    # SUBPLOT 3: DOWNSTREAM RECALL COMPARISON (BAR CHART)
    # ─────────────────────────────────────────────────────────────────
    ax3 = axes[2]
    x = np.arange(len(CLASSES))
    width = 0.35

    rects1 = ax3.bar(x - width/2, RECALL_WITH_LABEL, width, label='With Label (UAR: 69.24%)', color='#3B82F6', edgecolor='black', linewidth=0.8)
    rects2 = ax3.bar(x + width/2, RECALL_NO_LABEL, width, label='No Label (UAR: 69.16%)', color='#10B981', edgecolor='black', linewidth=0.8)

    ax3.set_title("(c) Downstream Affect Recognition Recall\n(RAER 528 Video Test Set)", pad=10)
    ax3.set_ylabel("Per-Class Recall (%)", fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASSES, rotation=25, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', linestyle=':', alpha=0.6)
    ax3.legend(loc='lower right', fontsize=8.8, framealpha=0.9)

    # Thêm giá trị số dọc theo thân cột (chuẩn IEEE, hoàn toàn không bị đè chữ)
    for rect in rects1:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h / 2),
                     ha='center', va='center', rotation=90, color='white', fontweight='bold', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h / 2),
                     ha='center', va='center', rotation=90, color='white', fontweight='bold', fontsize=8)

    out_png = os.path.join(ROOT_DIR, "outputs", "fig_proof_semantic_robustness.png")
    out_pdf = os.path.join(ROOT_DIR, "outputs", "fig_proof_semantic_robustness.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã tạo thành công hình ảnh bằng chứng:")
    print(f"  • PNG: {out_png}")
    print(f"  • PDF: {out_pdf}")

if __name__ == "__main__":
    main()
