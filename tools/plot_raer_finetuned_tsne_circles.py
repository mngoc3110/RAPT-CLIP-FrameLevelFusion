#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_raer_finetuned_tsne_circles.py
──────────────────────────────────────────
Vẽ lại đồ thị t-SNE của mô hình RAPT-CLIP 3-Stream (Face + Body + Context)
từ checkpoint SOTA: outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth
trên tập kiểm thử RAER Test Set (528 video clips, 5 lớp cảm xúc):

Quy chuẩn yêu cầu:
1. 100% tất cả các nhãn đều dùng MARKER HÌNH TRÒN ('o').
2. Chuẩn bảng màu đồng nhất chính xác theo yêu cầu người dùng:
   - Neutrality  (n=345): #3b82f6 (Blue)
   - Enjoyment   (n=32) : #f59e0b (Amber/Orange)
   - Confusion   (n=7)  : #ef4444 (Coral Red)
   - Fatigue     (n=39) : #06b6d4 (Cyan)
   - Distraction (n=105): #10b981 (Emerald Green)
3. Tối ưu z-order: Vẽ lớp đa số trước (Neutrality) và đưa các lớp thiểu số
   (đặc biệt Confusion n=7) lên trên cùng để không bị che khuất.
4. Xuất cả ảnh PNG độ phân giải cao 300 DPI và vector PDF cho bản thảo bài báo.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_TSNE_DIR = os.path.join(ROOT_DIR, 'outputs', 'TSNE')
IMAGE_DIR = os.path.join(ROOT_DIR, 'image')
os.makedirs(OUTPUT_TSNE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# 5 lớp cảm xúc học đường RAER
CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

# Bảng màu chuẩn tuyệt đối khớp với ảnh người dùng cung cấp
CLASS_COLORS = {
    'Neutrality':  '#3b82f6',  # Xanh lam
    'Enjoyment':   '#f59e0b',  # Cam tươi / Vàng hổ phách
    'Confusion':   '#ef4444',  # Đỏ san hô
    'Fatigue':     '#06b6d4',  # Xanh lơ / Cyan
    'Distraction': '#10b981'   # Xanh ngọc lục bảo / Emerald
}

def load_raer_3stream_features():
    feat_path = os.path.join(OUTPUT_TSNE_DIR, 'extracted_features.npy')
    lbl_path = os.path.join(OUTPUT_TSNE_DIR, 'extracted_labels.npy')
    
    if not os.path.exists(feat_path) or not os.path.exists(lbl_path):
        raise FileNotFoundError(f"Không tìm thấy file vector đặc trưng 3 luồng tại: {feat_path}")
        
    print(f"=> Nạp đặc trưng 3-Stream (Face + Body + Context) từ: {feat_path}")
    features = np.load(feat_path)
    labels = np.load(lbl_path)
    print(f"=> Kích thước ma trận đặc trưng: {features.shape}, Nhãn: {labels.shape}")
    return features, labels

def compute_tsne(features):
    print("=> Tính toán tọa độ t-SNE 2D (perplexity=30, random_state=42, max_iter=1000)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    emb_2d = tsne.fit_transform(features)
    return emb_2d

def plot_tsne(emb_2d, labels, title_text=None):
    print("=> Đang vẽ đồ thị t-SNE toàn bộ hình tròn (All Circles)...")
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Grid nhẹ nhàng, hiện đại
    ax.grid(True, linestyle='--', alpha=0.35, color='#d1d5db', zorder=1)
    ax.set_axisbelow(True)

    # Thứ tự vẽ để các lớp ít mẫu (Confusion n=7, Enjoyment n=32) nổi rõ lên trên
    plot_order = ['Neutrality', 'Distraction', 'Fatigue', 'Enjoyment', 'Confusion']
    
    # Kích thước chấm: Confusion cho to hơn nhẹ (s=60) với viền nhẹ để không bị chìm
    point_sizes = {
        'Neutrality':  48,
        'Distraction': 52,
        'Fatigue':     55,
        'Enjoyment':   58,
        'Confusion':   68
    }

    scatter_handles = []
    scatter_labels = []

    for idx, cname in enumerate(plot_order):
        c_idx = CLASS_NAMES.index(cname)
        mask = (labels == c_idx)
        count = int(np.sum(mask))
        if count == 0:
            continue
            
        color = CLASS_COLORS[cname]
        zorder = 3 + idx * 2
        s = point_sizes[cname]

        # Toàn bộ đều dùng marker='o' (HÌNH TRÒN)
        sc = ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=color,
            marker='o',
            s=s,
            alpha=0.85,
            edgecolors='white' if cname in ['Confusion', 'Enjoyment'] else 'none',
            linewidths=0.6 if cname in ['Confusion', 'Enjoyment'] else 0.0,
            zorder=zorder,
            label=f"{cname} (n={count})"
        )

    # Sắp xếp lại thứ tự Legend theo danh sách lớp chuẩn
    handles, lbls = ax.get_legend_handles_labels()
    # Sort handles by CLASS_NAMES
    sorted_pairs = sorted(zip(handles, lbls), key=lambda pair: [CLASS_NAMES.index(c) for c in CLASS_NAMES if pair[1].startswith(c)][0])
    sorted_handles = [p[0] for p in sorted_pairs]
    sorted_labels = [p[1] for p in sorted_pairs]

    legend = ax.legend(
        sorted_handles, sorted_labels,
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        facecolor='#ffffff',
        edgecolor='#cbd5e1',
        fontsize=10.5,
        shadow=True,
        borderpad=0.8,
        labelspacing=0.5
    )
    legend.set_zorder(20)

    # Tiêu đề và nhãn trục
    title = title_text if title_text else "t-SNE Feature Space: RAPT-CLIP-FrameLevelFusion on RAER Test Set"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='semibold', labelpad=8, color='#1e293b')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='semibold', labelpad=8, color='#1e293b')

    # Border spines
    for spine in ax.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(0.9)

    plt.tight_layout()

    # Xuất ra nhiều định dạng và đường dẫn phục vụ bài báo
    out_paths = [
        os.path.join(OUTPUT_TSNE_DIR, 'raer_tsne_finetuned_3stream_circles.png'),
        os.path.join(OUTPUT_TSNE_DIR, 'raer_tsne_finetuned_3stream_circles.pdf'),
        os.path.join(OUTPUT_TSNE_DIR, 'raer_tsne_b.png'),  # Cập nhật panel (b) cho bài báo
        os.path.join(IMAGE_DIR, 'tsne_finetuned.png')
    ]

    for p in out_paths:
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu: {p}")

    plt.close()

def main():
    features, labels = load_raer_3stream_features()
    emb_2d = compute_tsne(features)
    plot_tsne(emb_2d, labels)
    print("\n🎉 Hoàn thành vẽ lại t-SNE 3 luồng với 100% marker hình tròn chuẩn màu!")

if __name__ == '__main__':
    main()

