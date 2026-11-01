#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_raer_pretrained_tsne_circles.py
──────────────────────────────────────────
Vẽ lại không gian biểu diễn t-SNE của Pre-trained CLIP Features 3 LUỒNG
(Face + Body + Context) trên bộ dữ liệu RAER Test Set (528 video clips, 5 lớp cảm xúc):

Quy chuẩn yêu cầu:
1. Trích xuất đầy đủ 3 luồng: Face + Body + Context (1536-d).
2. 100% tất cả các nhãn đều dùng MARKER HÌNH TRÒN ('o').
3. Giữ nguyên màu sắc chuẩn tuyệt đối khớp 1:1 với ảnh RAPT-CLIP:
   - Neutrality  (n=345): #3b82f6 (Xanh lam)
   - Enjoyment   (n=32) : #f59e0b (Vàng hổ phách / Cam ấm)
   - Confusion   (n=7)  : #ef4444 (Đỏ san hô)
   - Fatigue     (n=39) : #06b6d4 (Cyan / Xanh lơ)
   - Distraction (n=105): #10b981 (Xanh ngọc lục bảo)
4. Tối ưu z-order: Vẽ lớp đa số trước (Neutrality) và đưa các lớp thiểu số
   (đặc biệt Confusion n=7) lên trên cùng để không bị che khuất.
5. Xuất định dạng PNG 300 DPI và vector PDF chất lượng cao.
"""

import os
import sys
import torch
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

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

# Màu sắc chuẩn đồng nhất tuyệt đối với bản Finetuned 3 luồng
CLASS_COLORS = {
    'Neutrality':  '#3b82f6',  # Xanh lam
    'Enjoyment':   '#f59e0b',  # Cam tươi / Vàng hổ phách
    'Confusion':   '#ef4444',  # Đỏ san hô
    'Fatigue':     '#06b6d4',  # Xanh lơ / Cyan
    'Distraction': '#10b981'   # Xanh ngọc lục bảo / Emerald
}

def load_raer_pretrained_3stream_features():
    feature_file = os.path.join(ROOT_DIR, 'outputs', 'raer_extracted_features.pt')
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Không tìm thấy file trích xuất đặc trưng: {feature_file}")
    
    print(f"=> Nạp đặc trưng RAER từ: {feature_file}")
    data = torch.load(feature_file, map_location='cpu', weights_only=False)
    raw_face = data['raw_face_all']  # (8448, 512)
    body = data['body_all']          # (8448, 512)
    ctx = data['ctx_all']            # (8448, 512) -> LUỒNG BỐI CẢNH (CONTEXT)
    labels = np.array(data['targets_all'])     # (528,)

    num_samples = len(labels)
    
    # Mean-pooling qua 16 frame cho từng luồng Face, Body, Context để đại diện cho toàn clip
    f_mean = raw_face.view(num_samples, 16, 512).mean(dim=1).numpy()
    b_mean = body.view(num_samples, 16, 512).mean(dim=1).numpy()
    c_mean = ctx.view(num_samples, 16, 512).mean(dim=1).numpy()

    # Chuẩn hóa L2 trước khi nối để 3 luồng đóng góp cân bằng
    f_norm = f_mean / np.linalg.norm(f_mean, axis=1, keepdims=True)
    b_norm = b_mean / np.linalg.norm(b_mean, axis=1, keepdims=True)
    c_norm = c_mean / np.linalg.norm(c_mean, axis=1, keepdims=True)

    # Ghép 3 nhánh: Face + Body + Context (528, 1536)
    concat_features = np.concatenate([f_norm, b_norm, c_norm], axis=1)
    print(f"=> Ma trận đặc trưng Pretrained CLIP 3 Luồng (Face + Body + Context): {concat_features.shape}")
    return concat_features, labels

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
    embeddings_2d = tsne.fit_transform(features)
    return embeddings_2d

def plot_raer_pretrained_tsne_circles(emb_2d, labels):
    print("=> Đang vẽ đồ thị t-SNE 3 luồng toàn bộ hình tròn cho Pre-trained CLIP trên RAER...")
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Grid nhẹ nhàng, thanh thoát
    ax.grid(True, linestyle='--', alpha=0.35, color='#d1d5db', zorder=1)
    ax.set_axisbelow(True)

    # Thứ tự vẽ: Vẽ lớp đông nhất trước (Neutrality), các lớp ít mẫu hơn vẽ sau để nổi bật
    plot_order = ['Neutrality', 'Distraction', 'Fatigue', 'Enjoyment', 'Confusion']

    point_sizes = {
        'Neutrality':  48,
        'Distraction': 52,
        'Fatigue':     55,
        'Enjoyment':   58,
        'Confusion':   68
    }

    for idx, cname in enumerate(plot_order):
        c_idx = CLASS_NAMES.index(cname)
        mask = (labels == c_idx)
        count = int(np.sum(mask))
        if count == 0:
            continue

        color = CLASS_COLORS[cname]
        zorder = 3 + idx * 2
        s = point_sizes[cname]

        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=color,
            marker='o',  # TOÀN BỘ DÙNG HÌNH TRÒN THEO YÊU CẦU
            label=f"{cname} (n={count})",
            alpha=0.85,
            s=s,
            edgecolors='white' if cname in ['Confusion', 'Enjoyment'] else 'none',
            linewidths=0.6 if cname in ['Confusion', 'Enjoyment'] else 0.0,
            zorder=zorder
        )

    # Sắp xếp lại Legend theo đúng thứ tự CLASS_NAMES
    handles, lbls = ax.get_legend_handles_labels()
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

    # Tiêu đề và nhãn trục chuẩn bài báo
    ax.set_title("t-SNE Feature Space: Pre-trained CLIP (Face+Body+Context) on RAER Test Set", fontsize=13, fontweight='bold', pad=14, color='#0f172a')
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='semibold', labelpad=8, color='#1e293b')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='semibold', labelpad=8, color='#1e293b')

    # Border spines
    for spine in ax.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(0.9)

    plt.tight_layout()

    # Đường dẫn xuất file
    out_paths = [
        os.path.join(OUTPUT_TSNE_DIR, 'raer_tsne_pretrained_circles.png'),
        os.path.join(OUTPUT_TSNE_DIR, 'raer_tsne_pretrained_circles.pdf'),
        os.path.join(OUTPUT_TSNE_DIR, 'raer_tsne_a.png'),
        os.path.join(IMAGE_DIR, 'tsne_pretrained.png')
    ]

    for p in out_paths:
        if p.endswith('.pdf'):
            plt.savefig(p, bbox_inches='tight')
        else:
            plt.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Đã lưu file: {p}")

    plt.close()

def main():
    print("=" * 80)
    print(" 🚀 VẼ LẠI t-SNE PRE-TRAINED CLIP 3 LUỒNG (FACE + BODY + CONTEXT)")
    print("=" * 80)
    
    features, labels = load_raer_pretrained_3stream_features()
    emb_2d = compute_tsne(features)
    plot_raer_pretrained_tsne_circles(emb_2d, labels)
    
    print("\n🎉 HOÀN TẤT VẼ t-SNE PRE-TRAINED 3 LUỒNG TOÀN BỘ HÌNH TRÒN!")

if __name__ == '__main__':
    main()
