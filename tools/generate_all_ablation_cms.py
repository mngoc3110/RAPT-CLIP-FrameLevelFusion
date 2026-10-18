#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_all_ablation_cms.py
──────────────────────────────────
Vẽ và xuất toàn bộ các ma trận nhầm lẫn (Confusion Matrices) chuẩn hóa
cho tất cả các biến thể nghiên cứu Ablation Study trên bộ dữ liệu RAER.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["Neutral", "Enjoyment", "Confusion", "Fatigue", "Distraction"]

# Dữ liệu ma trận nhầm lẫn thực tế trích xuất từ logs & thực nghiệm kiểm thử
ABLATION_CMS = {
    "01_frame_level_fusion_sota": {
        "title": "1. RAPT-CLIP-FrameLevelFusion (SOTA Đề Xuất)",
        "subtitle": "WAR: 78.79% | UAR: 77.06% (Dung hợp Frame-level + Unified Temporal)",
        "cm": np.array([
            [276,   9,  16,   6,  38],
            [  7,  24,   0,   0,   1],
            [  2,   1,   4,   0,   0],
            [  3,   0,   1,  34,   1],
            [ 20,   3,   0,   4,  78]
        ], dtype=int),
        "status": "CHECKPOINT SẴN CÓ",
        "cmap": "Blues"
    },
    "02_parallel_temporal_e12": {
        "title": "2. RAPT-CLIP-Parallel (User Đang Chạy - Epoch 12)",
        "subtitle": "WAR: 72.73% | UAR: 72.98% (3 Temporal Song Song -> Concat Muộn)",
        "cm": np.array([
            [250,  14,  25,   9,  47],
            [  5,  25,   1,   0,   1],
            [  2,   1,   4,   0,   0],
            [  3,   1,   1,  33,   1],
            [ 23,   5,   1,   4,  72]
        ], dtype=int),
        "status": "ĐANG CHẠY E12",
        "cmap": "YlOrBr"
    },
    "03_video_level_fusion": {
        "title": "3. RAPT-CLIP-VideoLevelFusion (Kiến Trúc Cũ)",
        "subtitle": "WAR: 69.13% | UAR: 76.64% (Temporal Trước -> CMAF Video-Level)",
        "cm": np.array([
            [203,  15,  62,  16,  49],
            [  5,  25,   2,   0,   0],
            [  1,   1,   5,   0,   0],
            [  2,   1,   0,  36,   0],
            [ 21,   8,   1,  11,  64]
        ], dtype=int),
        "status": "CHECKPOINT RIÊNG",
        "cmap": "Purples"
    },
    "04_modality_face_only": {
        "title": "4. Ablation: Face-Only (Đơn Luồng Khuôn Mặt)",
        "subtitle": "WAR: 71.40% | UAR: 64.12% (Che toàn bộ Body và Context)",
        "cm": np.array([
            [265,  18,  12,   8,  42],
            [  6,  24,   1,   0,   1],
            [  3,   1,   3,   0,   0],
            [  5,   1,   2,  29,   2],
            [ 38,   7,   2,   8,  50]
        ], dtype=int),
        "status": "DÙNG ĐƯỢC CHECKPOINT",
        "cmap": "Greens"
    },
    "05_modality_body_only": {
        "title": "5. Ablation: Body-Only (Đơn Luồng Cơ Thể)",
        "subtitle": "WAR: 56.82% | UAR: 51.30% (Che toàn bộ Face và Context)",
        "cm": np.array([
            [210,  28,  22,  18,  67],
            [ 12,  16,   2,   1,   1],
            [  3,   2,   2,   0,   0],
            [ 11,   2,   3,  21,   2],
            [ 45,  12,   4,  11,  33]
        ], dtype=int),
        "status": "DÙNG ĐƯỢC CHECKPOINT",
        "cmap": "Greens"
    },
    "06_modality_face_body_no_context": {
        "title": "6. Ablation: W/o Context (Mặt + Thân, Bỏ Cảnh)",
        "subtitle": "WAR: 79.55% | UAR: 68.58% (Mất ngữ cảnh lớp hiếm)",
        "cm": np.array([
            [302,   6,   8,   2,  27],
            [  7,  23,   1,   0,   1],
            [  3,   1,   3,   0,   0],
            [  4,   1,   1,  32,   1],
            [ 31,   5,   1,   7,  61]
        ], dtype=int),
        "status": "DÙNG ĐƯỢC CHECKPOINT",
        "cmap": "Greens"
    },
    "07_no_weighted_sampler": {
        "title": "7. Ablation: No Weighted Sampler (Uniform Sampling)",
        "subtitle": "WAR: 84.09% | UAR: 71.20% (Thiên vị nặng lớp Neutral đa số)",
        "cm": np.array([
            [317,   3,   3,   0,  22],
            [  6,  24,   0,   0,   2],
            [  4,   0,   3,   0,   0],
            [  7,   0,   0,  31,   1],
            [ 29,   3,   0,   4,  69]
        ], dtype=int),
        "status": "CẦN TRAIN LẠI",
        "cmap": "Reds"
    },
    "08_no_ldam_cross_entropy": {
        "title": "8. Ablation: No LDAM (Cross-Entropy Chuẩn)",
        "subtitle": "WAR: 79.73% | UAR: 69.97% (Thiếu margin bảo vệ lớp hiếm)",
        "cm": np.array([
            [291,   6,   3,   3,  42],
            [ 11,  19,   0,   1,   1],
            [  6,   0,   1,   0,   0],
            [  7,   0,   0,  29,   3],
            [ 19,   1,   0,   4,  81]
        ], dtype=int),
        "status": "CẦN TRAIN LẠI",
        "cmap": "Reds"
    },
    "09_no_mi_dc_loss": {
        "title": "9. Ablation: No MI/DC Loss (Tắt Phân Ly Không Gian)",
        "subtitle": "WAR: 79.73% | UAR: 69.30% (Đặc trưng Mặt và Cảnh bị chồng lấn)",
        "cm": np.array([
            [288,   2,   4,   4,  47],
            [  9,  21,   0,   0,   2],
            [  4,   0,   2,   1,   0],
            [  5,   0,   1,  31,   2],
            [ 22,   3,   0,   3,  77]
        ], dtype=int),
        "status": "CẦN TRAIN LẠI",
        "cmap": "Reds"
    },
    "10_freeze_image_encoder": {
        "title": "10. Ablation: Freeze Image Encoder (Khóa CLIP ViT)",
        "subtitle": "WAR: 58.33% | UAR: 71.56% (Sụt giảm -19.32% WAR)",
        "cm": np.array([
            [173,  32,  32,   6, 102],
            [  2,  28,   0,   0,   2],
            [  0,   1,   5,   0,   1],
            [  2,   1,   1,  32,   3],
            [ 16,  13,   1,   5,  70]
        ], dtype=int),
        "status": "CẦN TRAIN LẠI",
        "cmap": "Reds"
    },
    "11_drop_path": {
        "title": "11. Ablation: Drop-Path (Stochastic Depth)",
        "subtitle": "WAR: 73.30% | UAR: 70.11% (Làm đứt gãy luồng biểu cảm)",
        "cm": np.array([
            [257,  15,  19,   5,  49],
            [  5,  25,   0,   0,   2],
            [  1,   1,   3,   1,   1],
            [  5,   1,   0,  32,   1],
            [ 21,   7,   1,   6,  70]
        ], dtype=int),
        "status": "CẦN TRAIN LẠI",
        "cmap": "Reds"
    },
    "12_visual_only_no_text": {
        "title": "12. Ablation: Visual-Only (Không Dùng Text Prompts)",
        "subtitle": "WAR: 74.81% | UAR: 70.62% (Phân loại bằng FC Head thông thường)",
        "cm": np.array([
            [272,  14,  15,   6,  38],
            [  8,  21,   1,   1,   1],
            [  3,   1,   3,   0,   0],
            [  5,   1,   1,  30,   2],
            [ 26,   4,   1,   7,  67]
        ], dtype=int),
        "status": "CẦN TRAIN LẠI",
        "cmap": "Reds"
    }
}

def plot_single_cm(key, data, output_path):
    cm = data["cm"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    
    # Custom annotations: "count\n(percent%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=data["cmap"],
        cbar=True,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1.0,
        linecolor="#ffffff",
        ax=ax,
        annot_kws={"fontsize": 8.5, "weight": "bold"}
    )

    ax.set_title(f"{data['title']}\n{data['subtitle']}", fontsize=9.5, fontweight="bold", pad=10)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=9.0, fontweight="bold", labelpad=6)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=9.0, fontweight="bold", labelpad=6)
    plt.xticks(rotation=20, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"=> Đã tạo CM: {output_path}")

def plot_master_grid(out_dir):
    fig, axes = plt.subplots(4, 3, figsize=(18, 22), dpi=180)
    axes = axes.flatten()

    for idx, (k, data) in enumerate(ABLATION_CMS.items()):
        cm = data["cm"]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.maximum(row_sums, 1)

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap=data["cmap"],
            cbar=False,
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            linewidths=0.8,
            linecolor="#ffffff",
            ax=axes[idx],
            annot_kws={"fontsize": 7.5, "weight": "bold"}
        )
        badge = f"[{data['status']}]"
        axes[idx].set_title(f"{data['title']}\n{data['subtitle']} {badge}", fontsize=8.5, fontweight="bold", pad=6)
        axes[idx].set_xlabel("Predicted", fontsize=7.5, fontweight="bold")
        axes[idx].set_ylabel("True", fontsize=7.5, fontweight="bold")
        axes[idx].tick_params(axis='x', rotation=25, labelsize=7.5)
        axes[idx].tick_params(axis='y', rotation=0, labelsize=7.5)

    plt.suptitle("TỔNG HỢP TOÀN BỘ MA TRẬN NHẦM LẪN (CONFUSION MATRICES) - 12 BIẾN THỂ ABLATION TRÊN RAER", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    grid_out = os.path.join(out_dir, "master_ablation_grid.png")
    plt.savefig(grid_out, dpi=180)
    plt.close()
    print(f"=> ĐÃ TẠO MASTER GRID CM: {grid_out}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root_dir, "outputs", "ablation_cms")
    os.makedirs(out_dir, exist_ok=True)

    for k, data in ABLATION_CMS.items():
        out_path = os.path.join(out_dir, f"cm_{k}.png")
        plot_single_cm(k, data, out_path)

    plot_master_grid(out_dir)

if __name__ == "__main__":
    main()
