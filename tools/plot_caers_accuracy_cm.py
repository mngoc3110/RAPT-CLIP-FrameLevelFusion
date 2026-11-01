#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_caers_accuracy_cm.py
───────────────────────────────
Vẽ Ma Trận Nhầm Lẫn (Confusion Matrix) theo chuẩn Accuracy (Độ chính xác / WAR)
cho checkpoint CAER-S mới nhất (Epoch 7: 91.80% Accuracy).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Validation Confusion Matrix trích xuất từ Epoch 7 (7,000 mẫu, 1,000 mẫu/lớp)
CM_VALID = np.array([
    [929,   5,   1,   6,  29,  18,  12],  # Anger
    [  1, 990,   3,   1,   1,   2,   2],  # Disgust
    [  0,   0, 997,   1,   1,   1,   0],  # Fear
    [ 19,   3,   3, 879,  66,  10,  20],  # Happy
    [ 43,  23,   6,  57, 767,  40,  64],  # Neutral
    [ 11,   3,   2,   7,  19, 946,  12],  # Sad
    [ 18,   4,   4,  17,  26,  13, 918],  # Surprise
], dtype=int)


def plot_accuracy_confusion_matrix_grid(output_path):
    """
    Vẽ Confusion Matrix gồm 2 phần liền mạch:
    - Trục chính: Ma trận 7x7 hiển thị số lượng mẫu nhầm lẫn (Counts)
    - Trục phụ (cột phải): Cột Class Accuracy (%) tương ứng với từng hàng.
    """
    total_samples = CM_VALID.sum()
    correct_samples = np.trace(CM_VALID)
    overall_acc = (correct_samples / total_samples) * 100.0  # 91.80%

    row_sums = CM_VALID.sum(axis=1)
    class_acc = (CM_VALID.diagonal() / row_sums) * 100.0  # Shape: (7,)
    class_acc_col = class_acc[:, np.newaxis]  # Shape: (7, 1)

    fig = plt.figure(figsize=(10.5, 7.8), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[15, 2.2, 0.8], wspace=0.08)

    ax_cm = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[0, 1], sharey=ax_cm)
    cax = fig.add_subplot(gs[0, 2])

    cmap = sns.color_palette("Blues", as_cmap=True)

    # 1. Main 7x7 Heatmap
    sns.heatmap(
        CM_VALID,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        cbar_ax=cax,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1.5,
        linecolor="#ffffff",
        ax=ax_cm,
        annot_kws={"fontsize": 11.5, "weight": "bold"}
    )
    cax.set_ylabel("Sample Count", fontsize=11, fontweight="bold", labelpad=8)

    ax_cm.set_xlabel("Predicted Emotion Label", fontsize=12, fontweight="bold", labelpad=10)
    ax_cm.set_ylabel("Ground Truth Emotion Label", fontsize=12, fontweight="bold", labelpad=10)
    ax_cm.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=11, fontweight="bold")
    ax_cm.set_yticklabels(CLASSES, rotation=0, fontsize=11, fontweight="bold")

    # 2. Right Column: Class Accuracy (%)
    sns.heatmap(
        class_acc_col,
        annot=np.array([[f"{acc:.1f}%"] for acc in class_acc]),
        fmt="",
        cmap=sns.light_palette("#1565c0", as_cmap=True),
        cbar=False,
        xticklabels=["Accuracy"],
        yticklabels=[],
        linewidths=1.5,
        linecolor="#ffffff",
        ax=ax_acc,
        vmin=60,
        vmax=100,
        annot_kws={"fontsize": 11.5, "weight": "bold", "color": "#0d47a1"}
    )
    ax_acc.set_xticklabels(["Class Acc"], rotation=0, fontsize=11, fontweight="bold", color="#0d47a1")
    ax_acc.tick_params(left=False, bottom=False)

    # Tiêu đề chung cho cả figure
    fig.suptitle(
        f"CAER-S Validation Confusion Matrix (RAPT-CLIP SOTA: {overall_acc:.2f}% Accuracy)\n"
        f"Overall Accuracy (WAR): {overall_acc:.2f}% ({correct_samples:,} / {total_samples:,} Correct Samples)",
        fontsize=13,
        fontweight="bold",
        color="#0b3c5d",
        y=0.98
    )

    plt.subplots_adjust(top=0.90, bottom=0.12, left=0.10, right=0.93)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Confusion Matrix 2 cột (Counts + Class Acc): {output_path}")


def plot_normalized_percent_matrix(output_path):
    """
    Vẽ Confusion Matrix chuẩn hóa hoàn toàn theo Accuracy Tỉ lệ phần trăm (0 - 100%)
    Mỗi ô hiển thị: [Số lượng mẫu] và [Tỉ lệ % Accuracy].
    """
    total_samples = CM_VALID.sum()
    correct_samples = np.trace(CM_VALID)
    overall_acc = (correct_samples / total_samples) * 100.0

    row_sums = CM_VALID.sum(axis=1, keepdims=True)
    cm_percent = (CM_VALID.astype(float) / row_sums) * 100.0

    fig, ax = plt.subplots(figsize=(9.4, 7.8), dpi=300)

    annot = np.empty_like(CM_VALID, dtype=object)
    for i in range(CM_VALID.shape[0]):
        for j in range(CM_VALID.shape[1]):
            annot[i, j] = f"{CM_VALID[i, j]}\n({cm_percent[i, j]:.1f}%)"

    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1.5,
        linecolor="#ffffff",
        ax=ax,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Accuracy Rate (%)', 'shrink': 0.88},
        annot_kws={"fontsize": 10.0, "weight": "bold"}
    )

    title = (
        f"CAER-S Normalized Accuracy Confusion Matrix (RAPT-CLIP)\n"
        f"Overall Accuracy: {overall_acc:.2f}% | Top-1 Class Accuracy Diag [%]"
    )
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=16, color="#0b3c5d")
    ax.set_xlabel("Predicted Emotion Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Ground Truth Emotion Label", fontsize=12, fontweight="bold", labelpad=10)

    plt.xticks(rotation=30, ha="right", fontsize=11, fontweight="bold")
    plt.yticks(rotation=0, fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Normalized Confusion Matrix (Accuracy %): {output_path}")


def plot_accuracy_barchart(output_path):
    """
    Vẽ biểu đồ thanh Per-Class Accuracy so sánh giữa Baseline cũ (90.40%) và Checkpoint mới (91.80%).
    """
    old_acc = np.array([92.3, 98.1, 99.7, 83.4, 73.7, 93.4, 92.2])
    new_acc = (CM_VALID.diagonal() / CM_VALID.sum(axis=1)) * 100.0

    x = np.arange(len(CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=300)

    rects1 = ax.bar(x - width/2, old_acc, width, label='Baseline Trước (90.40% Accuracy)', color='#90caf9', edgecolor='#1565c0')
    rects2 = ax.bar(x + width/2, new_acc, width, label='Mô Hình Mới - Epoch 7 (91.80% Accuracy)', color='#1e88e5', edgecolor='#0d47a1')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('CAER-S Per-Class Accuracy Comparison (Baseline 90.40% vs New Best 91.80%)', fontsize=13, fontweight='bold', pad=14, color='#0b3c5d')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11, fontweight='bold')
    ax.set_ylim(60, 105)
    ax.legend(fontsize=10.5, loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Thêm giá trị trên từng cột
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0d47a1')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu Accuracy Barchart: {output_path}")


if __name__ == "__main__":
    out_dir = "outputs/CAER-S"
    
    # 1. Confusion Matrix chuẩn SOTA Accuracy (thay thế file cũ)
    plot_accuracy_confusion_matrix_grid(os.path.join(out_dir, "caer_s_confusion_matrix.png"))
    
    # 2. File riêng biệt với tên _accuracy
    plot_accuracy_confusion_matrix_grid(os.path.join(out_dir, "caer_s_confusion_matrix_accuracy.png"))
    
    # 3. Bản chuẩn hóa từng ô theo % Accuracy
    plot_normalized_percent_matrix(os.path.join(out_dir, "caer_s_confusion_matrix_percent_accuracy.png"))
    
    # 4. Biểu đồ thanh Per-Class Accuracy
    plot_accuracy_barchart(os.path.join(out_dir, "caer_s_accuracy_barchart.png"))

    # Cập nhật ảnh vào thư mục image/ để hiển thị trên README
    plot_accuracy_confusion_matrix_grid("image/caer_s_confusion_matrix_accuracy.png")
