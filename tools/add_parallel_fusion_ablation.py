#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/add_parallel_fusion_ablation.py
─────────────────────────────────────
Thêm kết quả thực nghiệm Ablation "CMAF -> 3 nhánh song song" (Parallel Temporal Fusion)
từ Epoch 19 (528 test videos trên RAER) vào cơ sở dữ liệu ablation JSON
và tạo ảnh ma trận nhầm lẫn (Confusion Matrix) chuẩn hóa độ phân giải cao.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

def plot_cm(cm, war, uar, best_uar, title, subtitle, out_path, cmap="YlOrBr"):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=1.0,
        linecolor="#ffffff",
        ax=ax,
        annot_kws={"fontsize": 8.5, "weight": "bold"}
    )

    full_title = f"{title}\nWAR: {war:.2f}% | UAR: {uar:.2f}% (Best: {best_uar:.2f}%) — {subtitle}"
    ax.set_title(full_title, fontsize=8.8, fontweight="bold", pad=10)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=9.0, fontweight="bold", labelpad=6)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=9.0, fontweight="bold", labelpad=6)
    plt.xticks(rotation=20, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"=> Đã lưu Ma trận nhầm lẫn: {out_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(root_dir, "outputs", "raer_checkpoint_ablation_results.json")
    cm_out = os.path.join(root_dir, "outputs", "ablation_fresh_cms", "cm_16_parallel_temporal_fusion.png")

    # Ma trận nhầm lẫn Validation từ Epoch 19 của biến thể Parallel (3 nhánh song song)
    # Ground Truth row totals: Neutral (345), Enjoyment (32), Confusion (7), Fatigue (39), Distraction (105) = 528 videos
    cm_data = np.array([
        [238, 11, 23, 12, 61],
        [  1, 25,  0,  2,  4],
        [  2,  1,  4,  0,  0],
        [  3,  0,  0, 34,  2],
        [ 13,  3,  0,  6, 83]
    ], dtype=int)

    war = 72.73
    uar = 74.10
    best_uar = 75.16

    # Vẽ ma trận nhầm lẫn
    plot_cm(
        cm_data, war, uar, best_uar,
        "16. Parallel Temporal Fusion (3 Nhánh Song Song)",
        "Decoupled CMAF → 3 Temporal Riêng",
        cm_out,
        cmap="YlOrBr"
    )

    parallel_entry = {
        "id": "16_parallel_temporal_fusion",
        "name": "16. Parallel Temporal Fusion (CMAF → 3 Nhánh Song Song)",
        "subtitle": "CMAF Decoupled → 3 Nhánh Temporal Transformers Độc Lập → Concat Video-level",
        "war": war,
        "uar": uar,
        "best_uar": best_uar,
        "cm": cm_data.tolist(),
        "cm_image": cm_out
    }

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Cập nhật nếu đã tồn tại hoặc thêm mới
    data = [x for x in data if x.get("id") != "16_parallel_temporal_fusion"]
    data.append(parallel_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"=> Đã cập nhật file JSON {json_path} (Tổng cộng {len(data)} biến thể ablation).")

if __name__ == "__main__":
    main()
