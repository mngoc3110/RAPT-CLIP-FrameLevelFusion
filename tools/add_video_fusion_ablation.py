#!/usr/bin/env python3
"""
tools/add_video_fusion_ablation.py
Adds RAER Video-Level Fusion ablation to results and generates its confusion matrix.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

def plot_cm(cm, war, uar, title, subtitle, out_path, cmap="Purples"):
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

    full_title = f"{title}\nWAR: {war:.2f}% | UAR: {uar:.2f}% ({subtitle})"
    ax.set_title(full_title, fontsize=9.0, fontweight="bold", pad=10)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=9.0, fontweight="bold", labelpad=6)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=9.0, fontweight="bold", labelpad=6)
    plt.xticks(rotation=20, ha="right", fontsize=8.5)
    plt.yticks(rotation=0, fontsize=8.5)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"=> Saved CM: {out_path}")

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(root_dir, "outputs", "raer_checkpoint_ablation_results.json")
    cm_out = os.path.join(root_dir, "outputs", "ablation_fresh_cms", "cm_15_video_level_fusion.png")

    cm_data = np.array([
        [203, 15, 62, 16, 49],
        [  5, 25,  2,  0,  0],
        [  1,  1,  5,  0,  0],
        [  2,  1,  0, 36,  0],
        [ 21,  8,  1, 11, 64]
    ])

    war = 63.07
    uar = 72.33

    plot_cm(
        cm_data, war, uar,
        "15. RAER Video-Level Fusion",
        "Temporal Trước CMAF (Late Video Fusion)",
        cm_out,
        cmap="Purples"
    )

    video_entry = {
        "id": "15_video_level_fusion",
        "name": "15. RAER Video-Level Fusion (Temporal Trước CMAF)",
        "subtitle": "Temporal riêng từng luồng rồi mới chạy CMAF (Video-Level Late Fusion)",
        "war": war,
        "uar": uar,
        "cm": cm_data.tolist(),
        "cm_image": cm_out
    }

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Filter out if already present
    data = [x for x in data if x.get("id") != "15_video_level_fusion"]
    data.append(video_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("=> Successfully updated JSON with 15_video_level_fusion!")

if __name__ == "__main__":
    main()
