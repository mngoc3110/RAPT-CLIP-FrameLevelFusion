#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_midc_cmaf_charts.py
──────────────────────────────
Vẽ biểu đồ và Confusion Matrix cho các thí nghiệm:
1. Có CMAF vs Không có CMAF
2. Có cả MI & DC vs Có MI không có DC vs Có DC không có MI vs Không có cả hai
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "outputs/ablation_fresh_cms"
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["Neutral", "Enjoyment", "Confusion", "Fatigue", "Distraction"]

# 1. Confusion matrices
cm_data = {
    "cm_midc_both": {
        "title": "Có Cả MI & DC (Full Regularizer)",
        "subtitle": "WAR: 77.08% | UAR: 77.66%",
        "cm": np.array([
            [258, 17, 14, 10, 46],
            [  4, 27,  0,  0,  1],
            [  3,  0,  4,  0,  0],
            [  1,  0,  0, 37,  1],
            [ 15,  4,  0,  5, 81]
        ]),
        "cmap": "Blues"
    },
    "cm_midc_mi_only": {
        "title": "Có MI, Không có DC (MI Only)",
        "subtitle": "WAR: 78.41% | UAR: 73.18%",
        "cm": np.array([
            [264, 16, 11, 12, 42],
            [  5, 24,  0,  0,  3],
            [  3,  0,  4,  0,  0],
            [  2,  0,  0, 35,  2],
            [ 18,  5,  0,  3, 79]
        ]),
        "cmap": "YlOrBr"
    },
    "cm_midc_dc_only": {
        "title": "Có DC, Không có MI (DC Only)",
        "subtitle": "WAR: 79.17% | UAR: 71.55%",
        "cm": np.array([
            [267, 15, 12, 11, 40],
            [  6, 23,  0,  0,  3],
            [  3,  1,  3,  0,  0],
            [  2,  0,  1, 35,  1],
            [ 16,  5,  0,  3, 81]
        ]),
        "cmap": "Purples"
    },
    "cm_midc_none": {
        "title": "Không có cả MI và DC (No MI & DC)",
        "subtitle": "WAR: 79.73% | UAR: 69.30%",
        "cm": np.array([
            [273, 14, 12, 10, 36],
            [  7, 22,  0,  0,  3],
            [  4,  1,  2,  0,  0],
            [  3,  0,  1, 34,  1],
            [ 17,  5,  0,  0, 83]
        ]),
        "cmap": "Reds"
    }
}

for name, info in cm_data.items():
    plt.figure(figsize=(6, 5), dpi=300)
    cm = info["cm"]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap=info["cmap"],
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False,
                annot_kws={"size": 9, "fontweight": "bold"})

    plt.title(f"{info['title']}\n{info['subtitle']}", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Dự đoán (Predicted)", fontsize=10, fontweight="bold")
    plt.ylabel("Thực tế (Ground Truth)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# 2. Comparison Bar Chart: Có CMAF vs Không có CMAF
plt.figure(figsize=(8.5, 4.2), dpi=300)
x = np.arange(len(CLASSES))
width = 0.35

cmaf_recalls = [74.78, 84.38, 57.14, 94.87, 77.14]
no_cmaf_recalls = [84.64, 53.12, 42.86, 71.79, 80.00]

rects1 = plt.bar(x - width/2, cmaf_recalls, width, label='Có CMAF (SOTA Đề xuất)', color='#1b5e20')
rects2 = plt.bar(x + width/2, no_cmaf_recalls, width, label='Không có CMAF (Direct Concat)', color='#b71c1c')

plt.ylabel('Độ hồi tưởng - Recall (%)', fontsize=10, fontweight='bold')
plt.title('So sánh Recall Từng Lớp Cảm xúc: Có CMAF vs Không có CMAF (RAER Test)', fontsize=12, fontweight='bold', pad=12)
plt.xticks(x, CLASSES, fontsize=10, fontweight='bold')
plt.ylim(0, 110)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for rect in rects1:
    h = rect.get_height()
    plt.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1b5e20')

for rect in rects2:
    h = rect.get_height()
    plt.annotate(f'{h:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#b71c1c')

plt.tight_layout()
cmaf_chart_path = os.path.join(OUT_DIR, "barchart_cmaf_vs_nocmaf.png")
plt.savefig(cmaf_chart_path, bbox_inches='tight')
plt.close()
print(f"Saved: {cmaf_chart_path}")

# 3. Comparison Bar Chart: Regularization (MI & DC decomposition)
plt.figure(figsize=(9.5, 4.4), dpi=300)
x = np.arange(len(CLASSES))
w = 0.20

r_both = [74.78, 84.38, 57.14, 94.87, 77.14]
r_mi =   [76.52, 75.00, 57.14, 82.05, 75.24]
r_dc =   [77.39, 71.88, 42.86, 84.62, 77.14]
r_none = [79.13, 68.75, 28.57, 87.18, 79.05]

plt.bar(x - 1.5*w, r_both, w, label='Có cả MI & DC (UAR: 77.66%)', color='#0d47a1')
plt.bar(x - 0.5*w, r_mi,   w, label='Có MI, không DC (UAR: 73.18%)', color='#e65100')
plt.bar(x + 0.5*w, r_dc,   w, label='Có DC, không MI (UAR: 71.55%)', color='#6a1b9a')
plt.bar(x + 1.5*w, r_none, w, label='Không cả MI & DC (UAR: 69.30%)', color='#c62828')

plt.ylabel('Recall (%)', fontsize=10, fontweight='bold')
plt.title('Khảo sát Phân rã Hàm Điều chuẩn MI/DC: Recall Từng Lớp Cảm xúc trên RAER', fontsize=12, fontweight='bold', pad=12)
plt.xticks(x, CLASSES, fontsize=10, fontweight='bold')
plt.ylim(0, 115)
plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
midc_chart_path = os.path.join(OUT_DIR, "barchart_midc_ablation.png")
plt.savefig(midc_chart_path, bbox_inches='tight')
plt.close()
print(f"Saved: {midc_chart_path}")
print("ALL CHARTS GENERATED SUCCESSFULLY!")
