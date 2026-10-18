import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    output_dir = 'outputs/EMOTIC'
    
    # 26 emotion class results
    data = [
        {"idx": 0, "name": "Affection", "val_ap": 60.58, "test_ap": 47.84, "f1": 38.12, "thresh": 0.43},
        {"idx": 1, "name": "Anger", "val_ap": 51.30, "test_ap": 18.79, "f1": 26.57, "thresh": 0.57},
        {"idx": 2, "name": "Annoyance", "val_ap": 40.51, "test_ap": 27.33, "f1": 34.24, "thresh": 0.55},
        {"idx": 3, "name": "Anticipation", "val_ap": 96.55, "test_ap": 59.22, "f1": 61.36, "thresh": 0.37},
        {"idx": 4, "name": "Aversion", "val_ap": 31.79, "test_ap": 11.21, "f1": 20.45, "thresh": 0.55},
        {"idx": 5, "name": "Confidence", "val_ap": 85.49, "test_ap": 81.77, "f1": 69.98, "thresh": 0.35},
        {"idx": 6, "name": "Disapproval", "val_ap": 46.11, "test_ap": 22.22, "f1": 34.77, "thresh": 0.53},
        {"idx": 7, "name": "Disconnection", "val_ap": 57.99, "test_ap": 38.41, "f1": 40.77, "thresh": 0.43},
        {"idx": 8, "name": "Disquietment", "val_ap": 32.58, "test_ap": 26.79, "f1": 31.87, "thresh": 0.45},
        {"idx": 9, "name": "Doubt/Confusion", "val_ap": 27.70, "test_ap": 23.00, "f1": 23.75, "thresh": 0.51},
        {"idx": 10, "name": "Embarrassment", "val_ap": 8.60, "test_ap": 2.54, "f1": 6.17, "thresh": 0.51},
        {"idx": 11, "name": "Engagement", "val_ap": 98.55, "test_ap": 87.80, "f1": 85.96, "thresh": 0.41},
        {"idx": 12, "name": "Esteem", "val_ap": 27.76, "test_ap": 16.26, "f1": 25.40, "thresh": 0.39},
        {"idx": 13, "name": "Excitement", "val_ap": 86.38, "test_ap": 77.23, "f1": 72.71, "thresh": 0.45},
        {"idx": 14, "name": "Fatigue", "val_ap": 35.05, "test_ap": 24.54, "f1": 24.14, "thresh": 0.59},
        {"idx": 15, "name": "Fear", "val_ap": 11.73, "test_ap": 12.36, "f1": 18.86, "thresh": 0.50},
        {"idx": 16, "name": "Happiness", "val_ap": 90.14, "test_ap": 81.50, "f1": 62.64, "thresh": 0.38},
        {"idx": 17, "name": "Pain", "val_ap": 45.12, "test_ap": 17.26, "f1": 22.76, "thresh": 0.54},
        {"idx": 18, "name": "Peace", "val_ap": 36.06, "test_ap": 32.87, "f1": 36.87, "thresh": 0.44},
        {"idx": 19, "name": "Pleasure", "val_ap": 64.23, "test_ap": 57.34, "f1": 56.32, "thresh": 0.46},
        {"idx": 20, "name": "Sadness", "val_ap": 42.74, "test_ap": 38.19, "f1": 45.95, "thresh": 0.56},
        {"idx": 21, "name": "Sensitivity", "val_ap": 16.85, "test_ap": 13.93, "f1": 18.40, "thresh": 0.47},
        {"idx": 22, "name": "Suffering", "val_ap": 52.26, "test_ap": 43.65, "f1": 49.10, "thresh": 0.55},
        {"idx": 23, "name": "Surprise", "val_ap": 15.87, "test_ap": 11.79, "f1": 14.87, "thresh": 0.47},
        {"idx": 24, "name": "Sympathy", "val_ap": 45.62, "test_ap": 17.10, "f1": 20.78, "thresh": 0.42},
        {"idx": 25, "name": "Yearning", "val_ap": 13.91, "test_ap": 12.79, "f1": 15.32, "thresh": 0.39}
    ]
    
    macro_map = 34.76
    macro_f1 = np.mean([d['f1'] for d in data])
    
    names = [d['name'] for d in data]
    test_aps = [d['test_ap'] for d in data]
    f1_scores = [d['f1'] for d in data]
    val_aps = [d['val_ap'] for d in data]

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({'font.size': 10})

    # =========================================================================
    # BIỂU ĐỒ 1: CỘT NHÓM TEST AP (%) & TEST F1-SCORE (%) CHO 26 LỚP
    # =========================================================================
    fig, ax = plt.subplots(figsize=(20, 8.5), dpi=300)
    
    x = np.arange(len(names))
    width = 0.38

    rects1 = ax.bar(x - width/2, test_aps, width, label='Test AP (%)', color='#2563eb', edgecolor='#1d4ed8', alpha=0.92)
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Test F1-Score (%)', color='#059669', edgecolor='#047857', alpha=0.92)

    # Reference lines
    ax.axhline(macro_map, color='#dc2626', linestyle='--', linewidth=2, label=f'Macro mAP trung bình ({macro_map:.2f}%)')
    ax.axhline(macro_f1, color='#d97706', linestyle=':', linewidth=2, label=f'Macro F1 trung bình ({macro_f1:.2f}%)')

    ax.set_ylabel('Điểm số phần trăm (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Đánh giá hiệu năng RAPT-CLIP trên tập TEST bộ dữ liệu EMOTIC (5,292 mẫu)\n[So sánh điểm AP và F1-Score qua 26 nhãn cảm xúc đa nhãn]', fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=42, ha='right', fontsize=10.5, fontweight='500')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=11)

    # Value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, rotation=90)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, rotation=90, color='#065f46')

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, 'emotic_test_metrics_barchart.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Saved Chart 1 to {chart1_path}")

    # =========================================================================
    # BIỂU ĐỒ 2: XẾP HẠNG THỨ TỰ HIỆU NĂNG 26 LỚP CẢM XÚC (HORIZONTAL BAR CHART)
    # =========================================================================
    # Sort data by test_ap descending
    sorted_data = sorted(data, key=lambda d: d['test_ap'], reverse=True)
    s_names = [d['name'] for d in sorted_data]
    s_aps = [d['test_ap'] for d in sorted_data]
    s_f1s = [d['f1'] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
    y_pos = np.arange(len(s_names))

    # Color palette gradient from blue/green (high) to warm amber/red (low tail)
    colors = []
    for ap in s_aps:
        if ap >= 60:
            colors.append('#16a34a') # green
        elif ap >= 35:
            colors.append('#2563eb') # blue
        elif ap >= 20:
            colors.append('#d97706') # amber
        else:
            colors.append('#dc2626') # red

    bars = ax.barh(y_pos, s_aps, align='center', color=colors, edgecolor='#334155', height=0.68, alpha=0.88)
    ax.axvline(macro_map, color='#b91c1c', linestyle='--', linewidth=2, label=f'Macro mAP = {macro_map:.2f}% (Baseline Benchmark)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_names, fontsize=10.5, fontweight='600')
    ax.invert_yaxis()  # top-down ranking
    ax.set_xlabel('Average Precision - AP (%) trên tập kiểm thử (Test Set)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Thứ bậc xếp hạng độ chính xác (Test AP) của 26 cảm xúc EMOTIC\n[Mô hình RAPT-CLIP: ViT-B/16 + CMAF Tri-stream | model_best.zip]', fontsize=13.5, fontweight='bold', pad=14)
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', frameon=True, fontsize=11)

    # Annotate bar values
    for i, bar in enumerate(bars):
        w = bar.get_width()
        f1_val = s_f1s[i]
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{w:.2f}%  (F1: {f1_val:.1f}%)', 
                va='center', ha='left', fontsize=8.8, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, 'emotic_test_ranking_barchart.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Saved Chart 2 to {chart2_path}")

    # =========================================================================
    # BIỂU ĐỒ 3: SO SÁNH VALIDATION AP VS TEST AP (ĐỘ KHÁI QUÁT HÓA)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(20, 8.5), dpi=300)
    rects_v = ax.bar(x - width/2, val_aps, width, label='Validation AP (%) [Epoch 6]', color='#8b5cf6', edgecolor='#7c3aed', alpha=0.9)
    rects_t = ax.bar(x + width/2, test_aps, width, label='Test AP (%) [Unseen Benchmark]', color='#0ea5e9', edgecolor='#0284c7', alpha=0.9)

    ax.axhline(46.98, color='#7c3aed', linestyle='--', linewidth=1.8, label='Validation Macro mAP: 46.98%')
    ax.axhline(macro_map, color='#0284c7', linestyle='--', linewidth=1.8, label=f'Test Macro mAP: {macro_map:.2f}%')

    ax.set_ylabel('Average Precision AP (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Đối chiếu độ chính xác: Tập Validation vs Tập Test (EMOTIC Dataset)\n[Kiểm chứng khả năng tổng quát hóa trên dữ liệu hoàn toàn chưa từng thấy]', fontsize=14, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=42, ha='right', fontsize=10.5, fontweight='500')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=11)

    for rect in rects_v:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, rotation=90, color='#6d28d9')

    for rect in rects_t:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, rotation=90, color='#0369a1')

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, 'emotic_val_test_comparison_barchart.png')
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Saved Chart 3 to {chart3_path}")

if __name__ == '__main__':
    main()
