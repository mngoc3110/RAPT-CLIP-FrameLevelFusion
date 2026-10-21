#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_raer_ablation_pdf.py
───────────────────────────────────
Biên soạn và xuất file PDF Báo Cáo Khoa Học về Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Studies)
cho mô hình RAPT-CLIP:
- Bố cục tối ưu 5 trang in chuẩn A4, kích thước chữ (8pt - 8.5pt) và hình ảnh ma trận nhầm lẫn
  được căn chỉnh to rõ, sắc nét, đảm bảo in ra đọc được dễ dàng:
  • Trang 1: Toàn bộ bảng số liệu thực nghiệm RAER (Bảng 1: 16 Checkpoints & Bảng 3A + 3B: Feature Probing)
  • Trang 2: 8 Ma trận nhầm lẫn cốt lõi RAER (Cặp 1 &rarr; Cặp 4, hiển thị 2 cột to rõ)
  • Trang 3: 8 Ma trận nhầm lẫn còn lại (Cặp 5, 6 & Bộ 3 mô thức dung hợp) + Bảng so sánh 3 mô thức
  • Trang 4: Không gian biểu diễn t-SNE, Grad-CAM RAER & Toàn bộ đánh giá đa miền EMOTIC
  • Trang 5: Toàn bộ kết quả huấn luyện & đánh giá bộ dữ liệu CAER-S
"""

import os
import sys
import json
import base64
import weasyprint

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image_path)[1].lstrip('.').lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

def build_pdf():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_pdf = os.path.join(root_dir, "Bao_Cao_Ablation_RAER_RAPT_CLIP.pdf")

    # Load fresh ablation JSON
    json_path = os.path.join(root_dir, "outputs", "raer_checkpoint_ablation_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            ablation_results = json.load(f)
    else:
        ablation_results = []

    # Map of images
    cm_dir = os.path.join(root_dir, "outputs", "ablation_fresh_cms")
    cms_b64 = {}
    for item in ablation_results:
        img_id = item["id"]
        img_path = os.path.join(cm_dir, f"cm_{img_id}.png")
        cms_b64[img_id] = get_base64_image(img_path)

    # Architectural overview images
    tsne_fine = get_base64_image(os.path.join(root_dir, "image", "tsne_finetuned.png"))
    gradcam_img = get_base64_image(os.path.join(root_dir, "image", "gradcam_sample.png"))

    # EMOTIC results images
    emotic_ranking_img = get_base64_image(os.path.join(root_dir, "outputs", "EMOTIC", "emotic_test_ranking_barchart.png"))
    emotic_val_test_img = get_base64_image(os.path.join(root_dir, "outputs", "EMOTIC", "emotic_val_test_comparison_barchart.png"))
    emotic_grad_1 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Emotic", "1.jpeg"))
    emotic_grad_2 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Emotic", "2.jpeg"))
    emotic_grad_3 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Emotic", "3.jpeg"))

    # CAER-S results images
    caers_cm_img = get_base64_image(os.path.join(root_dir, "outputs", "CAER-S", "caer_s_confusion_matrix.png"))
    caers_recall_img = get_base64_image(os.path.join(root_dir, "outputs", "CAER-S", "caer_s_recall_barchart.png"))
    caers_grad_1 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Caers", "1_Anger.png"))
    caers_grad_2 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Caers", "4_Happy.png"))
    caers_grad_3 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Caers", "3_Fear.png"))

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo Khoa học: Nghiên cứu Cắt bỏ Thành phần (Ablation Study) RAER - RAPT-CLIP</title>
<style>
    @page {{
        size: A4;
        margin: 7mm 10mm 7mm 10mm;
        @bottom-right {{
            content: "Trang " counter(page) " / " counter(pages);
            font-size: 7.2pt;
            font-family: Arial, Helvetica, sans-serif;
            color: #555;
        }}
        @bottom-left {{
            content: "RAPT-CLIP: Báo Cáo Thực Nghiệm Cắt Bỏ Thành Phần (Ablation Studies)";
            font-size: 7.2pt;
            font-family: Arial, Helvetica, sans-serif;
            color: #777;
        }}
    }}

    body {{
        font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
        color: #1a1a1a;
        line-height: 1.3;
        font-size: 8.2pt;
    }}

    .header-box {{
        border-bottom: 2px solid #0d47a1;
        padding-bottom: 3px;
        margin-bottom: 5px;
    }}

    h1.doc-title {{
        font-size: 12pt;
        font-weight: bold;
        color: #0d47a1;
        margin: 0 0 2px 0;
        text-transform: uppercase;
        letter-spacing: 0.2px;
    }}

    .doc-subtitle {{
        font-size: 8.2pt;
        color: #333;
        font-weight: bold;
        margin: 0;
    }}

    h2.section-title {{
        font-size: 9pt;
        font-weight: bold;
        color: #0d47a1;
        border-left: 3.5px solid #0d47a1;
        padding-left: 5px;
        margin: 6px 0 3px 0;
        text-transform: uppercase;
    }}

    h3.subsection-title {{
        font-size: 8.2pt;
        font-weight: bold;
        color: #1565c0;
        margin: 4px 0 2px 0;
    }}

    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 2px 0 4px 0;
        font-size: 7.2pt;
        page-break-inside: avoid;
    }}

    table.data-table th {{
        background-color: #0d47a1;
        color: #ffffff;
        font-weight: bold;
        padding: 2px 3px;
        border: 1px solid #0a3880;
        text-align: center;
    }}

    table.data-table td {{
        border: 1px solid #d0d7de;
        padding: 1.6px 3px;
        vertical-align: middle;
    }}

    table.data-table tr.group-header td {{
        background-color: #e3f2fd;
        font-weight: bold;
        color: #0d47a1;
        font-size: 7.3pt;
        padding: 1.8px 3px;
    }}

    table.data-table tr:nth-child(even):not(.group-header) {{
        background-color: #fcfcfc;
    }}

    .badge-sota {{
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b5e20;
        border: 1px solid #a5d6a7;
        padding: 0.5px 3px;
        border-radius: 2.5px;
        font-weight: bold;
        font-size: 6.5pt;
    }}

    .badge-drop {{
        display: inline-block;
        background-color: #ffebee;
        color: #b71c1c;
        border: 1px solid #ef9a9a;
        padding: 0.5px 3px;
        border-radius: 2.5px;
        font-weight: bold;
        font-size: 6.5pt;
    }}

    table.side-by-side {{
        width: 100%;
        border-collapse: collapse;
        margin: 2px 0;
        page-break-inside: avoid;
    }}

    table.side-by-side td {{
        vertical-align: top;
        padding: 0 3px;
        border: none;
    }}

    table.cm-pair-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1px 0 3px 0;
        page-break-inside: avoid;
    }}

    table.cm-pair-table td {{
        width: 50%;
        padding: 1px 3px;
        vertical-align: top;
        text-align: center;
        border: none;
    }}

    table.cm-pair-table img {{
        width: 98%;
        max-height: 135px;
        object-fit: contain;
        border: 1px solid #ccc;
        border-radius: 2.5px;
    }}

    table.cm-triple-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1px 0 3px 0;
        page-break-inside: avoid;
    }}

    table.cm-triple-table td {{
        width: 33.33%;
        padding: 1px 2px;
        vertical-align: top;
        text-align: center;
        border: none;
    }}

    table.cm-triple-table img {{
        width: 98%;
        max-height: 125px;
        object-fit: contain;
        border: 1px solid #ccc;
        border-radius: 2.5px;
    }}

    table.two-col-figures {{
        width: 100%;
        border-collapse: collapse;
        margin: 2px 0 3px 0;
        page-break-inside: avoid;
    }}

    table.two-col-figures td {{
        width: 50%;
        padding: 1px 3px;
        vertical-align: top;
        text-align: center;
        border: none;
    }}

    table.two-col-figures img {{
        width: 98%;
        max-height: 140px;
        object-fit: contain;
        border: 1px solid #ccc;
        border-radius: 2.5px;
    }}

    table.three-col-figures {{
        width: 100%;
        border-collapse: collapse;
        margin: 2px 0;
        page-break-inside: avoid;
    }}

    table.three-col-figures td {{
        width: 33.33%;
        padding: 1px 2px;
        vertical-align: top;
        text-align: center;
        border: none;
    }}

    table.three-col-figures img {{
        width: 98%;
        max-height: 118px;
        object-fit: contain;
        border: 1px solid #ccc;
        border-radius: 2.5px;
    }}

    .figure-caption {{
        font-size: 7.1pt;
        font-weight: bold;
        color: #222;
        margin-top: 2px;
        line-height: 1.25;
    }}

    .page-break {{
        page-break-before: always;
    }}

    code {{
        font-family: Consolas, "Courier New", monospace;
        font-size: 7pt;
        background-color: #f4f5f7;
        padding: 0.5px 2px;
        border-radius: 2px;
        color: #c2185b;
    }}
</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- TRANG 1: TOÀN BỘ CÁC BẢNG SỐ LIỆU THỰC NGHIỆM RAER (BẢNG 1 + BẢNG 3A/3B) -->
<!-- ========================================================================= -->
<div class="header-box">
    <h1 class="doc-title">Báo Cáo Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Studies)</h1>
    <div class="doc-subtitle">Bộ Dữ Liệu Cảm Xúc Video Học Đường RAER &bull; Kiến Trúc RAPT-CLIP Tri-Stream (528 Test Clips, 5 Emotion Classes)</div>
</div>

<h2 class="section-title">1. Bảng Tổng Hợp Kết Quả Thực Nghiệm Đo Trực Tiếp Từ Checkpoints (16 Thí Nghiệm)</h2>

<table class="data-table">
    <thead>
        <tr>
            <th style="width: 4%;">STT</th>
            <th style="width: 33%;">Biến thể Thí nghiệm (Ablation Variant)</th>
            <th style="width: 30%;">Cơ chế Can thiệp / Bỏ module</th>
            <th style="width: 9%;">WAR (%)</th>
            <th style="width: 9%;">UAR (%)</th>
            <th style="width: 15%;">&Delta; so với SOTA</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color: #e8f5e9; font-weight: bold;">
            <td style="text-align: center;">1</td>
            <td>RAPT-CLIP Frame-Level Fusion (SOTA Đề Xuất)</td>
            <td>Đầy đủ: Face + Body + Context + CMAF + EAA + Temporal</td>
            <td style="text-align: center; color: #2e7d32;">77.08%</td>
            <td style="text-align: center; color: #2e7d32;">77.66%</td>
            <td style="text-align: center; color: #2e7d32;">Baseline Chuẩn Checkpoint</td>
        </tr>
        <tr class="group-header">
            <td colspan="6">A. Cắt bỏ các Khối Kiến Trúc Cốt Lõi (Modular Structural Ablation)</td>
        </tr>
        <tr>
            <td style="text-align: center;">2</td>
            <td><strong>Cắt Khối CMAF (Tắt Cross-Attention)</strong></td>
            <td>Ghép trực tiếp Concat [Face, Body, Context] không qua CMAF</td>
            <td style="text-align: center;">80.30%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">66.48%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">&Delta;UAR: -11.18%</td>
        </tr>
        <tr>
            <td style="text-align: center;">3</td>
            <td><strong>Cắt Khối EAA (Tắt Face Adapter)</strong></td>
            <td>Dùng trực tiếp CLIP ViT Face tokens không qua Adapter</td>
            <td style="text-align: center;">76.14%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">69.14%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">&Delta;UAR: -8.52%</td>
        </tr>
        <tr>
            <td style="text-align: center;">4</td>
            <td><strong>Cắt Khối Temporal Transformer</strong></td>
            <td>Mean Pooling trực tiếp 16 frames thay vì Transformer</td>
            <td style="text-align: center;">79.36%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">68.81%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">&Delta;UAR: -8.85%</td>
        </tr>
        <tr>
            <td style="text-align: center;">5</td>
            <td><strong>Cắt Khối Attention Pooling</strong></td>
            <td>Dùng Mean Pooling sau Temporal Transformer thay vì Attn-Pool</td>
            <td style="text-align: center;">78.79%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">68.47%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">&Delta;UAR: -9.19%</td>
        </tr>
        <tr>
            <td style="text-align: center;">6</td>
            <td><strong>Cắt Prompt Tuning (Fixed Descriptors)</strong></td>
            <td>Huấn luyện với Text Prompt cố định không có CoOp</td>
            <td style="text-align: center; font-weight: bold;">80.49%</td>
            <td style="text-align: center; color: #c62828; font-weight: bold;">68.62%</td>
            <td style="text-align: center; color: #c62828; font-weight: bold;">&Delta;UAR: -9.04%</td>
        </tr>

        <tr class="group-header">
            <td colspan="6">B. Cắt bỏ các Luồng Dữ Liệu Đa Phương Thức (Modality Pruning)</td>
        </tr>
        <tr>
            <td style="text-align: center;">7</td>
            <td><strong>Cắt Luồng Context (W/o Context)</strong></td>
            <td>Zero-out Context (Chỉ dùng Mặt + Thân)</td>
            <td style="text-align: center;">72.54%</td>
            <td style="text-align: center;">75.51%</td>
            <td style="text-align: center; color: #e65100;">&Delta;UAR: -2.15% | &Delta;WAR: -4.54%</td>
        </tr>
        <tr>
            <td style="text-align: center;">8</td>
            <td><strong>Cắt Luồng Face (W/o Face)</strong></td>
            <td>Zero-out Face (Chỉ dùng Thân + Bối Cảnh)</td>
            <td style="text-align: center;">77.27%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">67.90%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">&Delta;UAR: -9.76%</td>
        </tr>
        <tr>
            <td style="text-align: center;">9</td>
            <td><strong>Cắt Luồng Body (W/o Body)</strong></td>
            <td>Zero-out Body (Chỉ dùng Mặt + Bối Cảnh)</td>
            <td style="text-align: center;">75.00%</td>
            <td style="text-align: center;">77.57%</td>
            <td style="text-align: center; color: #e65100;">&Delta;UAR: -0.09% | &Delta;WAR: -2.08%</td>
        </tr>
        <tr>
            <td style="text-align: center;">10</td>
            <td><strong>Đơn Luồng Face-Only</strong></td>
            <td>Zero-out cả Body và Context</td>
            <td style="text-align: center; color: #c62828; font-weight: bold;">32.58%</td>
            <td style="text-align: center;">61.22%</td>
            <td style="text-align: center; color: #c62828; font-weight: bold;">&Delta;UAR: -16.44% | &Delta;WAR: -44.50%</td>
        </tr>
        <tr>
            <td style="text-align: center;">11</td>
            <td><strong>Đơn Luồng Body-Only</strong></td>
            <td>Zero-out cả Face và Context</td>
            <td style="text-align: center;">75.95%</td>
            <td style="text-align: center;">67.78%</td>
            <td style="text-align: center; color: #c62828;">&Delta;UAR: -9.88%</td>
        </tr>
        <tr>
            <td style="text-align: center;">12</td>
            <td><strong>Đơn Luồng Context-Only</strong></td>
            <td>Zero-out cả Face và Body</td>
            <td style="text-align: center;">75.00%</td>
            <td style="text-align: center;">70.38%</td>
            <td style="text-align: center; color: #c62828;">&Delta;UAR: -7.28%</td>
        </tr>

        <tr class="group-header">
            <td colspan="6">C. Độ Bền Vững & Xáo Trộn Không Gian / Thời Gian (Robustness)</td>
        </tr>
        <tr>
            <td style="text-align: center;">13</td>
            <td><strong>Xáo Trộn Thời Gian (Temporal Shuffling)</strong></td>
            <td>Đảo lộn ngẫu nhiên vị trí 16 frames trước khi vào Temporal</td>
            <td style="text-align: center;">77.08%</td>
            <td style="text-align: center;">75.37%</td>
            <td style="text-align: center; color: #e65100; font-weight: bold;">&Delta;UAR: -2.29%</td>
        </tr>
        <tr>
            <td style="text-align: center;">14</td>
            <td><strong>Nhiễu Thị Giác Gaussian (&sigma; = 0.1)</strong></td>
            <td>Bơm nhiễu ngẫu nhiên vào vector đặc trưng</td>
            <td style="text-align: center;">77.65%</td>
            <td style="text-align: center;">77.27%</td>
            <td style="text-align: center; color: #2e7d32; font-weight: bold;">&Delta;UAR: -0.39%</td>
        </tr>

        <tr class="group-header">
            <td colspan="6">D. So Sánh Các Mô Thức Dung Hợp Không - Thời Gian (Spatio-Temporal Fusion Paradigms)</td>
        </tr>
        <tr style="background-color: #fffde7;">
            <td style="text-align: center;">15</td>
            <td><strong>Video-Level Fusion (Temporal Trước CMAF)</strong></td>
            <td>Temporal độc lập 3 luồng trước &rarr; CMAF dung hợp video-level</td>
            <td style="text-align: center; color: #c62828; font-weight: bold;">63.07%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">72.33%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">&Delta;UAR: -5.33% | &Delta;WAR: -14.01%</td>
        </tr>
        <tr style="background-color: #fff9c4;">
            <td style="text-align: center;">16</td>
            <td><strong>Parallel Temporal Fusion (3 Nhánh Song Song)</strong></td>
            <td>CMAF Decoupled &rarr; 3 Temporal song song &rarr; Concat video-level</td>
            <td style="text-align: center; font-weight: bold; color: #e65100;">72.73%</td>
            <td style="text-align: center; font-weight: bold; color: #e65100;">74.10% <small>(Best: 75.16%)</small></td>
            <td style="text-align: center; color: #e65100; font-weight: bold;">&Delta;UAR: -3.56% | &Delta;WAR: -4.35%</td>
        </tr>
    </tbody>
</table>

<h2 class="section-title" style="margin-top: 6px;">2. Thực Nghiệm Huấn Luyện Cắt Bỏ Thành Phần (Feature-Level Probing trên RAER)</h2>

<table class="data-table" style="margin: 2px 0;">
    <thead>
        <tr style="background-color: #0d47a1; color: white;">
            <th style="width: 24%;">Cấu hình Ablation</th>
            <th style="width: 38%;">Mô tả Biến thể Thiết kế &amp; Hàm Mục tiêu</th>
            <th style="width: 10%; text-align: center;">WAR (%)</th>
            <th style="width: 10%; text-align: center;">UAR (%)</th>
            <th style="width: 10%; text-align: center;">&Delta; UAR</th>
            <th style="width: 8%; text-align: center;">Best Ep</th>
        </tr>
    </thead>
    <tbody>
        <tr><td><strong>No_LDAM</strong></td><td>Thay thế LDAM bằng chuẩn Cross-Entropy Loss</td><td style="text-align: center;">77.46%</td><td style="text-align: center;">69.83%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-7.83%</td><td style="text-align: center;">16</td></tr>
        <tr><td><strong>No_Adapter</strong></td><td>Loại bỏ Face Adapter EAA (Dùng trực tiếp frozen CLIP face tokens)</td><td style="text-align: center;">78.41%</td><td style="text-align: center;">69.69%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-7.97%</td><td style="text-align: center;">17</td></tr>
        <tr><td><strong>No_Attn_Pooling</strong></td><td>Thay thế Self-Attention Pooling bằng Mean Pooling</td><td style="text-align: center;">76.89%</td><td style="text-align: center;">72.25%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-5.41%</td><td style="text-align: center;">20</td></tr>
        <tr style="background-color: #ffebee;"><td><strong>No_Weighted_Sampler</strong></td><td>Lấy mẫu ngẫu nhiên đồng đều (Uniform Sampler) thay cho Weighted Sampler</td><td style="text-align: center;">78.22%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">62.38%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-15.28%</td><td style="text-align: center;">10</td></tr>
        <tr><td><strong>Fusion_GFI</strong></td><td>Thay CMAF bằng Gated Feature Integration (Cổng nhân tuyến tính Sigmoid)</td><td style="text-align: center;">67.23%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-10.24%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-10.24%</td><td style="text-align: center;">14</td></tr>
        <tr><td><strong>Fusion_Concat</strong></td><td>Thay CMAF bằng Ghép kênh trực tiếp (Direct Concat + Linear Projection)</td><td style="text-align: center;">67.80%</td><td style="text-align: center;">72.13%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-5.53%</td><td style="text-align: center;">12</td></tr>
        <tr><td><strong>No_Modality_Dropout</strong></td><td>Đặt Modality Dropout = 0.0 (Không triệt tiêu ngẫu nhiên luồng khi train)</td><td style="text-align: center;">73.48%</td><td style="text-align: center;">70.09%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-7.57%</td><td style="text-align: center;">11</td></tr>
        <tr><td><strong>No_Temporal_Layers</strong></td><td>Loại bỏ các lớp Transformer trên trục thời gian (Mean pooling thô qua 16 đoạn)</td><td style="text-align: center;">73.11%</td><td style="text-align: center;">70.48%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">-7.18%</td><td style="text-align: center;">15</td></tr>
    </tbody>
</table>

<h3 class="subsection-title" style="margin-top: 4px;">Chi Tiết Recall 5 Lớp Cảm Xúc RAER</h3>
<table class="data-table" style="margin: 2px 0;">
    <thead>
        <tr style="background-color: #1565c0; color: white;">
            <th style="width: 25%;">Cấu hình Ablation</th>
            <th style="width: 15%; text-align: center;">Neutrality (115)</th>
            <th style="width: 15%; text-align: center;">Enjoyment (32)</th>
            <th style="width: 15%; text-align: center;">Confusion (7)</th>
            <th style="width: 15%; text-align: center;">Fatigue (39)</th>
            <th style="width: 15%; text-align: center;">Distraction (105)</th>
        </tr>
    </thead>
    <tbody>
        <tr><td><strong>No_LDAM</strong></td><td style="text-align: center;">76.81%</td><td style="text-align: center;">75.00%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">28.57%</td><td style="text-align: center;">89.74%</td><td style="text-align: center;">79.05%</td></tr>
        <tr><td><strong>No_Adapter</strong></td><td style="text-align: center;">79.13%</td><td style="text-align: center; color: #b71c1c;">62.50%</td><td style="text-align: center;">42.86%</td><td style="text-align: center;">82.05%</td><td style="text-align: center;">81.90%</td></tr>
        <tr><td><strong>No_Attn_Pooling</strong></td><td style="text-align: center;">75.65%</td><td style="text-align: center;">78.12%</td><td style="text-align: center;">42.86%</td><td style="text-align: center;">84.62%</td><td style="text-align: center;">80.00%</td></tr>
        <tr style="background-color: #ffebee;"><td><strong>No_Weighted_Sampler</strong></td><td style="text-align: center;">82.90%</td><td style="text-align: center;">81.25%</td><td style="text-align: center; color: #b71c1c; font-weight: bold;">0.00%</td><td style="text-align: center;">82.05%</td><td style="text-align: center;">65.71%</td></tr>
        <tr><td><strong>Fusion_GFI</strong></td><td style="text-align: center;">62.61%</td><td style="text-align: center;">75.00%</td><td style="text-align: center;">42.86%</td><td style="text-align: center;">79.49%</td><td style="text-align: center;">77.14%</td></tr>
        <tr><td><strong>Fusion_Concat</strong></td><td style="text-align: center;">61.74%</td><td style="text-align: center;">78.12%</td><td style="text-align: center; color: #1b5e20;">57.14%</td><td style="text-align: center;">84.62%</td><td style="text-align: center;">79.05%</td></tr>
        <tr><td><strong>No_Modality_Dropout</strong></td><td style="text-align: center;">72.17%</td><td style="text-align: center;">78.12%</td><td style="text-align: center;">42.86%</td><td style="text-align: center;">82.05%</td><td style="text-align: center;">75.24%</td></tr>
        <tr><td><strong>No_Temporal_Layers</strong></td><td style="text-align: center;">70.72%</td><td style="text-align: center;">81.25%</td><td style="text-align: center;">42.86%</td><td style="text-align: center;">79.49%</td><td style="text-align: center;">78.10%</td></tr>
    </tbody>
</table>

<!-- ========================================================================= -->
<!-- TRANG 2: TRỰC QUAN HÓA MA TRẬN NHẦM LẪN (HÌNH 1 -> HÌNH 4, KHỔ TO RÕ 2 CỘT) -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h2 class="section-title">3. Trực Quan Hóa Ma Trận Nhầm Lẫn (Confusion Matrices) Trên RAER (Phần 1)</h2>

<h3 class="subsection-title">3.1 Cắt Khối Dung Hợp CMAF vs Full Model SOTA</h3>
<table class="cm-pair-table">
    <tr>
        <td>
            <img src="{cms_b64.get('01_full_tristream_sota', '')}" alt="Full Model SOTA">
            <div class="figure-caption">Hình 1a: Full Model SOTA (WAR: 77.08% | UAR: 77.66%)</div>
        </td>
        <td>
            <img src="{cms_b64.get('02_bypass_cmaf', '')}" alt="Bypass CMAF">
            <div class="figure-caption">Hình 1b: Cắt Khối CMAF (WAR: 80.30% | UAR: 66.48% &bull; &Delta;UAR: -11.18%)</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title">3.2 Cắt Khối EAA Face Adapter vs Cắt CoOp Prompt Tuning</h3>
<table class="cm-pair-table">
    <tr>
        <td>
            <img src="{cms_b64.get('03_bypass_eaa', '')}" alt="Bypass EAA">
            <div class="figure-caption">Hình 2a: Cắt Khối EAA Face Adapter (WAR: 76.14% | UAR: 69.14% &bull; &Delta;UAR: -8.52%)</div>
        </td>
        <td>
            <img src="{cms_b64.get('14_hand_crafted_prompts', '')}" alt="Cắt Prompt Tuning">
            <div class="figure-caption">Hình 2b: Cắt Prompt Tuning (WAR: 80.49% | UAR: 68.62% &bull; &Delta;UAR: -9.04%)</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title">3.3 Khảo Sát Mô Hình Hóa Thời Gian: Cắt Temporal Transformer vs Cắt Attention Pooling</h3>
<table class="cm-pair-table">
    <tr>
        <td>
            <img src="{cms_b64.get('04_bypass_temporal', '')}" alt="Bypass Temporal">
            <div class="figure-caption">Hình 3a: Cắt Temporal Transformer (WAR: 79.36% | UAR: 68.81% &bull; &Delta;UAR: -8.85%)</div>
        </td>
        <td>
            <img src="{cms_b64.get('05_no_attn_pool', '')}" alt="No Attn Pool">
            <div class="figure-caption">Hình 3b: Cắt Attn-Pool (Mean) (WAR: 78.79% | UAR: 68.47% &bull; &Delta;UAR: -9.19%)</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title">3.4 Đóng Góp Các Luồng Khuôn Mặt (Face) & Thân Thể (Body)</h3>
<table class="cm-pair-table">
    <tr>
        <td>
            <img src="{cms_b64.get('07_wo_face', '')}" alt="W/o Face">
            <div class="figure-caption">Hình 4a: Cắt Luồng Face (WAR: 77.27% | UAR: 67.90% &bull; &Delta;UAR: -9.76%)</div>
        </td>
        <td>
            <img src="{cms_b64.get('08_wo_body', '')}" alt="W/o Body">
            <div class="figure-caption">Hình 4b: Cắt Luồng Body (WAR: 75.00% | UAR: 77.57% &bull; &Delta;WAR: -2.08%)</div>
        </td>
    </tr>
</table>

<!-- ========================================================================= -->
<!-- TRANG 3: MA TRẬN NHẦM LẪN (HÌNH 5 -> HÌNH 7) & SO SÁNH 3 MÔ THỨC DUNG HỢP -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h2 class="section-title">3. Trực Quan Hóa Ma Trận Nhầm Lẫn (Confusion Matrices) Trên RAER (Phần 2)</h2>

<h3 class="subsection-title">3.5 Khảo Sát Đơn Luồng (Single-Stream Modalities)</h3>
<table class="cm-pair-table">
    <tr>
        <td>
            <img src="{cms_b64.get('09_face_only', '')}" alt="Face Only">
            <div class="figure-caption">Hình 5a: Đơn Luồng Face-Only (WAR: 32.58% | UAR: 61.22%)</div>
        </td>
        <td>
            <img src="{cms_b64.get('11_context_only', '')}" alt="Context Only">
            <div class="figure-caption">Hình 5b: Đơn Luồng Context-Only (WAR: 75.00% | UAR: 70.38%)</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title">3.6 Tính Bền Vững Với Xáo Trộn Thời Gian & Nhiễu Thị Giác</h3>
<table class="cm-pair-table">
    <tr>
        <td>
            <img src="{cms_b64.get('12_temporal_shuffling', '')}" alt="Temporal Shuffling">
            <div class="figure-caption">Hình 6a: Xáo Trộn Thời Gian (WAR: 77.08% | UAR: 75.37% &bull; &Delta;UAR: -2.29%)</div>
        </td>
        <td>
            <img src="{cms_b64.get('13_gaussian_noise_01', '')}" alt="Gaussian Noise">
            <div class="figure-caption">Hình 6b: Nhiễu Gaussian &sigma;=0.1 (WAR: 77.65% | UAR: 77.27% &bull; &Delta;UAR: -0.39%)</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title">3.7 Đối So Sánh 3 Mô Thức Dung Hợp Không - Thời Gian (Spatio-Temporal Fusion Paradigms)</h3>
<table class="cm-triple-table">
    <tr>
        <td>
            <img src="{cms_b64.get('01_full_tristream_sota', '')}" alt="Frame-Level Fusion SOTA">
            <div class="figure-caption">Hình 7a: Frame-Level (SOTA)<br>WAR: 77.08% | UAR: 77.66%</div>
        </td>
        <td>
            <img src="{cms_b64.get('16_parallel_temporal_fusion', '')}" alt="Parallel 3 Branches">
            <div class="figure-caption">Hình 7b: Parallel Temporal (3 nhánh)<br>WAR: 72.73% | UAR: 74.10%</div>
        </td>
        <td>
            <img src="{cms_b64.get('15_video_level_fusion', '')}" alt="Video-Level Fusion">
            <div class="figure-caption">Hình 7c: Video-Level Fusion<br>WAR: 63.07% | UAR: 72.33%</div>
        </td>
    </tr>
</table>

<table class="data-table" style="margin-top: 4px;">
    <thead>
        <tr style="background-color: #0d47a1; color: white;">
            <th style="width: 26%;">Mô thức Dung hợp Không - Thời gian</th>
            <th style="width: 36%;">Cơ chế Tương tác Liên luồng</th>
            <th style="width: 14%; text-align: center;">Tham số Temporal</th>
            <th style="width: 14%; text-align: center;">WAR / UAR (%)</th>
            <th style="width: 10%; text-align: center;">Xếp hạng</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color: #e8f5e9;">
            <td><strong>Frame-Level Fusion (Hình 7a)</strong></td>
            <td>CMAF tương tác từng frame &rarr; Concat &rarr; Unified Temporal</td>
            <td style="text-align: center; font-weight: bold;">~4.7M (1 khối)</td>
            <td style="text-align: center; color: #1b5e20; font-weight: bold;">77.08% / 77.66%</td>
            <td style="text-align: center;"><span class="badge-sota">SOTA ĐỀ XUẤT</span></td>
        </tr>
        <tr>
            <td><strong>Parallel Temporal (Hình 7b)</strong></td>
            <td>3 khối Temporal độc lập (Face, Body, Context) &rarr; Concat muộn</td>
            <td style="text-align: center; color: #b71c1c;">~9.4M (3 khối)</td>
            <td style="text-align: center; font-weight: bold;">72.73% / 74.10%</td>
            <td style="text-align: center;"><span class="badge-drop">-3.56% UAR</span></td>
        </tr>
        <tr style="background-color: #ffebee;">
            <td><strong>Video-Level Fusion (Hình 7c)</strong></td>
            <td>Nén thời gian trước khi cho tương tác CMAF</td>
            <td style="text-align: center;">~4.7M (1 khối)</td>
            <td style="text-align: center; color: #b71c1c; font-weight: bold;">63.07% / 72.33%</td>
            <td style="text-align: center;"><span class="badge-drop">-14.01% WAR</span></td>
        </tr>
    </tbody>
</table>

<!-- ========================================================================= -->
<!-- TRANG 4: T-SNE, GRAD-CAM RAER & BỘ DỮ LIỆU TỰ NHIÊN ĐA NHÃN EMOTIC -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h2 class="section-title">4. Trực Quan Hóa Đặc Trưng t-SNE & Grad-CAM Trên RAER</h2>

<table class="two-col-figures">
    <tr>
        <td>
            <img src="{tsne_fine}" alt="t-SNE Representation">
            <div class="figure-caption">Hình 8a: Không gian biểu diễn phân bố đặc trưng t-SNE của RAPT-CLIP Frame-Level Fusion</div>
        </td>
        <td>
            <img src="{gradcam_img}" alt="Grad-CAM 4-Stream">
            <div class="figure-caption">Hình 8b: Bản đồ Grad-CAM đa luồng (Face, Body, Context, Full Frame) trên RAER</div>
        </td>
    </tr>
</table>

<h2 class="section-title" style="margin-top: 5px;">5. Đánh Giá Khả Năng Mở Rộng Đa Miền: Bộ Dữ Liệu Tự Nhiên Đa Nhãn EMOTIC (5,292 Test Images)</h2>

<table class="side-by-side">
    <tr>
        <td style="width: 48%;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 42%;">Phương pháp</th>
                        <th style="width: 32%; text-align: center;">mAP (%)</th>
                        <th style="width: 26%; text-align: center;">So với RAPT</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Kosti et al. (CVPR'17)</td><td style="text-align: center;">27.38%</td><td style="text-align: center; color: #2e7d32; font-weight: bold;">+7.38%</td></tr>
                    <tr><td>EmotiCon (CVPR'20)</td><td style="text-align: center;">33.10%</td><td style="text-align: center; color: #2e7d32; font-weight: bold;">+1.66%</td></tr>
                    <tr style="background-color: #e8f5e9; font-weight: bold;">
                        <td><strong>RAPT-CLIP (Ours)</strong></td>
                        <td style="text-align: center; color: #1b5e20;"><strong>34.76%</strong></td>
                        <td style="text-align: center;"><span class="badge-sota">SOTA</span></td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td style="width: 52%;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 40%;">Tham số</th>
                        <th style="width: 60%;">Cấu hình thực nghiệm</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Backbone</td><td>CLIP ViT-B/16 (224 &times; 224, unfrozen)</td></tr>
                    <tr><td>Epochs / Batch</td><td>25 Epochs (Đỉnh Ep 6) | Batch Size: 16</td></tr>
                    <tr><td>Loss / Scheduler</td><td>Dynamic ASL (&gamma;<sub>-</sub>=4.0, clip=0.05) | MultiStepLR</td></tr>
                    <tr><td>Fusion / Prompts</td><td>CMAF Tri-Stream (1536-d) | 8 learnable tokens</td></tr>
                </tbody>
            </table>
        </td>
    </tr>
</table>

<table class="two-col-figures" style="margin-top: 2px;">
    <tr>
        <td style="width: 50%;">
            <img src="{emotic_ranking_img}" alt="EMOTIC Test AP Ranking" style="max-height: 120px;">
            <div class="figure-caption">Hình 9a: Bảng xếp hạng Average Precision (AP) trên 26 lớp cảm xúc tập Test EMOTIC</div>
        </td>
        <td style="width: 50%;">
            <img src="{emotic_val_test_img}" alt="EMOTIC Val vs Test Comparison" style="max-height: 120px;">
            <div class="figure-caption">Hình 9b: So sánh tương quan Average Precision giữa Validation (46.98%) và Test (34.76%)</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title" style="margin-top: 3px;">Trực Quan Hóa Grad-CAM Trên Tập Dữ Liệu Tự Nhiên EMOTIC</h3>
<table class="three-col-figures">
    <tr>
        <td>
            <img src="{emotic_grad_1}" alt="Grad-CAM EMOTIC 1">
            <div class="figure-caption">Hình 10a: Mẫu 1 (Mặt &amp; Cảnh)</div>
        </td>
        <td>
            <img src="{emotic_grad_2}" alt="Grad-CAM EMOTIC 2">
            <div class="figure-caption">Hình 10b: Mẫu 2 (Hành vi Cơ thể)</div>
        </td>
        <td>
            <img src="{emotic_grad_3}" alt="Grad-CAM EMOTIC 3">
            <div class="figure-caption">Hình 10c: Mẫu 3 (Cảm xúc Phức hợp)</div>
        </td>
    </tr>
</table>

<!-- ========================================================================= -->
<!-- TRANG 5: HUẤN LUYỆN VÀ ĐÁNH GIÁ TRÊN BỘ DỮ LIỆU CAER-S -->
<!-- ========================================================================= -->
<div class="page-break"></div>
<h2 class="section-title">6. Huấn Luyện và Đánh Giá Trên Bộ Dữ Liệu CAER-S (Static Images - 70,000 Ảnh)</h2>

<table class="side-by-side">
    <tr>
        <td style="width: 50%;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 40%;">Phương pháp</th>
                        <th style="width: 25%; text-align: center;">Năm</th>
                        <th style="width: 35%; text-align: center;">WAR / UAR</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>CAER-Net-S</td><td style="text-align: center;">2019</td><td style="text-align: center;">73.51%</td></tr>
                    <tr><td>EmotiCon</td><td style="text-align: center;">2020</td><td style="text-align: center;">77.00%</td></tr>
                    <tr><td>GRERN</td><td style="text-align: center;">2021</td><td style="text-align: center;">81.31%</td></tr>
                    <tr><td>EfficientFace</td><td style="text-align: center;">2021</td><td style="text-align: center;">85.87%</td></tr>
                    <tr><td>MA-Net</td><td style="text-align: center;">2021</td><td style="text-align: center;">88.42%</td></tr>
                    <tr><td>GLAMOR-Net</td><td style="text-align: center;">2023</td><td style="text-align: center;">89.88%</td></tr>
                    <tr><td>AGCD-Net</td><td style="text-align: center;">2023</td><td style="text-align: center;">90.65%</td></tr>
                    <tr><td>SAFER</td><td style="text-align: center;">2024</td><td style="text-align: center;">91.40%</td></tr>
                    <tr style="background-color: #e8f5e9; font-weight: bold;">
                        <td><strong>RAPT-CLIP (Ours)</strong></td>
                        <td style="text-align: center;"><strong>Ep 19</strong></td>
                        <td style="text-align: center; color: #1b5e20;"><strong>90.40%</strong></td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td style="width: 50%;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 40%;">Tham số</th>
                        <th style="width: 60%;">Cấu hình thực nghiệm (caer_s_cmaf.sh)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Backbone</td><td>CLIP ViT-B/16 (224 &times; 224, unfrozen)</td></tr>
                    <tr><td>Epochs / Batch</td><td>20 Epochs (Đỉnh Ep 19) | Batch Size: 16</td></tr>
                    <tr><td>Hàm mất mát</td><td>LDAM Loss (s = 1.0, m<sub>max</sub> = 0.5)</td></tr>
                    <tr><td>Bộ lấy mẫu</td><td>WeightedRandomSampler (Cân bằng batch)</td></tr>
                    <tr><td>Tốc độ học</td><td>Base: 2e-5 | ViT: 1e-6 | Prompt: 3e-4</td></tr>
                    <tr><td>Luồng dữ liệu</td><td>Triple-Stream: Face + Body + Masked Context</td></tr>
                    <tr><td>Dung hợp</td><td>Tridirectional Frame-Level CMAF (1536-d)</td></tr>
                    <tr><td>Temporal</td><td>Temporal Transformer (1 layer, Attn-Pool)</td></tr>
                    <tr><td>TTA</td><td>Horizontal Flip trên cả 3 luồng</td></tr>
                </tbody>
            </table>
        </td>
    </tr>
</table>

<table class="side-by-side" style="margin-top: 3px;">
    <tr>
        <td style="width: 44%;">
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 12%;">ID</th>
                        <th style="width: 38%;">Cảm xúc</th>
                        <th style="width: 25%; text-align: center;">Đúng</th>
                        <th style="width: 25%; text-align: center;">Recall</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>0</td><td>Anger</td><td style="text-align: center;">923/1k</td><td style="text-align: center; font-weight: bold;">92.30%</td></tr>
                    <tr><td>1</td><td>Disgust</td><td style="text-align: center;">981/1k</td><td style="text-align: center; font-weight: bold;">98.10%</td></tr>
                    <tr><td>2</td><td>Fear</td><td style="text-align: center;">997/1k</td><td style="text-align: center; font-weight: bold; color: #1b5e20;">99.70%</td></tr>
                    <tr><td>3</td><td>Happy</td><td style="text-align: center;">834/1k</td><td style="text-align: center; font-weight: bold;">83.40%</td></tr>
                    <tr><td>4</td><td>Neutral</td><td style="text-align: center;">737/1k</td><td style="text-align: center; font-weight: bold; color: #b71c1c;">73.70%</td></tr>
                    <tr><td>5</td><td>Sad</td><td style="text-align: center;">934/1k</td><td style="text-align: center; font-weight: bold;">93.40%</td></tr>
                    <tr><td>6</td><td>Surprise</td><td style="text-align: center;">922/1k</td><td style="text-align: center; font-weight: bold;">92.20%</td></tr>
                    <tr style="background-color: #e8f5e9; font-weight: bold;">
                        <td colspan="2"><strong>WAR / UAR</strong></td>
                        <td style="text-align: center; color: #1b5e20;"><strong>6,328</strong></td>
                        <td style="text-align: center; color: #1b5e20;"><strong>90.40%</strong></td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td style="width: 56%;">
            <table class="data-table" style="text-align: center;">
                <thead>
                    <tr>
                        <th style="width: 16%;">True &bsol; Pred</th>
                        <th style="width: 12%;">Ang</th>
                        <th style="width: 12%;">Dis</th>
                        <th style="width: 12%;">Fea</th>
                        <th style="width: 12%;">Hap</th>
                        <th style="width: 12%;">Neu</th>
                        <th style="width: 12%;">Sad</th>
                        <th style="width: 12%;">Sur</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><strong>Anger</strong></td><td style="background-color: #bbdefb; font-weight: bold;">923</td><td>10</td><td>1</td><td>7</td><td>28</td><td>17</td><td>14</td></tr>
                    <tr><td><strong>Disgust</strong></td><td>4</td><td style="background-color: #bbdefb; font-weight: bold;">981</td><td>2</td><td>4</td><td>1</td><td>7</td><td>1</td></tr>
                    <tr><td><strong>Fear</strong></td><td>0</td><td>1</td><td style="background-color: #bbdefb; font-weight: bold;">997</td><td>0</td><td>0</td><td>2</td><td>0</td></tr>
                    <tr><td><strong>Happy</strong></td><td>25</td><td>4</td><td>8</td><td style="background-color: #bbdefb; font-weight: bold;">834</td><td>70</td><td>17</td><td>42</td></tr>
                    <tr><td><strong>Neutral</strong></td><td>53</td><td>29</td><td>10</td><td>56</td><td style="background-color: #bbdefb; font-weight: bold;">737</td><td>47</td><td>68</td></tr>
                    <tr><td><strong>Sad</strong></td><td>16</td><td>1</td><td>5</td><td>11</td><td>19</td><td style="background-color: #bbdefb; font-weight: bold;">934</td><td>14</td></tr>
                    <tr><td><strong>Surprise</strong></td><td>19</td><td>3</td><td>3</td><td>16</td><td>22</td><td>15</td><td style="background-color: #bbdefb; font-weight: bold;">922</td></tr>
                </tbody>
            </table>
        </td>
    </tr>
</table>

<table class="two-col-figures" style="margin-top: 2px;">
    <tr>
        <td style="width: 48%;">
            <img src="{caers_recall_img}" alt="CAER-S Recall Barchart" style="max-height: 115px;">
            <div class="figure-caption">Hình 11a: Biểu đồ thanh Per-Class Recall trên 7 lớp cảm xúc CAER-S</div>
        </td>
        <td style="width: 52%;">
            <img src="{caers_cm_img}" alt="CAER-S Confusion Matrix" style="max-height: 115px;">
            <div class="figure-caption">Hình 11b: Bản đồ nhiệt Ma trận nhầm lẫn thực nghiệm trên 7,000 mẫu CAER-S</div>
        </td>
    </tr>
</table>

<h3 class="subsection-title" style="margin-top: 3px;">Trực Quan Hóa Grad-CAM Đa Luồng Trên CAER-S (Model: Caers-finetune.pth)</h3>
<table class="three-col-figures">
    <tr>
        <td>
            <img src="{caers_grad_1}" alt="CAER-S Grad-CAM Anger" style="max-height: 105px;">
            <div class="figure-caption">Hình 12a: Mẫu <strong>Anger</strong> (95.2%)</div>
        </td>
        <td>
            <img src="{caers_grad_2}" alt="CAER-S Grad-CAM Happy" style="max-height: 105px;">
            <div class="figure-caption">Hình 12b: Mẫu <strong>Happy</strong> (95.1%)</div>
        </td>
        <td>
            <img src="{caers_grad_3}" alt="CAER-S Grad-CAM Fear" style="max-height: 105px;">
            <div class="figure-caption">Hình 12c: Mẫu <strong>Fear</strong> (93.6%)</div>
        </td>
    </tr>
</table>

</body>
</html>
"""

    print("=> Đang chuyển đổi HTML sang PDF bằng WeasyPrint...")
    weasyprint.HTML(string=html).write_pdf(output_pdf)
    print(f"=> ĐÃ XUẤT THÀNH CÔNG FILE BÁO CÁO ABLATION: {output_pdf}")
    print(f"=> Dung lượng file: {os.path.getsize(output_pdf):,} bytes")
    return output_pdf

if __name__ == "__main__":
    build_pdf()
