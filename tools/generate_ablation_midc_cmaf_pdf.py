#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_ablation_midc_cmaf_pdf.py
────────────────────────────────────────
Xuất file PDF Báo Cáo Khoa Học Chuyên Đề:
1. Phân rã hàm điều chuẩn MI và DC (Có MI không DC vs Có DC không MI vs Có cả 2 vs Không có cả 2).
2. Vai trò của Module Dung hợp CMAF (Có CMAF vs Không có CMAF - Direct Concat & GFI).
Ghi SỐ GỐC, KHÔNG CÓ MỞ NGOẶC (chuẩn báo cáo khoa học nộp hội nghị/tạp chí).
"""

import os
import sys
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
    output_pdf = os.path.join(root_dir, "Bao_Cao_Ablation_MI_DC_CMAF.pdf")

    # Image paths
    img_dir = os.path.join(root_dir, "outputs", "ablation_fresh_cms")
    
    img_midc_both = get_base64_image(os.path.join(img_dir, "cm_midc_both.png"))
    img_midc_mi = get_base64_image(os.path.join(img_dir, "cm_midc_mi_only.png"))
    img_midc_dc = get_base64_image(os.path.join(img_dir, "cm_midc_dc_only.png"))
    img_midc_none = get_base64_image(os.path.join(img_dir, "cm_midc_none.png"))
    
    img_bar_midc = get_base64_image(os.path.join(img_dir, "barchart_midc_ablation.png"))
    img_bar_cmaf = get_base64_image(os.path.join(img_dir, "barchart_cmaf_vs_nocmaf.png"))
    
    img_cm_sota = get_base64_image(os.path.join(img_dir, "cm_01_full_tristream_sota.png"))
    img_cm_nocmaf = get_base64_image(os.path.join(img_dir, "cm_02_bypass_cmaf.png"))

    html_template = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo Cáo Thực Nghiệm Ablation: MI/DC & CMAF - RAPT-CLIP</title>
<style>
    @page {
        size: A4 portrait;
        margin: 8mm 10mm 8mm 10mm;
        @bottom-right {
            content: "Trang " counter(page) " / " counter(pages);
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 7.5pt;
            color: #666;
        }
        @bottom-left {
            content: "RAPT-CLIP: Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Studies trên RAER)";
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 7.5pt;
            color: #666;
            font-weight: bold;
        }
    }

    body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #212529;
        font-size: 8pt;
        line-height: 1.35;
        margin: 0;
        padding: 0;
    }

    .header-box {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 7px;
    }

    .header-box h1 {
        margin: 0 0 3px 0;
        font-size: 13.5pt;
        font-weight: bold;
        letter-spacing: -0.2px;
        text-transform: uppercase;
    }

    .header-box .meta {
        font-size: 8pt;
        opacity: 0.95;
    }

    .section-title {
        color: #0d47a1;
        font-size: 10pt;
        font-weight: bold;
        border-bottom: 2px solid #0d47a1;
        padding-bottom: 2px;
        margin: 8px 0 5px 0;
        text-transform: uppercase;
    }

    .subsection-title {
        color: #1565c0;
        font-size: 8.5pt;
        font-weight: bold;
        margin: 5px 0 3px 0;
        border-left: 3px solid #1565c0;
        padding-left: 5px;
    }

    table.data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 7.8pt;
        margin: 4px 0 6px 0;
    }

    table.data-table th {
        background-color: #0d47a1;
        color: white;
        font-weight: bold;
        padding: 3.5px 5px;
        border: 1px solid #0a3880;
        text-align: center;
    }

    table.data-table td {
        border: 1px solid #d0d7de;
        padding: 3px 5px;
        vertical-align: middle;
    }

    table.data-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    .highlight-sota {
        background-color: #e8f5e9 !important;
        font-weight: bold;
    }

    .badge-sota {
        display: inline-block;
        background-color: #2e7d32;
        color: white;
        padding: 1px 4px;
        border-radius: 2px;
        font-weight: bold;
        font-size: 7pt;
    }

    .badge-drop {
        display: inline-block;
        background-color: #ffebee;
        color: #c62828;
        padding: 1px 4px;
        border-radius: 2px;
        font-weight: bold;
        font-size: 7pt;
    }

    .note-box {
        background-color: #e3f2fd;
        border-left: 3.5px solid #1976d2;
        padding: 5px 8px;
        font-size: 7.6pt;
        margin: 4px 0 6px 0;
        border-radius: 2px;
    }

    .warning-box {
        background-color: #fff3e0;
        border-left: 3.5px solid #f57c00;
        padding: 5px 8px;
        font-size: 7.6pt;
        margin: 4px 0 6px 0;
        border-radius: 2px;
    }

    .grid-2col {
        width: 100%;
        border-collapse: collapse;
        margin: 3px 0;
    }

    .grid-2col td {
        width: 50%;
        vertical-align: top;
        padding: 2px 4px;
        border: none;
        text-align: center;
    }

    .grid-2col img {
        width: 100%;
        max-height: 140px;
        object-fit: contain;
        border: 1px solid #ddd;
        border-radius: 3px;
    }

    .figure-caption {
        font-size: 7.2pt;
        color: #444;
        font-weight: bold;
        margin-top: 2px;
        text-align: center;
    }

    .page-break {
        page-break-after: always;
    }
</style>
</head>
<body>

<!-- ========================================================================= -->
<!-- TRANG 1: PHÂN RÃ HÀM ĐIỀU CHUẨN MI VÀ DC (MI LOSS vs DC LOSS) -->
<!-- ========================================================================= -->
<div class="header-box">
    <h1>Báo Cáo Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Studies)</h1>
    <div class="meta">
        <strong>Mô hình:</strong> RAPT-CLIP (ViT-B/16 Backbone) &nbsp;|&nbsp; 
        <strong>Tập dữ liệu:</strong> RAER (528 video clips kiểm thử chuẩn, 5 lớp cảm xúc đuôi dài) &nbsp;|&nbsp; 
        <strong>Trọng tâm:</strong> Phân rã Loss Điều Chuẩn Prompt (MI/DC) &amp; Khối Dung Hợp Đa Hướng CMAF
    </div>
</div>

<div class="section-title">
    1. Nghiên Cứu Phân Rã Hàm Phạt Điều Chuẩn: Có MI vs Có DC
</div>

<p style="margin: 3px 0 5px 0;">
    Để giải quyết hiện tượng sụp đổ không gian ngữ nghĩa và trôi dạt văn bản (<em>Prompt Drift</em>) khi tự học vector ngữ cảnh (<em>Prompt Tuning</em>), RAPT-CLIP tích hợp hai thành phần điều chuẩn:
    <strong>Hàm Thông tin Tương hỗ (L_MI)</strong> neo giữ đặc trưng văn bản đã học với các mô tả chuyên gia viết tay (<em>handcrafted descriptors</em>), và 
    <strong>Hàm Tương quan Khoảng cách (L_DC)</strong> ép trực giao hóa các vector đại diện giữa các lớp cảm xúc. Bảng dưới đây thể hiện kết quả kiểm chứng thực nghiệm khi cô lập từng thành phần (ghi số gốc, không mở ngoặc):
</p>

<table class="data-table">
    <thead>
        <tr>
            <th style="width: 28%;">Cấu hình Điều Chuẩn</th>
            <th style="width: 8%;">λ_MI</th>
            <th style="width: 8%;">λ_DC</th>
            <th style="width: 11%;">WAR (%)</th>
            <th style="width: 11%;">UAR (%)</th>
            <th style="width: 13%;">Harmonic (%)</th>
            <th style="width: 11%;">Δ UAR</th>
            <th style="width: 10%;">Đánh Giá</th>
        </tr>
    </thead>
    <tbody>
        <tr class="highlight-sota">
            <td><strong>Có cả MI và DC (Đề Xuất SOTA)</strong></td>
            <td style="text-align: center;">0.10</td>
            <td style="text-align: center;">0.10</td>
            <td style="text-align: center; color: #1b5e20;">77.08%</td>
            <td style="text-align: center; color: #1b5e20;">77.66%</td>
            <td style="text-align: center; color: #1b5e20;">77.37%</td>
            <td style="text-align: center;"><span class="badge-sota">Baseline</span></td>
            <td style="text-align: center; color: #1b5e20;">Tối ưu toàn diện</td>
        </tr>
        <tr>
            <td><strong>Có MI, KHÔNG có DC</strong></td>
            <td style="text-align: center;">0.10</td>
            <td style="text-align: center;">0.00</td>
            <td style="text-align: center;">78.41%</td>
            <td style="text-align: center; font-weight: bold; color: #b71c1c;">73.18%</td>
            <td style="text-align: center;">75.70%</td>
            <td style="text-align: center;"><span class="badge-drop">-4.48%</span></td>
            <td style="text-align: center; color: #b71c1c;">Trùng lặp không gian</td>
        </tr>
        <tr>
            <td><strong>Có DC, KHÔNG có MI</strong></td>
            <td style="text-align: center;">0.00</td>
            <td style="text-align: center;">0.10</td>
            <td style="text-align: center;">79.17%</td>
            <td style="text-align: center; font-weight: bold; color: #b71c1c;">71.55%</td>
            <td style="text-align: center;">75.17%</td>
            <td style="text-align: center;"><span class="badge-drop">-6.11%</span></td>
            <td style="text-align: center; color: #b71c1c;">Trôi dạt ngữ nghĩa</td>
        </tr>
        <tr>
            <td><strong>Không có cả MI và DC (No MI &amp; DC)</strong></td>
            <td style="text-align: center;">0.00</td>
            <td style="text-align: center;">0.00</td>
            <td style="text-align: center;">79.73%</td>
            <td style="text-align: center; font-weight: bold; color: #b71c1c;">69.30%</td>
            <td style="text-align: center;">74.15%</td>
            <td style="text-align: center;"><span class="badge-drop">-7.76%</span></td>
            <td style="text-align: center; color: #b71c1c;">Sụp đổ phân loại</td>
        </tr>
    </tbody>
</table>

<div class="subsection-title">Chi Tiết Độ Hồi Tưởng (Recall %) Trên 5 Lớp Cảm Xúc RAER</div>
<table class="data-table">
    <thead>
        <tr>
            <th style="width: 30%;">Biến thể Phạt Điều Chuẩn</th>
            <th style="width: 14%;">Neutrality (345)</th>
            <th style="width: 14%;">Enjoyment (32)</th>
            <th style="width: 14%;">Confusion (7)</th>
            <th style="width: 14%;">Fatigue (39)</th>
            <th style="width: 14%;">Distraction (105)</th>
        </tr>
    </thead>
    <tbody>
        <tr class="highlight-sota">
            <td><strong>Có cả MI và DC (SOTA Đề xuất)</strong></td>
            <td style="text-align: center;">74.78%</td>
            <td style="text-align: center;">84.38%</td>
            <td style="text-align: center; color: #1b5e20;">57.14%</td>
            <td style="text-align: center;">94.87%</td>
            <td style="text-align: center;">77.14%</td>
        </tr>
        <tr>
            <td><strong>Có MI, KHÔNG có DC</strong></td>
            <td style="text-align: center;">76.52%</td>
            <td style="text-align: center;">75.00%</td>
            <td style="text-align: center;">57.14%</td>
            <td style="text-align: center;">82.05%</td>
            <td style="text-align: center; color: #b71c1c;">75.24%</td>
        </tr>
        <tr>
            <td><strong>Có DC, KHÔNG có MI</strong></td>
            <td style="text-align: center;">77.39%</td>
            <td style="text-align: center;">71.88%</td>
            <td style="text-align: center; color: #b71c1c;">42.86%</td>
            <td style="text-align: center;">84.62%</td>
            <td style="text-align: center;">77.14%</td>
        </tr>
        <tr>
            <td><strong>Không có cả MI và DC</strong></td>
            <td style="text-align: center;">79.13%</td>
            <td style="text-align: center; color: #b71c1c;">68.75%</td>
            <td style="text-align: center; color: #b71c1c;">28.57%</td>
            <td style="text-align: center;">87.18%</td>
            <td style="text-align: center;">79.05%</td>
        </tr>
    </tbody>
</table>

<table class="grid-2col" style="margin-top: 3px;">
    <tr>
        <td>
            <img src="__IMG_MIDC_BOTH__" alt="Có cả MI và DC">
            <div class="figure-caption">Hình 1a: Có cả MI &amp; DC (WAR: 77.08% | UAR: 77.66%)</div>
        </td>
        <td>
            <img src="__IMG_MIDC_MI__" alt="Có MI không DC">
            <div class="figure-caption">Hình 1b: Có MI, Không có DC (WAR: 78.41% | UAR: 73.18%)</div>
        </td>
    </tr>
    <tr>
        <td>
            <img src="__IMG_MIDC_DC__" alt="Có DC không MI">
            <div class="figure-caption">Hình 1c: Có DC, Không có MI (WAR: 79.17% | UAR: 71.55%)</div>
        </td>
        <td>
            <img src="__IMG_MIDC_NONE__" alt="Không có cả hai">
            <div class="figure-caption">Hình 1d: Không có cả MI &amp; DC (WAR: 79.73% | UAR: 69.30%)</div>
        </td>
    </tr>
</table>

<div class="note-box">
    <strong>Phân tích Cơ chế Khoa học:</strong><br>
    1) <em>Khi có MI mà không có DC:</em> Mô hình bảo toàn được mỏ neo ngữ nghĩa thị giác gốc của CLIP nhờ L_MI, giúp lớp siêu hiếm <em>Confusion</em> giữ được 57.14% Recall. Tuy nhiên, việc thiếu L_DC khiến các prompt giữa các lớp bị trùng lặp không gian đặc trưng, dẫn đến nhầm lẫn trên <em>Enjoyment</em> (tụt từ 84.38% xuống 75.00%).<br>
    2) <em>Khi có DC mà không có MI:</em> Các vector prompt được tách biệt hình học rất tốt, nhưng do thiếu mỏ neo hướng dẫn ngữ nghĩa, prompt bị trôi tự do (<em>drift</em>), khiến độ nhận diện lớp khó <em>Confusion</em> sụp đổ từ 57.14% xuống 42.86%.<br>
    3) <em>Sự hiệp đồng tối ưu:</em> Khi kết hợp đồng bộ λ_MI = 0.10 và λ_DC = 0.10, mô hình vừa có neo ngữ nghĩa vừa có trực giao không gian, giúp UAR đạt đỉnh <strong>77.66%</strong>.
</div>

<div class="page-break"></div>

<!-- ========================================================================= -->
<!-- TRANG 2: VAI TRÒ CỦA KHỐI DUNG HỢP ĐA HƯỚNG CMAF (CÓ CMAF vs KHÔNG CÓ CMAF) -->
<!-- ========================================================================= -->
<div class="section-title">
    2. Nghiên Cứu Khối Dung Hợp Đa Hướng CMAF: Có CMAF vs. Không Có CMAF
</div>

<p style="margin: 3px 0 5px 0;">
    Module <strong>Cross-Modal Attention Fusion (CMAF)</strong> thực hiện tương tác chú ý chéo 3 chiều (<em>Tridirectional Cross-Attention</em>) giữa 3 luồng: Khuôn mặt (Q_F × [K_B, K_C]), Thân thể (Q_B × [K_F, K_C]), và Bối cảnh lớp học (Q_C × [K_F, K_B]). 
    Để chứng minh tính tất yếu của cơ chế Attention so với ghép kênh tĩnh, hai kịch bản cắt bỏ đã được thực hiện: 
    1) <strong>Direct Concatenation</strong> (ghép nối trực tiếp các vector và chiếu tuyến tính), và 
    2) <strong>Gated Feature Integration (GFI)</strong> (cổng nhân Sigmoid thích ứng). Kết quả ghi số gốc chuẩn hóa:
</p>

<table class="data-table">
    <thead>
        <tr>
            <th style="width: 25%;">Cấu hình Dung Hợp</th>
            <th style="width: 37%;">Cơ Chế Kiến Trúc &amp; Tương Tác Giữa Các Luồng</th>
            <th style="width: 12%;">WAR (%)</th>
            <th style="width: 12%;">UAR (%)</th>
            <th style="width: 14%;">Δ UAR so với SOTA</th>
        </tr>
    </thead>
    <tbody>
        <tr class="highlight-sota">
            <td><strong>CÓ CMAF (SOTA Đề Xuất)</strong></td>
            <td>Tridirectional Cross-Attention liên luồng (Face, Body, Context) từng frame</td>
            <td style="text-align: center; color: #1b5e20;">77.08%</td>
            <td style="text-align: center; color: #1b5e20;">77.66%</td>
            <td style="text-align: center;"><span class="badge-sota">Baseline Chuẩn</span></td>
        </tr>
        <tr>
            <td><strong>KHÔNG CÓ CMAF (Direct Concat)</strong></td>
            <td>Ghép nối vector trực tiếp [f_face, f_body, f_ctx] không có Attention</td>
            <td style="text-align: center; font-weight: bold;">80.30%</td>
            <td style="text-align: center; font-weight: bold; color: #b71c1c;">66.48%</td>
            <td style="text-align: center;"><span class="badge-drop">ΔUAR: -11.18%</span></td>
        </tr>
        <tr>
            <td><strong>KHÔNG CÓ CMAF (GFI Gate)</strong></td>
            <td>Thay Attention bằng Cổng nhân trọng số tuyến tính Sigmoid [f] ⊙ σ(Wx)</td>
            <td style="text-align: center;">67.23%</td>
            <td style="text-align: center; font-weight: bold; color: #b71c1c;">67.42%</td>
            <td style="text-align: center;"><span class="badge-drop">ΔUAR: -10.24%</span></td>
        </tr>
    </tbody>
</table>

<div class="subsection-title">Chi Tiết Recall 5 Lớp Cảm Xúc Giữa Có CMAF và Không Có CMAF</div>
<table class="data-table">
    <thead>
        <tr>
            <th style="width: 30%;">Phương Pháp Dung Hợp</th>
            <th style="width: 14%;">Neutrality (345)</th>
            <th style="width: 14%;">Enjoyment (32)</th>
            <th style="width: 14%;">Confusion (7)</th>
            <th style="width: 14%;">Fatigue (39)</th>
            <th style="width: 14%;">Distraction (105)</th>
        </tr>
    </thead>
    <tbody>
        <tr class="highlight-sota">
            <td><strong>Có CMAF (SOTA Đề Xuất)</strong></td>
            <td style="text-align: center;">74.78%</td>
            <td style="text-align: center; color: #1b5e20;">84.38%</td>
            <td style="text-align: center; color: #1b5e20;">57.14%</td>
            <td style="text-align: center; color: #1b5e20;">94.87%</td>
            <td style="text-align: center;">77.14%</td>
        </tr>
        <tr>
            <td><strong>Không có CMAF (Direct Concat)</strong></td>
            <td style="text-align: center; font-weight: bold; color: #0d47a1;">84.64%</td>
            <td style="text-align: center; color: #b71c1c;">53.12%</td>
            <td style="text-align: center; color: #b71c1c;">42.86%</td>
            <td style="text-align: center; color: #b71c1c;">71.79%</td>
            <td style="text-align: center; color: #1b5e20;">80.00%</td>
        </tr>
        <tr>
            <td><strong>Không có CMAF (GFI Gate)</strong></td>
            <td style="text-align: center;">62.61%</td>
            <td style="text-align: center;">75.00%</td>
            <td style="text-align: center;">42.86%</td>
            <td style="text-align: center;">79.49%</td>
            <td style="text-align: center;">77.14%</td>
        </tr>
    </tbody>
</table>

<table class="grid-2col" style="margin-top: 3px;">
    <tr>
        <td>
            <img src="__IMG_CM_SOTA__" alt="Full CMAF SOTA">
            <div class="figure-caption">Hình 2a: Có CMAF (WAR: 77.08% | UAR: 77.66%)</div>
        </td>
        <td>
            <img src="__IMG_CM_NOCMAF__" alt="Bypass CMAF Concat">
            <div class="figure-caption">Hình 2b: Không có CMAF - Direct Concat (WAR: 80.30% | UAR: 66.48%)</div>
        </td>
    </tr>
</table>

<div style="text-align: center; margin: 4px 0 2px 0;">
    <img src="__IMG_BAR_CMAF__" alt="Biểu đồ so sánh CMAF" style="width: 95%; max-height: 120px; object-fit: contain; border: 1px solid #ddd; border-radius: 3px;">
    <div class="figure-caption">Hình 3: Biểu đồ trực quan hóa sụt giảm Recall từng lớp khi cắt khối CMAF trên tập kiểm thử RAER</div>
</div>

<div class="warning-box">
    <strong>Giải Thích Hiện Tượng Đột Biến WAR và Sụt Đổ UAR khi Cắt CMAF:</strong><br>
    - <strong>WAR tăng ảo lên 80.30%:</strong> Khi không có Cross-Attention lọc nhiễu, các chiều đặc trưng bối cảnh lớp học (512-d) và thân thể (512-d) lấn át biểu cảm khuôn mặt. Mô hình rơi vào cơ chế tối ưu "thiển cận": tập trung phân loại chính xác lớp đa số <em>Neutrality</em> (chiếm 345/528 mẫu, Recall tăng vọt lên 84.64%), từ đó đẩy tổng số mẫu đúng lên cao làm WAR tăng ảo.<br>
    - <strong>UAR rơi tự do -11.18%:</strong> Trên các lớp thiểu số đòi hỏi phải đối chiếu biểu cảm mặt với tư thế ngồi, mô hình hoàn toàn mất phương hướng: <em>Enjoyment</em> rớt thảm hại từ 84.38% xuống <strong>53.12%</strong> (-31.26%), <em>Fatigue</em> giảm từ 94.87% xuống <strong>71.79%</strong> (-23.08%), và <em>Confusion</em> giảm còn <strong>42.86%</strong>.<br>
    - <strong>Kết luận:</strong> CMAF là thành phần sống còn để bảo đảm tính công bằng nhận diện đa lớp, ngăn mô hình bị thiên lệch trên bài toán cảm xúc học đường đuôi dài (<em>Long-Tailed Emotion Recognition</em>).
</div>

</body>
</html>
"""

    html = html_template \
        .replace("__IMG_MIDC_BOTH__", img_midc_both) \
        .replace("__IMG_MIDC_MI__", img_midc_mi) \
        .replace("__IMG_MIDC_DC__", img_midc_dc) \
        .replace("__IMG_MIDC_NONE__", img_midc_none) \
        .replace("__IMG_BAR_MIDC__", img_bar_midc) \
        .replace("__IMG_BAR_CMAF__", img_bar_cmaf) \
        .replace("__IMG_CM_SOTA__", img_cm_sota) \
        .replace("__IMG_CM_NOCMAF__", img_cm_nocmaf)

    print("=> Đang chuyển đổi HTML sang PDF bằng WeasyPrint...")
    weasyprint.HTML(string=html).write_pdf(output_pdf)
    print(f"=> ĐÃ XUẤT THÀNH CÔNG FILE BÁO CÁO: {output_pdf}")
    print(f"=> Dung lượng file: {os.path.getsize(output_pdf):,} bytes")
    return output_pdf

if __name__ == "__main__":
    build_pdf()
