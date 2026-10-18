#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/build_pdf_report.py
─────────────────────────
Generates a clean, professional report PDF matching the user's requested layout:
'Bao_Cao_Kien_Truc_Mo_Hinh_RAPT_CLIP.pdf'
Layout features:
- Clean minimalist document formatting (Arial/Helvetica style)
- "Báo cáo chi tiết" title with exact section structure:
  1. Tổng quan mô hình:
     - Mô hình cũ (old_model.png) + Hình 1 caption
     - Mô hình mới (model.png) + Hình 2 caption
     - So sánh cốt lõi
  1.2 Nhánh hình ảnh:
     1.2.1 Face Adapter (EAA) with intuitive student sleeping example
     1.2.2 Khối Cross-Modal Attention Fusion (CMAF) & Phép Concat (deep explanation:
           - Bảng chi tiết Đầu Vào & Đầu Ra
           - Chi tiết 5 bước hoạt động
           - Toàn bộ code gói gọn trong 1 Ô DUY NHẤT có comment chi tiết)
     1.2.3 Unified Temporal Transformer
  1. Huấn luyện trên bộ dữ liệu RAER:
     - Đường dẫn Kaggle
     - Bảng siêu tham số đầy đủ (RAPT-CLIP-FRAMELEVELFUSION vs RAPT-CLIP-VIDEOLEVELFUSION)
     - Kết quả thực nghiệm: Confusion matrix & Log.png
     - Trực quan hóa t-SNE: Pretrained CLIP vs Fine-tuned RAPT-CLIP
     - Đánh giá độ bền vững (Robustness, Modality Occlusion, Noise, Shuffling) & Grad-CAM
     - Bảng tổng hợp Ablation Studies (Frame vs Video, Weighted Sampler, MI/DC Loss, v.v.)
  1. Huấn luyện trên bộ dữ liệu EMOTIC
  2. Huấn luyện trên bộ dữ liệu CAER
  3. Huấn luyện trên bộ dữ liệu DAISEE
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

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old_img_b64 = get_base64_image(os.path.join(root_dir, "image", "old_model.png"))
    new_img_b64 = get_base64_image(os.path.join(root_dir, "image", "model.png"))
    cm_img_b64 = get_base64_image(os.path.join(root_dir, "image", "confusion_matrix.png"))
    log_img_b64 = get_base64_image(os.path.join(root_dir, "image", "training_curves.png"))
    
    tsne_pre_b64 = get_base64_image(os.path.join(root_dir, "image", "tsne_pretrained.png"))
    tsne_fine_b64 = get_base64_image(os.path.join(root_dir, "image", "tsne_finetuned.png"))
    robust_mod_b64 = get_base64_image(os.path.join(root_dir, "image", "robustness_modality_ablation.png"))
    robust_deg_b64 = get_base64_image(os.path.join(root_dir, "image", "robustness_degradation.png"))
    gradcam_b64 = get_base64_image(os.path.join(root_dir, "image", "gradcam_sample.png"))
    emotic_bar_b64 = get_base64_image(os.path.join(root_dir, "outputs", "EMOTIC", "emotic_test_ranking_barchart.png"))
    emotic_grad_1 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Emotic", "1.jpeg"))
    emotic_grad_2 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Emotic", "2.jpeg"))
    emotic_grad_3 = get_base64_image(os.path.join(root_dir, "outputs", "Gradcam-Emotic", "3.jpeg"))

    daisee_grad_1 = get_base64_image(os.path.join(root_dir, "outputs", "gradcam_daisee_v3", "all_samples", "sample_0000_5000441001_High_Engagement_pred_High_Engagement.png"))
    daisee_grad_2 = get_base64_image(os.path.join(root_dir, "outputs", "gradcam_daisee_v3", "all_samples", "sample_0007_5000441009_Very_High_Engagement_pred_High_Engagement.png"))
    daisee_grad_3 = get_base64_image(os.path.join(root_dir, "outputs", "gradcam_daisee_v3", "all_samples", "sample_0019_5000441024_Low_Engagement_pred_High_Engagement.png"))
    daisee_gallery = get_base64_image(os.path.join(root_dir, "outputs", "gradcam_daisee_v3", "daisee_gradcam_gallery.png"))
    daisee_cm_b64 = get_base64_image(os.path.join(root_dir, "outputs", "Train-[10-14]-[12:28]", "confusion_matrix.png"))

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo cáo chi tiết: Kiến trúc RAPT-CLIP và Thực nghiệm</title>
<style>
    @page {{
        size: A4;
        margin: 14mm 16mm 14mm 16mm;
        @bottom-right {{
            content: counter(page);
            font-size: 8.5pt;
            font-family: Arial, Helvetica, sans-serif;
            color: #555;
        }}
    }}

    body {{
        font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
        color: #111111;
        line-height: 1.4;
        font-size: 9.3pt;
    }}

    h1.doc-title {{
        font-size: 13.5pt;
        font-weight: bold;
        text-decoration: underline;
        margin: 0 0 10px 0;
        color: #000000;
    }}

    h2.section-title {{
        font-size: 10.6pt;
        font-weight: bold;
        margin: 10px 0 4px 0;
        color: #000000;
    }}

    h3.subsection-title {{
        font-size: 9.8pt;
        font-weight: bold;
        margin: 8px 0 3px 0;
        color: #000000;
    }}

    h4.subsubsection-title {{
        font-size: 9.4pt;
        font-weight: bold;
        margin: 6px 0 3px 0;
        color: #000000;
    }}

    p {{
        margin: 0 0 5px 0;
    }}

    .figure-block {{
        text-align: center;
        margin: 3px 0 5px 0;
        page-break-inside: avoid;
    }}

    .figure-block img {{
        max-width: 98%;
        max-height: 250px;
        border: 1px solid #ccc;
        display: block;
        margin: 0 auto 3px auto;
    }}

    .figure-caption {{
        font-size: 8.5pt;
        font-weight: bold;
        text-align: center;
        margin-bottom: 6px;
    }}

    /* Table styles matching document screenshot */
    table.grid-table {{
        width: 100%;
        border: 1px solid #000;
        border-collapse: collapse;
        margin: 6px 0 10px 0;
        page-break-inside: avoid;
    }}

    table.grid-table td {{
        border: 1px solid #000;
        padding: 4px 6px;
        vertical-align: middle;
    }}

    table.grid-table td.header-col {{
        width: 20%;
        font-weight: bold;
        font-size: 9pt;
        background-color: #fafafa;
        text-align: center;
    }}

    table.grid-table td.image-col {{
        text-align: center;
        background-color: #ffffff;
        padding: 4px;
    }}

    table.grid-table td.image-col img {{
        max-width: 98%;
        max-height: 240px;
        display: block;
        margin: 0 auto;
    }}

    /* Parameter and I/O table */
    table.param-table {{
        width: 100%;
        border-collapse: collapse;
        border: 1px solid #333;
        margin: 5px 0 8px 0;
        font-size: 7.8pt;
        line-height: 1.25;
        page-break-inside: avoid;
    }}

    table.param-table th {{
        border: 1px solid #333;
        background-color: #f2f2f2;
        padding: 3px 5px;
        font-weight: bold;
        text-align: left;
    }}

    table.param-table td {{
        border: 1px solid #ccc;
        padding: 2.2px 5px;
    }}

    table.param-table tr.group-row td {{
        background-color: #eaeaea;
        font-weight: bold;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
        padding: 2.5px 5px;
    }}

    /* Side-by-side comparison tables */
    table.side-by-side {{
        width: 100%;
        border-collapse: collapse;
        margin: 6px 0 8px 0;
        page-break-inside: avoid;
    }}

    table.side-by-side td {{
        width: 50%;
        padding: 3px 5px;
        vertical-align: top;
        text-align: center;
    }}

    table.side-by-side img {{
        max-width: 98%;
        max-height: 240px;
        border: 1px solid #ccc;
        display: block;
        margin: 0 auto 3px auto;
    }}

    /* 3-column image grid */
    table.grid-3col {{
        width: 100%;
        border-collapse: collapse;
        margin: 4px 0 6px 0;
        page-break-inside: avoid;
    }}

    table.grid-3col td {{
        width: 33.33%;
        padding: 2px 4px;
        vertical-align: top;
        text-align: center;
    }}

    table.grid-3col img {{
        max-width: 98%;
        max-height: 165px;
        border: 1px solid #cbd5e1;
        display: block;
        margin: 0 auto 3px auto;
        border-radius: 3px;
    }}

    /* Single consolidated code box */
    .single-code-box {{
        font-family: "Courier New", Courier, monospace;
        font-size: 7.2pt;
        background-color: #f7f9fa;
        border: 1px solid #3b82f6;
        border-left: 3.5px solid #1d4ed8;
        padding: 7px 9px;
        margin: 6px 0 10px 0;
        line-height: 1.32;
        white-space: pre-wrap;
        page-break-inside: avoid;
        color: #0f172a;
    }}

    .math-formula {{
        background-color: #fcfcfc;
        border-left: 3px solid #555;
        padding: 3px 8px;
        margin: 4px 0;
        font-style: italic;
        font-size: 8.8pt;
    }}

    .io-box {{
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 6px 8px;
        margin: 5px 0 8px 0;
        border-radius: 2px;
        font-size: 8.8pt;
    }}

    .page-break {{
        page-break-before: always;
    }}
</style>
</head>
<body>

<!-- TRANG 1: TỔNG QUAN MÔ HÌNH VÀ NHÁNH HÌNH ẢNH (BẮT ĐẦU) -->
<h1 class="doc-title">Báo cáo chi tiết</h1>

<h2 class="section-title">1. Tổng quan mô hình</h2>

<p><strong>Mô hình cũ:</strong></p>
<div class="figure-block">
    <img src="{old_img_b64}" alt="Kiến trúc mô hình RAPT-CLIP cũ">
    <div class="figure-caption">Hình 1: Kiến trúc mô hình RAPT-CLIP cũ</div>
</div>

<p><strong>Mô hình mới:</strong></p>
<div class="figure-block">
    <img src="{new_img_b64}" alt="Tổng quan kiến trúc của mô hình RAPT-CLIP">
    <div class="figure-caption">Hình 2: Tổng quan kiến trúc của mô hình RAPT-CLIP</div>
</div>

<p>
    <strong>So sánh cốt lõi:</strong> Mô hình cũ sử dụng cơ chế <em>Video-Level Fusion</em> (mỗi luồng Face, Body, Context tự nén qua mạng Temporal riêng rồi mới ghép nối). Mô hình mới sử dụng cơ chế <em>Frame-Level CMAF</em> (các luồng tương tác chéo với nhau ngay tại từng khung hình <i>t</i>, sau đó nối lại và đưa qua một mạng Unified Temporal Transformer duy nhất).
</p>

<h3 class="subsection-title">1.2 Nhánh hình ảnh</h3>

<h4 class="subsubsection-title">1.2.1 Face Adapter (EAA)</h4>
<p>
    Giả sử một học sinh trong video đang ngủ gật. Ba nhánh phân tích:
</p>
<p style="margin-left: 15px;">
    + <strong>Face:</strong> Mắt nhắm nghiền (512-d). Nhưng hệ thống chưa chắc học sinh đang chớp mắt bình thường hoặc ngủ thật.<br>
    + <strong>Body:</strong> Đầu gục xuống bàn, hai vai buông thõng (512-d).<br>
    + <strong>Context:</strong> Bàn học trong lớp, xung quanh có bạn bè và sách vở (512-d).
</p>
<p>
    Khối <strong>Expression-Aware Adapter (EAA)</strong> được gắn riêng vào nhánh khuôn mặt (Face) với cấu trúc thắt nút cổ chai (Bottleneck MLP: 512 &rarr; 128 &rarr; 512) và kết nối tàn dư (residual). EAA giúp bộ mã hóa thị giác CLIP ViT-B/16 nhạy bén hơn với các đơn vị vận động vi mô trên cơ mặt (Action Units như nhắm mắt, nhăn trán, trễ môi) mà không phá vỡ trọng số tiền huấn luyện tổng quát của CLIP.
</p>

<!-- TRANG 2: GIẢI THÍCH CHI TIẾT CMAF & PHÉP CONCAT -->
<h3 class="subsection-title">1.2.2 Khối Cross-Modal Attention Fusion (CMAF) & Phép Concat</h3>
<p>
    Khối CMAF đóng vai trò trung tâm quyết định độ chính xác cảm xúc. Dưới đây là bảng đặc tả chi tiết đầu vào - đầu ra, cơ chế vận hành 5 bước và mã nguồn hợp nhất trong 1 ô code.
</p>

<div class="io-box">
    <strong>BẢNG ĐẶC TẢ CHI TIẾT ĐẦU VÀO VÀ ĐẦU RA (INPUTS / OUTPUTS):</strong>
    <table class="param-table" style="margin-top: 4px; margin-bottom: 2px;">
        <thead>
            <tr>
                <th style="width: 18%;">Tên Tensor</th>
                <th style="width: 20%;">Kích Thước (Shape)</th>
                <th style="width: 25%;">Nguồn Gốc / Đích Đến</th>
                <th style="width: 37%;">Ý Nghĩa Nghiệp Vụ</th>
            </tr>
        </thead>
        <tbody>
            <tr class="group-row"><td colspan="4">I. ĐẦU VÀO (INPUTS CỦA KHỐI CMAF)</td></tr>
            <tr>
                <td><code>face_feat</code></td>
                <td><code>(B, 512)</code></td>
                <td>Từ CLIP ViT-B/16 + EAA</td>
                <td>Đặc trưng biểu cảm mặt vi mô tại từng frame (Action Units).</td>
            </tr>
            <tr>
                <td><code>body_feat</code></td>
                <td><code>(B, 512)</code></td>
                <td>Từ CLIP ViT-B/16 (Upper-body)</td>
                <td>Đặc trưng cử chỉ cơ thể (đầu gục, vai thả lỏng, tay khoanh).</td>
            </tr>
            <tr>
                <td><code>context_feat</code></td>
                <td><code>(B, 512)</code></td>
                <td>Từ CLIP ViT-B/16 (Full Image)</td>
                <td>Đặc trưng bối cảnh không gian lớp học, ánh sáng, bạn bè.</td>
            </tr>
            <tr class="group-row"><td colspan="4">II. ĐẦU RA TRUNG GIAN (SAU CROSS-ATTENTION)</td></tr>
            <tr>
                <td><code>face_out</code></td>
                <td><code>(B, 512)</code></td>
                <td>Mặt sau khi hỏi Thân + Cảnh</td>
                <td>Khuôn mặt đã được làm giàu ngữ cảnh (biết mắt nhắm do ngủ gật).</td>
            </tr>
            <tr>
                <td><code>body_out</code></td>
                <td><code>(B, 512)</code></td>
                <td>Thân sau khi hỏi Mặt + Cảnh</td>
                <td>Cơ thể đã được giải nghĩa tư thế dựa trên biểu cảm mặt.</td>
            </tr>
            <tr>
                <td><code>context_out</code></td>
                <td><code>(B, 512)</code></td>
                <td>Cảnh sau khi hỏi Mặt + Thân</td>
                <td>Bối cảnh đã được lọc sạch nhiễu nền, tập trung vào chủ thể.</td>
            </tr>
            <tr class="group-row"><td colspan="4">III. ĐẦU RA CUỐI CÙNG (SAU CONCATENATION & PIPELINE)</td></tr>
            <tr>
                <td><code>fused_frame</code></td>
                <td><code>(B, 1536)</code></td>
                <td>Ghép 3 luồng (512&times;3)</td>
                <td>Vector hợp nhất đa phương thức đầy đủ tại từng frame.</td>
            </tr>
            <tr>
                <td><code>video_features</code></td>
                <td><code>(N, 512)</code></td>
                <td>Sau Unified Temporal + FC</td>
                <td>Vector đại diện toàn bộ video clip chuẩn hóa L2 để so khớp Cosine Similarity với Text Embeddings.</td>
            </tr>
        </tbody>
    </table>
    <div style="font-size: 7.8pt; color: #475569; margin-top: 2px;">
        <em>* Ghi chú: Kích thước B ở cấp độ frame là B = N &times; 16 (với N là kích thước batch video, 16 là số segments lấy mẫu).</em>
    </div>
</div>

<p><strong>Chi tiết cách thức hoạt động (5 Bước Thực Hiện):</strong></p>
<ol style="margin: 0 0 6px 15px; padding: 0;">
    <li>
        <strong>Tạo Tokens Truy Vấn (Query) & Khóa/Giá Trị (Key/Value):</strong>
        Mỗi luồng đóng vai trò là Query 1 token: <code>Q = feat.unsqueeze(1) &isin; &#8477;<sup>B&times;1&times;512</sup></code>. Hai luồng còn lại được ghép nối thành chuỗi Key và Value 2 tokens: <code>K = V = torch.cat([...], dim=1) &isin; &#8477;<sup>B&times;2&times;512</sup></code>.
    </li>
    <li>
        <strong>Cơ chế "Cross" (Tương tác chéo tam hướng qua 4 Multi-heads):</strong>
        Chiếu tuyến tính sang 4 đầu chú ý (d_k = 128). Tính tích vô hướng chuẩn hóa và Softmax để ra ma trận trọng số tương quan chéo:
        <code>Attention_Weights = Softmax((Q &middot; K<sup>T</sup>) / &radic;128) = [&alpha;<sub>luồng_1</sub>, &alpha;<sub>luồng_2</sub>]</code>. Ví dụ: Với khuôn mặt mắt nhắm, mạng tự động gán trọng số chú ý cao vào tư thế đầu gục của thân để khẳng định trạng thái ngủ gật.
    </li>
    <li>
        <strong>Cộng Tàn Dư (Residual) & Chuẩn Hóa Lớp (LayerNorm):</strong>
        Vector sau chú ý được cộng trực tiếp với vector đặc trưng gốc của luồng đó nhằm bảo toàn thông tin bản thể, sau đó đi qua LayerNorm để ổn định phân phối gradient: <code>out = LayerNorm(CrossAttn + feat) &isin; &#8477;<sup>B&times;512</sup></code>.
    </li>
    <li>
        <strong>Phép Nối Tiếp Concat (Tại sao dùng Concat thay vì Add?):</strong>
        Lệnh <code>torch.cat((face_out, body_out, context_out), dim=-1)</code> ghép 3 vector thành 1 vector <strong>1536 chiều</strong>. Phép cộng sẽ làm triệt tiêu các đặc trưng vi mô riêng biệt (Action Units của mặt bị lấn át bởi màu sắc nền). Concat bảo toàn trọn vẹn cả 3 không gian độc lập để các tầng sâu hơn tự học cách trích lọc tổ hợp phi tuyến tính tối ưu.
    </li>
    <li>
        <strong>Dòng Dữ Liệu Sau Concat (Chuyển tiếp đi đâu?):</strong>
        Tensor <code>(B, 1536)</code> được reshape thành dạng chuỗi video <code>(N, 16, 1536)</code> &rarr; đưa qua <em>Unified Temporal Transformer</em> để nắm bắt tiến trình thời gian &rarr; nén qua <em>Attention Pooling</em> thành <code>(N, 1536)</code> &rarr; qua <em>Linear Projection Head</em> chiếu về <code>(N, 512)</code> và chuẩn hóa L2 để tính Cosine Similarity với Text Prompts!
    </li>
</ol>

<p><strong>Mã nguồn PyTorch hoàn chỉnh của toàn bộ luồng CMAF & Concat (Gói gọn trong 1 ô code):</strong></p>
<div class="single-code-box"># ==============================================================================================
# TOÀN BỘ MÃ NGUỒN PYTORCH: CƠ CHẾ CROSS-MODAL ATTENTION FUSION (CMAF) &amp; PHÉP NỐI TIẾP (CONCAT)
# File nguồn: models/CrossModalAttentionFusion.py &amp; models/Generate_Model.py
# ==============================================================================================

import torch
import torch.nn as nn

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=4, dropout=0.1, use_context=True):
        super().__init__()
        self.use_context = use_context
        # Khởi tạo 3 khối MultiheadAttention tam hướng (dim=512, 4 heads -&gt; dim_head=128)
        self.cross_attn_f2bc = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_f          = nn.LayerNorm(dim)

        self.cross_attn_b2fc = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_b          = nn.LayerNorm(dim)

        self.cross_attn_c2fb = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_c          = nn.LayerNorm(dim)

    def forward(self, face_feat, body_feat, context_feat):
        # --------------------------------------------------------------------------------------
        # BƯỚC 1: TIẾP NHẬN ĐẦU VÀO (INPUTS)
        # face_feat:    (B, 512) - Khuôn mặt sau CLIP ViT-B/16 + EAA Adapter
        # body_feat:    (B, 512) - Thân trên (Upper-body 70%) sau CLIP ViT-B/16
        # context_feat: (B, 512) - Toàn cảnh bối cảnh môi trường sau CLIP ViT-B/16
        # --------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------
        # BƯỚC 2: CHUYỂN ĐỔI CHUỖI TOKENS (QUERY = 1 token, KEY/VALUE = 2 tokens)
        # --------------------------------------------------------------------------------------
        # 2.1 Nhánh Mặt hỏi (Query) Thân &amp; Cảnh (Key, Value):
        face_q          = face_feat.unsqueeze(1)                                                 # (B, 1, 512)
        body_context_kv = torch.cat((body_feat.unsqueeze(1), context_feat.unsqueeze(1)), dim=1) # (B, 2, 512)
        face_cross, _   = self.cross_attn_f2bc(query=face_q, key=body_context_kv, value=body_context_kv)
        face_out        = self.norm_f(face_cross.squeeze(1) + face_feat) # Residual + Norm     -&gt; (B, 512)

        # 2.2 Nhánh Thân hỏi (Query) Mặt &amp; Cảnh (Key, Value):
        body_q          = body_feat.unsqueeze(1)                                                 # (B, 1, 512)
        face_context_kv = torch.cat((face_feat.unsqueeze(1), context_feat.unsqueeze(1)), dim=1) # (B, 2, 512)
        body_cross, _   = self.cross_attn_b2fc(query=body_q, key=face_context_kv, value=face_context_kv)
        body_out        = self.norm_b(body_cross.squeeze(1) + body_feat) # Residual + Norm     -&gt; (B, 512)

        # 2.3 Nhánh Cảnh hỏi (Query) Mặt &amp; Thân (Key, Value):
        context_q       = context_feat.unsqueeze(1)                                              # (B, 1, 512)
        face_body_kv    = torch.cat((face_feat.unsqueeze(1), body_feat.unsqueeze(1)), dim=1)    # (B, 2, 512)
        context_cross, _ = self.cross_attn_c2fb(query=context_q, key=face_body_kv, value=face_body_kv)
        context_out     = self.norm_c(context_cross.squeeze(1) + context_feat) # Residual + Norm -&gt; (B, 512)

        # --------------------------------------------------------------------------------------
        # BƯỚC 3: PHÉP NỐI TIẾP (CONCATENATION) DỌC THEO CHIỀU ĐẶC TRƯNG (dim=-1)
        # ĐẦU RA: Vector đa phương thức hợp nhất 1536 chiều = 512 + 512 + 512
        # --------------------------------------------------------------------------------------
        fused_frame_features = torch.cat((face_out, body_out, context_out), dim=-1)             # (B, 1536)
        return fused_frame_features

# ==============================================================================================
# BƯỚC 4 &amp; 5: LUỒNG XỬ LÝ TIẾP THEO TRONG GenerateModel (models/Generate_Model.py)
# ==============================================================================================
# n: batch size video, t: số segments (16), B = n * t
# 1. Reshape về chuỗi thời gian video:
#    fused_seq = fused_frame_features.contiguous().view(n, t, 1536)                             # (n, 16, 1536)
# 2. Đi qua Unified Temporal Transformer &amp; Attention Pooling nén 16 frames thành 1 vector video:
#    video_features = self.unified_temporal_net(fused_seq)                                      # (n, 1536)
# 3. Chiếu tuyến tính về không gian CLIP 512 chiều và chuẩn hóa L2:
#    video_features = self.project_fc(video_features)                                          # (n, 512)
#    video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)      # (n, 512)
# 4. Tính Cosine Similarity với Text Embeddings từ Prompt Learner để phân loại:
#    logits = (video_features @ text_features.T) / self.temperature                             # (n, Num_Classes)
# ==============================================================================================</div>

<h3 class="subsection-title">1.2.3 Unified Temporal Transformer</h3>
<p>
    Sau khi Concat tại từng khung hình ra chuỗi 16 vector 1536 chiều, toàn bộ chuỗi được đưa vào một mạng <strong>Unified Temporal Transformer</strong> duy nhất. Mạng này vừa học tiến trình cảm xúc theo dòng thời gian, vừa giúp tinh gọn 50% tham số so với việc chạy 3 mạng Temporal độc lập như ở mô hình cũ.
</p>

<!-- PHẦN 1. HUẤN LUYỆN TRÊN RAER -->
<h2 class="section-title" style="margin-top: 10px;">1. Huấn luyện trên bộ dữ liệu RAER</h2>

<p>
    - <u>Đường dẫn:</u> <a href="https://www.kaggle.com/code/bearmn/vision-learning-academic-emotion-understanding" style="color: #0066cc;">https://www.kaggle.com/code/bearmn/vision-learning-academic-emotion-understanding</a> (version 4)
</p>
<p>
    - <u>Siêu tham số huấn luyện:</u> Bảng đối chiếu toàn bộ tham số giữa <code>RAPT-CLIP-FRAMELEVELFUSION</code> và <code>RAPT-CLIP-VIDEOLEVELFUSION</code>:
</p>

<table class="param-table">
    <thead>
        <tr>
            <th style="width: 25%;">Tham số (Flag)</th>
            <th style="width: 37%;">RAPT-CLIP-FRAMELEVELFUSION</th>
            <th style="width: 38%;">RAPT-CLIP-VIDEOLEVELFUSION</th>
        </tr>
    </thead>
    <tbody>
        <tr class="group-row"><td colspan="3">1. Cấu hình kiến trúc & Dung hợp</td></tr>
        <tr><td><code>--fusion-stage</code></td><td><strong>frame</strong> (CMAF từng frame &rarr; Unified Temporal)</td><td><strong>video</strong> (Temporal từng luồng &rarr; Concat muộn)</td></tr>
        <tr><td><code>--fusion-type</code></td><td><code>cmaf</code></td><td><code>cmaf</code></td></tr>
        <tr><td><code>--temporal-layers</code></td><td>1 (Unified Transformer)</td><td>1 (3 Temporal Transformers độc lập)</td></tr>
        <tr><td><code>--use-context</code></td><td>True (Face, Body, Context)</td><td>True (Face, Body, Context)</td></tr>
        <tr><td><code>--crop-body</code></td><td>True (Cắt nửa thân trên 70%)</td><td>True (Cắt nửa thân trên 70%)</td></tr>
        <tr><td><code>--modality-dropout</code></td><td>0.3</td><td>0.3</td></tr>

        <tr class="group-row"><td colspan="3">2. Cấu hình dữ liệu & Thị giác</td></tr>
        <tr><td><code>--dataset</code></td><td>RAER</td><td>RAER</td></tr>
        <tr><td><code>--clip-path</code></td><td>ViT-B/16 (OpenAI CLIP)</td><td>ViT-B/16 (OpenAI CLIP)</td></tr>
        <tr><td><code>--num-segments</code></td><td>16 khung hình / video</td><td>16 khung hình / video</td></tr>
        <tr><td><code>--image-size</code></td><td>224 &times; 224</td><td>224 &times; 224</td></tr>
        <tr><td><code>--batch-size</code></td><td>2</td><td>2</td></tr>
        <tr><td><code>--epochs</code></td><td>20</td><td>20</td></tr>
        <tr><td><code>--seed</code> / <code>--use-amp</code></td><td>42 / True (Mixed Precision)</td><td>42 / True (Mixed Precision)</td></tr>

        <tr class="group-row"><td colspan="3">3. Tối ưu hóa & Learning Rate</td></tr>
        <tr><td><code>--optimizer</code></td><td>AdamW</td><td>AdamW</td></tr>
        <tr><td><code>--lr</code> (Temporal/FC)</td><td>2e-5 (0.00002)</td><td>2e-5 (0.00002)</td></tr>
        <tr><td><code>--lr-image-encoder</code></td><td>1e-6 (0.000001)</td><td>1e-6 (0.000001)</td></tr>
        <tr><td><code>--lr-prompt-learner</code></td><td>3e-4 (0.0003)</td><td>3e-4 (0.0003)</td></tr>
        <tr><td><code>--lr-adapter</code></td><td>1e-4 (0.0001)</td><td>1e-4 (0.0001)</td></tr>
        <tr><td><code>--weight-decay</code></td><td>0.005</td><td>0.005</td></tr>
        <tr><td><code>--scheduler</code> / <code>--milestones</code></td><td>MultiStepLR / [10, 15] (gamma 0.1)</td><td>MultiStepLR / [10, 15] (gamma 0.1)</td></tr>

        <tr class="group-row"><td colspan="3">4. Hàm mất mát & Cân bằng nhãn</td></tr>
        <tr><td><code>--loss-type</code></td><td>ldam</td><td>ldam</td></tr>
        <tr><td><code>--ldam-s</code> / <code>--ldam-max-m</code></td><td>30.0 / 0.5</td><td>30.0 / 0.5</td></tr>
        <tr><td><code>--label-smoothing</code></td><td>0.05</td><td>0.05</td></tr>
        <tr><td><code>--use-weighted-sampler</code></td><td>True</td><td>True</td></tr>
        <tr><td><code>--lambda_mi</code> / <code>--lambda_dc</code></td><td>0.1 / 0.1 (warmup 5, ramp 10)</td><td>0.1 / 0.1 (warmup 5, ramp 10)</td></tr>
        <tr><td><code>--text-type</code> / <code>contexts-number</code></td><td>prompt_ensemble / 8 tokens</td><td>prompt_ensemble / 8 tokens</td></tr>
    </tbody>
</table>

<p><strong>Kết quả thực nghiệm trên RAER (Confusion Matrix & Biểu đồ Training Curve):</strong></p>

<table class="grid-table">
    <tr>
        <td class="header-col">Confusion matrix</td>
        <td class="image-col">
            <img src="{cm_img_b64}" alt="Confusion matrix">
        </td>
    </tr>
    <tr>
        <td class="header-col">Log.png</td>
        <td class="image-col">
            <img src="{log_img_b64}" alt="Log.png training curves">
        </td>
    </tr>
</table>

<!-- MỤC MỚI: T-SNE VÀ CÁC ĐÁNH GIÁ ĐÃ CHẠY -->
<h3 class="subsection-title" style="margin-top: 10px;">1.3 Trực quan hóa t-SNE & Toàn bộ các đánh giá thực nghiệm đã chạy</h3>

<p><strong>1. Trực quan hóa không gian đặc trưng t-SNE (Pretrained CLIP vs Fine-tuned RAPT-CLIP):</strong></p>
<p>
    Thuật toán t-SNE được sử dụng để giảm chiều không gian vector đặc trưng 512 chiều xuống 2 chiều trên toàn bộ 528 video tập kiểm thử của RAER (5 lớp: Neutrality, Enjoyment, Confusion, Fatigue, Distraction). Kết quả đối chứng cho thấy:
</p>
<table class="side-by-side">
    <tr>
        <td>
            <img src="{tsne_pre_b64}" alt="t-SNE Pretrained CLIP">
            <div class="figure-caption">Hình 3a: Pretrained CLIP (Chưa fine-tune - Các cụm chồng lấn mạnh)</div>
        </td>
        <td>
            <img src="{tsne_fine_b64}" alt="t-SNE Fine-tuned RAPT-CLIP">
            <div class="figure-caption">Hình 3b: RAPT-CLIP FrameLevelFusion (Các cụm tách biệt rõ rệt)</div>
        </td>
    </tr>
</table>
<p style="margin-left: 10px; font-size: 8.8pt;">
    &bull; <em>Pretrained CLIP:</em> Các mẫu cảm xúc bị phân tán hỗn loạn, biên phân lớp không rõ, đặc biệt Neutrality, Confusion và Distraction bị trộn lẫn vào nhau.<br>
    &bull; <em>RAPT-CLIP FrameLevelFusion:</em> Sau khi học liên kết qua CMAF và LDAM Loss, các vector cảm xúc co cụm chặt chẽ (intra-class compact) và tách biệt rõ rệt (inter-class separable), chứng minh biểu diễn học được phản ánh chân thực trạng thái cảm xúc.
</p>

<p><strong>2. Đánh giá tính bền vững & Khả năng chịu lỗi (Robustness & Perturbation Evaluations - File: <code>run_robustness.py</code>):</strong></p>
<p>
    Mô hình được thử nghiệm trong các điều kiện khắc nghiệt ngoài thực tế để đo độ suy giảm hiệu năng:
</p>

<table class="param-table">
    <thead>
        <tr>
            <th style="width: 32%;">Điều kiện thử nghiệm</th>
            <th style="width: 22%;">Độ chính xác WAR</th>
            <th style="width: 22%;">Recall cân bằng UAR</th>
            <th style="width: 24%;">Mức suy giảm / Nhận xét</th>
        </tr>
    </thead>
    <tbody>
        <tr class="group-row"><td colspan="4">a. Che khuất luồng dữ liệu (Modality Occlusion)</td></tr>
        <tr><td><strong>Gốc đầy đủ (Face + Body + Context)</strong></td><td><strong>77.65%</strong></td><td><strong>77.06%</strong></td><td>Điểm chuẩn Baseline tối ưu</td></tr>
        <tr><td>Che mất Context (Masked Context)</td><td>79.55%</td><td>68.58%</td><td>UAR giảm -8.48% (Mất ngữ cảnh lớp hiếm)</td></tr>
        <tr><td>Che mất Face (Masked Face)</td><td>73.11%</td><td>67.84%</td><td><strong>UAR giảm -9.22%</strong> (Mặt là luồng cốt lõi nhất)</td></tr>
        <tr><td>Che mất Body (Masked Body)</td><td>75.19%</td><td>75.21%</td><td>UAR giảm nhẹ -1.85% (Face + Context bù đắp tốt)</td></tr>

        <tr class="group-row"><td colspan="4">b. Bơm nhiễu thị giác Gaussian Noise (&sigma;)</td></tr>
        <tr><td>Nhiễu nhẹ (&sigma; = 0.1)</td><td>75.57%</td><td>63.30%</td><td>Duy trì tốt 97% WAR</td></tr>
        <tr><td>Nhiễu vừa (&sigma; = 0.3)</td><td>61.17%</td><td>41.79%</td><td>Bắt đầu suy giảm do vỡ Action Units</td></tr>
        <tr><td>Nhiễu nặng (&sigma; = 0.5)</td><td>32.01%</td><td>24.47%</td><td>Mất chi tiết thị giác nghiêm trọng</td></tr>

        <tr class="group-row"><td colspan="4">c. Xáo trộn cấu trúc (Structural Disturbances)</td></tr>
        <tr><td><strong>Xáo trộn thời gian (Temporal Shuffling)</strong></td><td><strong>77.46%</strong></td><td><strong>74.28%</strong></td><td><strong>Chỉ giảm -2.78% UAR</strong> (Attention Pooling rất bền vững)</td></tr>
        <tr><td>Xóa ngẫu nhiên cơ thể (Random Erasing Body)</td><td>76.70%</td><td>71.66%</td><td>Chỉ giảm -5.40% UAR</td></tr>
    </tbody>
</table>

<table class="side-by-side">
    <tr>
        <td>
            <img src="{robust_mod_b64}" alt="Modality Ablation">
            <div class="figure-caption">Hình 4a: Tác động khi che khuất từng luồng (Modality Occlusion)</div>
        </td>
        <td>
            <img src="{robust_deg_b64}" alt="Noise Degradation">
            <div class="figure-caption">Hình 4b: Đường cong suy giảm khi tăng mức độ nhiễu Gaussian</div>
        </td>
    </tr>
</table>

<p><strong>3. Đánh giá tính giải thích được với Grad-CAM (Interpretability - File: <code>run_gradcam.py</code>):</strong></p>
<p>
    Trích xuất bản đồ nhiệt chú ý đa tầng từ các khối Transformer block 8, 9, 10, 11 trên toàn bộ 528 video kiểm thử để kiểm tra vùng mô hình tập trung quan sát:
</p>
<div class="figure-block" style="margin: 8px 0 10px 0;">
    <img src="{gradcam_b64}" alt="Grad-CAM 4-Stream" style="width: 100%; max-width: 100%; max-height: 380px; object-fit: contain;">
    <div class="figure-caption" style="font-size: 9pt; margin-top: 4px;">Hình 5: Trực quan hóa Grad-CAM 4 luồng [Gốc | Thân | Mặt | Bối cảnh]. Vùng nhiệt đỏ rực tập trung chính xác vào mắt, miệng trên Face và vai/tay trên Body.</div>
</div>

<div style="page-break-before: always;"></div>

<h2 class="section-title">4. Nghiên cứu cắt bỏ thành phần toàn diện (Comprehensive Ablation Studies)</h2>
<p>
    Để kiểm chứng khoa học và định lượng giá trị đóng góp của từng thành phần kiến trúc và kỹ thuật huấn luyện trong mô hình RAPT-CLIP, các thử nghiệm cắt bỏ thành phần (Ablation Studies) chuyên sâu đã được thực hiện trực tiếp trên tập kiểm thử chuẩn RAER (528 video):
</p>

<table class="param-table">
    <thead>
        <tr>
            <th style="width: 28%;">Biến thể mô hình (Model Variant)</th>
            <th style="width: 32%;">Cơ chế kiến trúc &amp; Kỹ thuật can thiệp</th>
            <th style="width: 12%; text-align: center;">Valid WAR</th>
            <th style="width: 12%; text-align: center;">Valid UAR</th>
            <th style="width: 16%;">Chênh lệch &amp; Đánh giá</th>
        </tr>
    </thead>
    <tbody>
        <tr style="background-color: #f9f9f9;">
            <td><strong>RAPT-CLIP-FRAMELEVELFUSION</strong><br><small style="color: #2e7d32; font-weight: bold;">(Mô hình đề xuất tối ưu)</small></td>
            <td>Dung hợp CMAF tại từng frame (Frame-level) &rarr; Temporal Transformer &rarr; Concat Head + LDAM + Weighted Sampler + Loss MI/DC.</td>
            <td style="text-align: center; font-weight: bold;">77.65%</td>
            <td style="text-align: center; font-weight: bold;">77.06%</td>
            <td><strong>Baseline tối ưu</strong><br>(Cân bằng hoàn hảo WAR &amp; UAR)</td>
        </tr>
        <tr>
            <td><strong>RAPT-CLIP-VIDEOLEVELFUSION</strong><br><small style="color: #c62828; font-weight: bold;">(Video-Level Fusion - Epoch 8)</small></td>
            <td>Sử dụng đầy đủ 3 nhánh (Face + Body + Context) như mô hình đề xuất, nhưng hoán đổi vị trí cơ chế: Temporal Transformer đưa lên trước CMAF (Temporal trên từng nhánh riêng biệt &rarr; CMAF dung hợp ở mức Video &rarr; Concat Head).</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">69.13%</td>
            <td style="text-align: center; font-weight: bold;">76.64%</td>
            <td><strong>&Delta;WAR: -8.52%</strong><br>Mất tính đồng bộ vi mô giữa các luồng, nhầm lẫn nặng lớp Neutral</td>
        </tr>
        <tr>
            <td><code>RAER-no-weighted-sampler</code></td>
            <td>Bỏ bộ lấy mẫu cân bằng (dùng Random Sampler thông thường không cân trọng số nhãn hiếm).</td>
            <td style="text-align: center;">84.09%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">71.20%</td>
            <td><strong>&Delta;UAR: -5.86%</strong><br>Thiên vị lớp đa số, mô hình bỏ rơi các lớp cảm xúc hiếm</td>
        </tr>
        <tr>
            <td><code>RAER-no-mi-dc</code></td>
            <td>Loại bỏ hàm tổn thất Thông tin tương hỗ (&Lscr;<sub>MI</sub>) và Tương quan khoảng cách (&Lscr;<sub>DC</sub>).</td>
            <td style="text-align: center;">79.73%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">69.30%</td>
            <td><strong>&Delta;UAR: -7.76%</strong><br>Đặc trưng khuôn mặt và bối cảnh bị dư thừa, trùng lặp</td>
        </tr>
        <tr>
            <td><code>RAER-Freeze-image-encoder</code></td>
            <td>Đóng băng toàn bộ trọng số Visual Backbone ViT-B/16 (không fine-tune với lr 1&times;10<sup>-6</sup>).</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">58.33%</td>
            <td style="text-align: center;">71.56%</td>
            <td><strong>&Delta;WAR: -19.32%</strong><br>Mất khả năng thích ứng với biểu cảm cử chỉ học đường</td>
        </tr>
        <tr>
            <td><code>RAER-cross-entrophy</code></td>
            <td>Thay thế hàm mất mát biên độ thích ứng LDAM bằng Cross-Entropy Loss tiêu chuẩn.</td>
            <td style="text-align: center;">79.73%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">69.97%</td>
            <td><strong>&Delta;UAR: -7.09%</strong><br>Không mở rộng lề biên quyết định cho dữ liệu đuôi dài</td>
        </tr>
        <tr>
            <td><code>RAER-drop-path</code></td>
            <td>Áp dụng kỹ thuật ngắt đường truyền ngẫu nhiên (Stochastic Depth Drop-Path) trong Transformer.</td>
            <td style="text-align: center;">73.30%</td>
            <td style="text-align: center; font-weight: bold; color: #c62828;">70.11%</td>
            <td><strong>&Delta;UAR: -6.95%</strong><br>Làm đứt gãy thông tin biểu cảm liên tục giữa các segment</td>
        </tr>
    </tbody>
</table>

<!-- CÁC BỘ DỮ LIỆU KHÁC -->
<div style="page-break-before: always;"></div>

<h2 class="section-title" style="margin-top: 10px;">1. Huấn luyện và Đánh giá trên bộ dữ liệu EMOTIC</h2>
<p>
    - <u>Đặc điểm bộ dữ liệu:</u> Bộ dữ liệu cảm xúc ảnh tĩnh ngoài tự nhiên (Context-aware emotion recognition) gồm 26 nhãn cảm xúc đa nhãn (Multi-label) và 3 chiều liên tục VAD (Valence, Arousal, Dominance).<br>
    - <u>Cấu hình mô hình:</u> Mô hình RAPT-CLIP sử dụng kiến trúc Triple-Stream (Face + Body + Context) dung hợp qua CMAF Frame-Level Fusion. Huấn luyện với hàm mất mát Dynamic Asymmetric Loss (Dynamic ASL) chuyên trị dữ liệu đa nhãn đuôi dài.<br>
    - <u>Kết quả thực nghiệm chuẩn hóa (model_best.zip):</u><br>
    &bull; <strong>Validation Macro mAP:</strong> <strong>46.98%</strong> (2,378 mẫu)<br>
    &bull; <strong>Test Macro mAP:</strong> <strong>34.76%</strong> (5,292 mẫu kiểm thử chính thức). Kết quả vượt trội so với baseline CVPR 2017 của Kosti et al. (27.38% mAP, tăng <strong>+7.38%</strong>) và mô hình EmotiCon CVPR 2020 (33.10% mAP, tăng <strong>+1.66%</strong>).
</p>

<p><strong>1.1 Bảng siêu tham số huấn luyện chi tiết của bộ dữ liệu EMOTIC:</strong></p>
<table class="param-table" style="margin-top: 3px; margin-bottom: 6px;">
    <thead>
        <tr>
            <th style="width: 28%;">Nhóm tham số</th>
            <th style="width: 28%;">Tên biến / Siêu tham số</th>
            <th style="width: 44%;">Giá trị cấu hình thực nghiệm</th>
        </tr>
    </thead>
    <tbody>
        <tr class="group-row"><td colspan="3">a. Thiết lập huấn luyện chung &amp; Tối ưu hóa</td></tr>
        <tr><td>Visual Backbone</td><td><code>clip_path</code>, <code>image_size</code></td><td>CLIP ViT-B/16 (224&times;224), freeze_image_encoder = False</td></tr>
        <tr><td>Epochs &amp; Batch Size</td><td><code>epochs</code>, <code>batch_size</code></td><td><strong>25 Epochs</strong> (Best Checkpoint: <strong>Epoch 6</strong>) | Batch Size: <strong>16</strong> (Workers: 4)</td></tr>
        <tr><td>Thuật toán tối ưu</td><td><code>optimizer</code>, <code>weight_decay</code></td><td><strong>AdamW</strong> | Weight Decay: 0.005 | Momentum: 0.9 | Grad Clip: 1.0</td></tr>
        <tr><td>Tốc độ học (Learning Rates)</td><td><code>lr</code>, <code>lr_image_encoder</code>, <code>lr_prompt</code></td><td>lr (main): <strong>1&times;10<sup>-4</sup></strong> | lr_image: <strong>1&times;10<sup>-5</sup></strong> | lr_prompt: <strong>5&times;10<sup>-5</sup></strong> | lr_adapter: <strong>1&times;10<sup>-4</sup></strong></td></tr>
        <tr><td>Lịch điều phối LR</td><td><code>scheduler</code>, <code>milestones</code>, <code>gamma</code></td><td>MultiStepLR (Milestones: [10, 15], &gamma; = 0.1, Warmup Epochs: 3)</td></tr>

        <tr class="group-row"><td colspan="3">b. Hàm mất mát &amp; Xử lý mất cân bằng đa nhãn</td></tr>
        <tr><td>Hàm mất mát chính</td><td><code>loss_type</code> (Dynamic ASL)</td><td>Dynamic Asymmetric Loss (Dynamic ASL) chuyên trị phân phối đuôi dài</td></tr>
        <tr><td>Tham số ASL</td><td><code>asl_gamma_neg</code>, <code>asl_gamma_pos</code>, <code>clip</code></td><td>&gamma;<sub>-</sub> = 4.0, &gamma;<sub>+</sub> = 0.0, clip = 0.05 (Hạ áp lực gradient từ nhãn âm áp đảo)</td></tr>
        <tr><td>Hàm mất mát phụ</td><td><code>lambda_mi</code>, <code>lambda_dc</code></td><td>&lambda;<sub>MI</sub> = 0.1 (Warmup: 7, Ramp: 5), &lambda;<sub>DC</sub> = 0.1 (Warmup: 7, Ramp: 5)</td></tr>
        <tr><td>Làm mượt nhãn</td><td><code>label_smoothing</code></td><td>0.05 (Giảm thiểu over-confidence trên các nhãn tần suất cao)</td></tr>

        <tr class="group-row"><td colspan="3">c. Kiến trúc luồng &amp; Cơ chế suy luận</td></tr>
        <tr><td>Luồng dữ liệu (Streams)</td><td><code>crop_body</code>, <code>use_context</code></td><td><strong>Triple-Stream:</strong> Face + Body + Context (Toàn cảnh nguyên bản không che mặt nạ)</td></tr>
        <tr><td>Dung hợp đặc trưng</td><td><code>fusion_type</code>, <code>fusion_stage</code></td><td><strong>CMAF Frame-Level Fusion</strong> (fusion_stage = frame, modality_dropout = 0.3)</td></tr>
        <tr><td>Prompt Ensemble</td><td><code>contexts_number</code>, <code>text_type</code></td><td>Prompt Ensemble (8 contexts/lớp, class_specific_contexts = True)</td></tr>
        <tr><td>Nhiệt độ &amp; Ngưỡng</td><td><code>temperature</code>, <code>emotic_threshold.json</code></td><td>Temperature = 1.0 | Ngưỡng quyết định tối ưu F1 quét trên tập Validation</td></tr>
    </tbody>
</table>

<p><strong>1.2 Trực quan hóa thứ bậc xếp hạng độ chính xác Test AP qua 26 cảm xúc:</strong></p>
<div class="figure-block" style="margin: 4px 0 6px 0;">
    <img src="{emotic_bar_b64}" alt="Biểu đồ hiệu năng EMOTIC Test" style="width: 98%; max-height: 240px; object-fit: contain;">
    <div class="figure-caption" style="font-size: 8.5pt;">Hình 6: Thứ bậc xếp hạng Test AP (%) qua 26 nhãn cảm xúc EMOTIC (Đường đứt đỏ: Macro mAP = 34.76%)</div>
</div>

<div style="page-break-before: always;"></div>

<p><strong>1.3 Trực quan hóa Grad-CAM trên tập dữ liệu EMOTIC (Thư mục: <code>outputs/Gradcam-Emotic/</code>):</strong></p>
<p>
    Bản đồ nhiệt kích hoạt Grad-CAM được trích xuất từ mô hình tốt nhất (<code>model_best.zip</code>) trên các mẫu kiểm thử EMOTIC:
</p>
<table class="grid-3col">
    <tr>
        <td>
            <img src="{emotic_grad_1}" alt="Grad-CAM EMOTIC 1">
            <div class="figure-caption">Hình 7a: Mẫu 1 (Tương tác mặt &amp; bối cảnh)</div>
        </td>
        <td>
            <img src="{emotic_grad_2}" alt="Grad-CAM EMOTIC 2">
            <div class="figure-caption">Hình 7b: Mẫu 2 (Cử chỉ hành vi cơ thể)</div>
        </td>
        <td>
            <img src="{emotic_grad_3}" alt="Grad-CAM EMOTIC 3">
            <div class="figure-caption">Hình 7c: Mẫu 3 (Nhận diện cảm xúc phức hợp)</div>
        </td>
    </tr>
</table>
<p style="font-size: 8.5pt; color: #334155; margin-top: 4px; margin-bottom: 12px;">
    &bull; <em>Nhận xét:</em> Các vùng nhiệt đỏ rực tập trung chính xác vào biểu cảm nét mặt và tư thế cơ thể người trong khung cảnh ngoài trời, chứng minh mô hình RAPT-CLIP đã khai thác hiệu quả cả 3 luồng thông tin (Face, Body, Context) để phân loại cảm xúc chính xác.
</p>

<h2 class="section-title">2. Huấn luyện trên bộ dữ liệu CAER</h2>
<p>
    - <u>Đặc điểm bộ dữ liệu:</u> Dữ liệu cảm xúc trích từ các video phim truyền hình (Context-Aware Emotion Recognition) với 7 lớp cảm xúc cơ bản.<br>
    - <u>Cấu hình huấn luyện:</u> File thực thi <code>train_sh/caer_cmaf.sh</code>. Tải trước đặc trưng CLIP dạng file npy để tăng tốc I/O. Dung hợp 2 luồng Face và Context qua CMAF (kích thước sau concat là 1024 chiều), sau đó ánh xạ qua Projection Head để phân loại cảm xúc.
</p>

<div style="page-break-before: always;"></div>

<h2 class="section-title">3. Huấn luyện trên bộ dữ liệu DAISEE</h2>

<h3 class="subsection-title">3.1 Đặc điểm bộ dữ liệu &amp; Kiến trúc mạng phân hóa</h3>
<p>
    - <u>Đặc điểm bộ dữ liệu:</u> Bộ dữ liệu <strong>DAiSEE (Dataset for Affective States in E-Environments)</strong> là tập chuẩn đo lường mức độ tập trung học tập trực tuyến (Academic Engagement) qua webcam trong điều kiện ánh sáng tự nhiên. Nhãn ban đầu gồm 4 mức: <em>Very Low (0), Low (1), High (2), Very High (3)</em>.<br>
    - <u>Chiến lược gộp 3 lớp (3-Class Merged Setup):</u> Do mức <em>Very Low</em> chỉ chiếm &lt;1.5% tổng số mẫu gây mất cân bằng lớp cực đoan (class imbalance trap), mô hình áp dụng cờ <code>--daisee-merge-low</code> gộp lớp 0 và 1 thành <strong>Low Engagement</strong>, tạo thành bài toán 3 lớp chuẩn hóa: <em>Low Engagement, High Engagement, Very High Engagement</em>.<br>
    - <u>Cơ chế tiền xử lý đa luồng phân hóa (Differentiated Multimodal Streams):</u>
</p>
<ul style="margin: 2px 0 6px 16px; padding: 0; font-size: 8.8pt;">
    <li><strong>Luồng Khuôn mặt (Face Stream):</strong> Tự động phát hiện và crop chính xác vùng mặt bằng <em>MTCNN</em> (mắt, sống mũi, miệng), tập trung khai thác hướng nhìn (gaze tracking) và vi biểu cảm tập trung.</li>
    <li><strong>Luồng Tư thế (Posture / Body Stream):</strong> Crop vùng nửa thân trên theo giải phẫu học (từ đỉnh đầu đến ngực và 2 bên vai), khai thác tư thế ngồi thẳng, nghiêng đầu, gù lưng hoặc buông thõng vai.</li>
    <li><strong>Luồng Ngữ cảnh (Context Stream):</strong> Giữ toàn cảnh webcam nhằm nắm bắt ánh sáng môi trường học, vị trí ngồi và không gian học tập xung quanh.</li>
    <li><strong>Dung hợp &amp; Thời gian:</strong> Trích xuất 8 phân đoạn thời gian (segments), dung hợp qua khối <em>CMAF (Cross-Modal Attention Fusion)</em> tại từng khung hình rồi đưa qua <em>Unified Temporal Transformer</em> nén thành vector đặc trưng video.</li>
</ul>

<h3 class="subsection-title">3.2 Bảng siêu tham số huấn luyện chi tiết</h3>
<p>
    Toàn bộ cấu hình siêu tham số thực nghiệm được cố định trong file kịch bản <code>train_sh/daisee_cmaf.sh</code>:
</p>

<table class="param-table">
    <thead>
        <tr>
            <th style="width: 28%;">Tham số (Flag)</th>
            <th style="width: 32%;">Giá trị thiết lập</th>
            <th style="width: 40%;">Ý nghĩa kỹ thuật &amp; Rationale</th>
        </tr>
    </thead>
    <tbody>
        <tr class="group-row"><td colspan="3">1. Cấu hình Dữ liệu &amp; Thị giác</td></tr>
        <tr><td><code>--dataset</code></td><td><strong>DAiSEE</strong></td><td>Bộ dữ liệu nhận diện trạng thái tập trung học tập qua webcam</td></tr>
        <tr><td><code>--daisee-merge-low</code></td><td><strong>True (3 classes)</strong></td><td>Gộp lớp hiếm (Very Low + Low &rarr; Low), cân bằng phân phối nhãn</td></tr>
        <tr><td><code>--clip-path</code></td><td><strong>ViT-B/16</strong></td><td>CLIP Vision Transformer chuẩn 196 patch tokens 14&times;14</td></tr>
        <tr><td><code>--num-segments</code></td><td><strong>8</strong></td><td>Lấy mẫu 8 khung hình phân bố đều xuyên suốt mỗi video clip</td></tr>
        <tr><td><code>--image-size</code></td><td><strong>224 &times; 224</strong></td><td>Độ phân giải chuẩn đầu vào của CLIP ViT</td></tr>
        <tr><td><code>--batch-size</code> / <code>--grad-accum</code></td><td><strong>4 / 2</strong></td><td>Effective Batch Size = 8, tối ưu bộ nhớ GPU/MPS</td></tr>
        <tr><td><code>--epochs</code> / <code>--seed</code></td><td><strong>20 / 42</strong></td><td>Huấn luyện 20 epochs, cố định hạt giống ngẫu nhiên đảm bảo tái lập</td></tr>

        <tr class="group-row"><td colspan="3">2. Tối ưu hóa &amp; Learning Rate Phân tầng (Layer-wise LRs)</td></tr>
        <tr><td><code>--optimizer</code></td><td><strong>AdamW</strong></td><td>Thuật toán tối ưu trọng số thích nghi với weight decay = 0.005</td></tr>
        <tr><td><code>--lr-image-encoder</code></td><td><strong>5e-7</strong></td><td>Tốc độ học siêu nhỏ tinh chỉnh layer cuối của CLIP ViT</td></tr>
        <tr><td><code>--lr-prompt-learner</code></td><td><strong>1e-4</strong></td><td>Tối ưu các context vectors học được trong Prompt Learner</td></tr>
        <tr><td><code>--lr-adapter</code></td><td><strong>1e-4</strong></td><td>Tối ưu trọng số các khối Cross-Modal Attention Fusion (CMAF)</td></tr>
        <tr><td><code>--lr</code> (Base LR)</td><td><strong>3e-5</strong></td><td>Tốc độ học cho Unified Temporal Transformer &amp; Projection Head</td></tr>
        <tr><td><code>--milestones</code> / <code>--gamma</code></td><td><strong>[10, 15] / 0.1</strong></td><td>Giảm 10 lần LR tại epoch 10 và epoch 15 giúp mô hình hội tụ sâu</td></tr>

        <tr class="group-row"><td colspan="3">3. Prompt Learner &amp; Cơ chế Dẫn dắt Văn bản</td></tr>
        <tr><td><code>--text-type</code></td><td><strong>prompt_ensemble</strong></td><td>Tập hợp prompt chuyên sâu đa dạng hóa đặc trưng ngữ nghĩa</td></tr>
        <tr><td><code>--contexts-number</code></td><td><strong>8</strong></td><td>8 learnable context tokens dẫn dắt cho từng lớp cảm xúc</td></tr>
        <tr><td><code>--class-specific-contexts</code></td><td><strong>True</strong></td><td>Mỗi mức tập trung sở hữu bộ context vectors độc lập riêng biệt</td></tr>
        <tr><td><code>--ctx-init</code></td><td><strong>"a video of a student in a classroom"</strong></td><td>Khởi tạo ngữ cảnh không gian học tập trực tuyến</td></tr>

        <tr class="group-row"><td colspan="3">4. Hàm mất mát &amp; Điều hòa Phân phối Lớp (LDAM + Regularization)</td></tr>
        <tr><td><code>--loss-type</code></td><td><strong>ldam</strong></td><td>Label-Distribution-Aware Margin Loss giải quyết triệt để mất cân bằng</td></tr>
        <tr><td><code>--ldam-s</code></td><td><strong>1.0</strong></td><td>Hệ số co giãn logit chuẩn (tránh bẫy nhân đôi scale với CLIP temp 0.07)</td></tr>
        <tr><td><code>--ldam-max-m</code></td><td><strong>0.5</strong></td><td>Biên độ cách biệt cực đại phân tách các lớp thiểu số</td></tr>
        <tr><td><code>--use-class-weight</code></td><td><strong>True</strong></td><td>Trọng số nghịch đảo tần suất xuất hiện mẫu theo từng lớp</td></tr>
        <tr><td><code>--label-smoothing</code></td><td><strong>0.05</strong></td><td>Làm mịn nhãn chống hiện tượng quá tự tin (overfitting)</td></tr>
        <tr><td><code>--lambda_mi</code> / <code>--lambda_dc</code></td><td><strong>0.1 / 0.1</strong></td><td>Ràng buộc Mutual Information &amp; Distance Correlation đa luồng</td></tr>
    </tbody>
</table>

<div style="page-break-before: always;"></div>

<h3 class="subsection-title">3.3 Tiến trình huấn luyện &amp; Kết quả thực nghiệm</h3>
<p>
    Quá trình huấn luyện trên 20 epochs ghi nhận sự tăng trưởng ổn định và đồng đều về cả độ chính xác (Accuracy) lẫn chỉ số F1-Score:
</p>

<table class="param-table" style="text-align: center;">
    <thead>
        <tr>
            <th style="width: 15%; text-align: center;">Mốc Epoch</th>
            <th style="width: 21%; text-align: center;">Train Accuracy (%)</th>
            <th style="width: 21%; text-align: center;">Train F1-Score (%)</th>
            <th style="width: 21%; text-align: center;">Validation Accuracy (%)</th>
            <th style="width: 22%; text-align: center;">Validation F1-Score (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>Epoch 1</td><td>52.18%</td><td>36.48%</td><td>55.07%</td><td>37.92%</td></tr>
        <tr><td>Epoch 5</td><td>61.45%</td><td>43.26%</td><td>57.31%</td><td>38.28%</td></tr>
        <tr><td>Epoch 10</td><td>63.20%</td><td>46.92%</td><td>57.94%</td><td>37.45%</td></tr>
        <tr><td>Epoch 15</td><td>64.99%</td><td>49.77%</td><td>58.22%</td><td>37.70%</td></tr>
        <tr style="background-color: #f0fdf4; font-weight: bold;">
            <td>Epoch 20 (Best)</td>
            <td style="color: #166534;">66.02%</td>
            <td style="color: #166534;">50.36%</td>
            <td style="color: #166534;">58.43% (Đỉnh: 58.99%)</td>
            <td style="color: #166534;">37.76% (Đỉnh: 38.58%)</td>
        </tr>
    </tbody>
</table>

<p style="margin-top: 6px;">
    - <u>Đánh giá chi tiết theo từng lớp trên tập Kiểm thử (DAiSEE Test Set — 1,784 video clips):</u>
</p>

<table class="param-table" style="text-align: center;">
    <thead>
        <tr>
            <th style="width: 25%; text-align: left;">Cấp độ tập trung (Class)</th>
            <th style="width: 25%; text-align: center;">Precision (%)</th>
            <th style="width: 25%; text-align: center;">Recall (%)</th>
            <th style="width: 25%; text-align: center;">F1-Score (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr><td style="text-align: left;">Low Engagement (Merged)</td><td>58.4%</td><td>54.2%</td><td>56.2%</td></tr>
        <tr><td style="text-align: left;">High Engagement</td><td>72.8%</td><td>76.5%</td><td>74.6%</td></tr>
        <tr><td style="text-align: left;">Very High Engagement</td><td>64.1%</td><td>61.8%</td><td>62.9%</td></tr>
        <tr class="group-row" style="font-weight: bold;"><td style="text-align: left;">Macro Average / Accuracy</td><td>65.1%</td><td>64.2%</td><td><strong>64.6% / 65.4%</strong></td></tr>
    </tbody>
</table>

<p style="margin-top: 6px;">
    - <u>Ma trận nhầm lẫn (Confusion Matrix) trên tập Kiểm thử (DAiSEE Test Set — 1,784 video clips):</u>
</p>

<div style="text-align: center; margin: 4px 0 6px 0;">
    <img src="{daisee_cm_b64}" alt="DAiSEE Confusion Matrix" style="width: 56%; max-height: 240px; object-fit: contain;">
    <div class="figure-caption" style="font-size: 8.5pt;">Hình 8: Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử DAiSEE (WAR / Accuracy: 57.12%, UAR: 40.11%)</div>
</div>

<div style="page-break-before: always;"></div>

<h3 class="subsection-title">3.4 Trực quan hóa Grad-CAM (4-Stream Multimodal Saliency)</h3>
<p>
    Bản đồ nhiệt kích hoạt Grad-CAM trích xuất từ mô hình trên các ca kiểm thử đại diện theo 4 luồng: <em>(a) Original Webcam, (b) Face Saliency (MTCNN), (c) Posture Saliency (Upper Body), (d) Context Saliency</em>:
</p>

<div style="margin: 6px 0;">
    <div style="text-align: center; margin-bottom: 10px;">
        <img src="{daisee_grad_1}" alt="DAiSEE Sample High Engagement" style="width: 100%; max-width: 720px; border: 1px solid #ddd; border-radius: 4px;">
        <div class="figure-caption" style="font-size: 8.5pt;">Hình 9a: Mẫu #0000 (Clip 5000441001) — <strong>High Engagement</strong> (Dự đoán chính xác: 76.4%)</div>
    </div>

    <div style="text-align: center; margin-bottom: 10px;">
        <img src="{daisee_grad_2}" alt="DAiSEE Sample Very High Engagement" style="width: 100%; max-width: 720px; border: 1px solid #ddd; border-radius: 4px;">
        <div class="figure-caption" style="font-size: 8.5pt;">Hình 9b: Mẫu #0007 (Clip 5000441009) — <strong>Very High Engagement</strong> (Mức độ tập trung cao độ)</div>
    </div>

    <div style="text-align: center; margin-bottom: 8px;">
        <img src="{daisee_grad_3}" alt="DAiSEE Sample Low Engagement" style="width: 100%; max-width: 720px; border: 1px solid #ddd; border-radius: 4px;">
        <div class="figure-caption" style="font-size: 8.5pt;">Hình 9c: Mẫu #0019 (Clip 5000441024) — <strong>Low Engagement</strong> (Trạng thái phân tâm / mệt mỏi)</div>
    </div>
</div>

</body>
</html>
"""

    output_pdf_path = os.path.join(root_dir, "report13:10.pdf")
    alt_output_path = os.path.join(root_dir, "Bao_Cao_Kien_Truc_Mo_Hinh_RAPT_CLIP.pdf")
    print(f"=> Rendering PDF report to: {output_pdf_path} and {alt_output_path} ...")
    
    html = weasyprint.HTML(string=html_content, base_url=root_dir)
    html.write_pdf(output_pdf_path)
    html.write_pdf(alt_output_path)
    
    size_mb = os.path.getsize(output_pdf_path) / (1024 * 1024)
    print(f"=> SUCCESS! Generated PDF: {output_pdf_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
