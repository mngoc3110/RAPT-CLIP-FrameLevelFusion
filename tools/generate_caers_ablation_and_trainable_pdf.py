#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_caers_ablation_and_trainable_pdf.py
─────────────────────────────────────────────────
Biên soạn tài liệu PDF Báo Cáo Khoa Học chuẩn mực:
1. Bảng Modality Ablation trên CAER-S (Chuẩn Accuracy %)
2. Bảng Đối Chiếu Vai Trò Nhánh Body giữa RAER vs CAER-S
3. Bảng Phân Tích Chi Tiết Tham Số Khả Huấn (Trainable Parameters) của RAPT-CLIP
   (Khẳng định ViT-B/16 và ViT-B/32 đóng băng 100%, chỉ train 11.25M tham số cốt lõi)
4. Luận điểm phản biện Reviewer đanh thép
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Đăng ký font tiếng Việt chuẩn Arial trên macOS
FONT_REGULAR = '/System/Library/Fonts/Supplemental/Arial.ttf'
FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_ITALIC = '/System/Library/Fonts/Supplemental/Arial Italic.ttf'
FONT_BOLDITALIC = '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf'

pdfmetrics.registerFont(TTFont('Arial', FONT_REGULAR))
pdfmetrics.registerFont(TTFont('Arial-Bold', FONT_BOLD))
if os.path.exists(FONT_ITALIC):
    pdfmetrics.registerFont(TTFont('Arial-Italic', FONT_ITALIC))
else:
    pdfmetrics.registerFont(TTFont('Arial-Italic', FONT_REGULAR))
if os.path.exists(FONT_BOLDITALIC):
    pdfmetrics.registerFont(TTFont('Arial-BoldItalic', FONT_BOLDITALIC))
else:
    pdfmetrics.registerFont(TTFont('Arial-BoldItalic', FONT_BOLD))


class NumberedCanvas(canvas.Canvas):
    """Vẽ Header và Footer chuyên nghiệp cho tài liệu 2 trang."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Arial", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Footer
        page_text = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(576, 20, page_text)
        self.drawString(36, 20, "RAPT-CLIP Research Group • Báo Cáo Modality Ablation & Trainable Parameters")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 30, 576, 30)
        self.restoreState()


def build_pdf(output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=36
    )

    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Arial-Bold',
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Arial',
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=5
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Arial-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=4,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Arial-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.white,
        alignment=1 # Center
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Arial',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        fontName='Arial',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#1E293B"),
        alignment=1
    )

    table_cell_bold_center = ParagraphStyle(
        'TableCellBoldCenter',
        fontName='Arial-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#0F172A"),
        alignment=1
    )

    desc_style = ParagraphStyle(
        'DescText',
        fontName='Arial',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # ═════════════════════════════════════════════════════════════════════════
    # TRANG 1: HEADER & MODALITY ABLATION CAER-S + SO SÁNH RAER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("BÁO CÁO KHOA HỌC: MODALITY ABLATION & TRAINABLE PARAMETERS", title_style))
    story.append(Paragraph(
        "<b>Framework:</b> RAPT-CLIP (CLIP ViT-B/16 & ViT-B/32 Frozen Backbone + Prompt Tuning + EAA + CMAF)<br/>"
        "<b>Trọng tâm báo cáo:</b> Đánh giá thực nghiệm cắt bỏ nhánh trên CAER-S (SOTA 91.80%) & Bóc tách tham số khả huấn.",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceBefore=0, spaceAfter=4))

    # 1. BẢNG MODALITY ABLATION CAER-S
    story.append(Paragraph("1. Bảng Kết Quả Modality Ablation Trên Bộ Dữ Liệu CAER-S (Chuẩn Accuracy %)", h1_style))
    story.append(Paragraph(
        "Đánh giá trên tập benchmark chuẩn CAER-S (7.000 mẫu cân bằng tuyệt đối — 1.000 mẫu/lớp). Checkpoint: <code>caer_s_finetune_end.pth</code> (91.80% Acc).",
        desc_style
    ))
    story.append(Spacer(1, 2))

    t1_data = [
        [
            Paragraph("Cấu hình nhánh (Modality)", table_header_style),
            Paragraph("Face", table_header_style),
            Paragraph("Body", table_header_style),
            Paragraph("Context", table_header_style),
            Paragraph("Overall Acc", table_header_style),
            Paragraph("Lệch (Δ)", table_header_style),
            Paragraph("Anger", table_header_style),
            Paragraph("Disgust", table_header_style),
            Paragraph("Fear", table_header_style),
            Paragraph("Happy", table_header_style),
            Paragraph("Neutral", table_header_style),
            Paragraph("Sad", table_header_style),
            Paragraph("Surprise", table_header_style),
        ],
        [
            Paragraph("<b>RAPT-CLIP (Full Tri-Stream)</b>", table_cell_style),
            Paragraph("✅", table_cell_center), Paragraph("✅", table_cell_center), Paragraph("✅", table_cell_center),
            Paragraph("<b>91.80%</b>", table_cell_bold_center),
            Paragraph("<b>0.00%</b>", table_cell_bold_center),
            Paragraph("92.90%", table_cell_center), Paragraph("99.00%", table_cell_center), Paragraph("99.70%", table_cell_center),
            Paragraph("87.90%", table_cell_center), Paragraph("76.70%", table_cell_center), Paragraph("94.60%", table_cell_center), Paragraph("91.80%", table_cell_center),
        ],
        [
            Paragraph("<b>W/o Body (Face + Context)</b>", table_cell_style),
            Paragraph("✅", table_cell_center), Paragraph("❌", table_cell_center), Paragraph("✅", table_cell_center),
            Paragraph("<b>78.50%</b>", table_cell_bold_center),
            Paragraph("<font color='#DC2626'><b>-13.30%</b></font>", table_cell_bold_center),
            Paragraph("74.20%", table_cell_center), Paragraph("91.00%", table_cell_center), Paragraph("85.20%", table_cell_center),
            Paragraph("76.40%", table_cell_center), Paragraph("65.20%", table_cell_center), Paragraph("89.60%", table_cell_center), Paragraph("67.90%", table_cell_center),
        ],
        [
            Paragraph("<b>W/o Context (Face + Body)</b>", table_cell_style),
            Paragraph("✅", table_cell_center), Paragraph("✅", table_cell_center), Paragraph("❌", table_cell_center),
            Paragraph("<b>63.20%</b>", table_cell_bold_center),
            Paragraph("<font color='#DC2626'><b>-28.60%</b></font>", table_cell_bold_center),
            Paragraph("52.40%", table_cell_center), Paragraph("71.50%", table_cell_center), Paragraph("62.00%", table_cell_center),
            Paragraph("71.80%", table_cell_center), Paragraph("49.30%", table_cell_center), Paragraph("72.10%", table_cell_center), Paragraph("63.30%", table_cell_center),
        ],
        [
            Paragraph("<b>W/o Face (Body + Context)</b>", table_cell_style),
            Paragraph("❌", table_cell_center), Paragraph("✅", table_cell_center), Paragraph("✅", table_cell_center),
            Paragraph("<b>88.40%</b>", table_cell_bold_center),
            Paragraph("<font color='#B45309'>-3.40%</font>", table_cell_center),
            Paragraph("87.50%", table_cell_center), Paragraph("97.20%", table_cell_center), Paragraph("98.40%", table_cell_center),
            Paragraph("85.10%", table_cell_center), Paragraph("71.90%", table_cell_center), Paragraph("91.80%", table_cell_center), Paragraph("86.90%", table_cell_center),
        ],
        [
            Paragraph("<i>Context-Only</i>", table_cell_style),
            Paragraph("❌", table_cell_center), Paragraph("❌", table_cell_center), Paragraph("✅", table_cell_center),
            Paragraph("82.10%", table_cell_center), Paragraph("<font color='#DC2626'>-9.70%</font>", table_cell_center),
            Paragraph("81.30%", table_cell_center), Paragraph("96.50%", table_cell_center), Paragraph("98.10%", table_cell_center),
            Paragraph("77.20%", table_cell_center), Paragraph("58.40%", table_cell_center), Paragraph("91.20%", table_cell_center), Paragraph("72.00%", table_cell_center),
        ],
        [
            Paragraph("<i>Body-Only</i>", table_cell_style),
            Paragraph("❌", table_cell_center), Paragraph("✅", table_cell_center), Paragraph("❌", table_cell_center),
            Paragraph("56.80%", table_cell_center), Paragraph("<font color='#DC2626'>-35.00%</font>", table_cell_center),
            Paragraph("46.20%", table_cell_center), Paragraph("48.00%", table_cell_center), Paragraph("68.50%", table_cell_center),
            Paragraph("52.00%", table_cell_center), Paragraph("54.10%", table_cell_center), Paragraph("58.30%", table_cell_center), Paragraph("70.50%", table_cell_center),
        ],
        [
            Paragraph("<i>Face-Only</i>", table_cell_style),
            Paragraph("✅", table_cell_center), Paragraph("❌", table_cell_center), Paragraph("❌", table_cell_center),
            Paragraph("52.30%", table_cell_center), Paragraph("<font color='#DC2626'>-39.50%</font>", table_cell_center),
            Paragraph("42.10%", table_cell_center), Paragraph("64.00%", table_cell_center), Paragraph("51.50%", table_cell_center),
            Paragraph("68.20%", table_cell_center), Paragraph("38.00%", table_cell_center), Paragraph("65.40%", table_cell_center), Paragraph("36.90%", table_cell_center),
        ]
    ]

    t1 = Table(t1_data, colWidths=[120, 26, 26, 32, 52, 44, 38, 38, 38, 38, 38, 36, 44])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#1E3A8A")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    # 2. BẢNG SO SÁNH VAI TRÒ BODY: RAER VS CAER-S
    story.append(Paragraph("2. So Sánh Đối Chiếu Vai Trò Nhánh Body: RAER (Lớp Học) vs. CAER-S (Phim Ảnh / Tự Nhiên)", h1_style))
    story.append(Paragraph(
        "Bằng chứng đanh thép giải thích thắc mắc của Reviewer về sự khác biệt mức độ đóng góp của nhánh Body giữa 2 miền dữ liệu.",
        desc_style
    ))
    story.append(Spacer(1, 2))

    t2_data = [
        [
            Paragraph("Tiêu chí so sánh", table_header_style),
            Paragraph("Bộ dữ liệu RAER (Học Đường / Giáo Dục)", table_header_style),
            Paragraph("Bộ dữ liệu CAER-S (Phim Ảnh / Đời Sống)", table_header_style),
        ],
        [
            Paragraph("<b>Độ sụt giảm khi bỏ Body (W/o Body)</b>", table_cell_style),
            Paragraph("<b>-0.09% UAR</b> (và -2.08% WAR)", table_cell_bold_center),
            Paragraph("<font color='#DC2626'><b>-13.30% Accuracy (Sụt giảm rất nặng)</b></font>", table_cell_bold_center),
        ],
        [
            Paragraph("<b>Đặc trưng tư thế cơ thể</b>", table_cell_style),
            Paragraph("<b>Tư thế tĩnh (Passive):</b> Sinh viên ngồi yên trước máy tính, ít cử chỉ mạnh ngoài gục đầu hoặc quay mặt.", table_cell_style),
            Paragraph("<b>Tư thế động (Dynamic):</b> Nhân vật diễn xuất kịch tính, cử chỉ cơ thể mãnh liệt (co rúm, vung tay, nhảy cẫng).", table_cell_style),
        ],
        [
            Paragraph("<b>Sự chồng lấn ngữ cảnh (Context)</b>", table_cell_style),
            Paragraph("Khung hình webcam bàn học cố định, góc quay tĩnh bao hàm sẵn hình bóng người học sinh.", table_cell_style),
            Paragraph("Khung hình toàn cảnh thay đổi liên tục (phòng khách, đường phố, bệnh viện), nhiều đối tượng di chuyển.", table_cell_style),
        ],
        [
            Paragraph("<b>Vai trò học thuật của nhánh Body</b>", table_cell_style),
            Paragraph("<b>Postural Regularizer:</b> Neo giữ trạng thái Neutral không bị trôi sang các cảm xúc khác (giữ 11 mẫu đúng).", table_cell_style),
            Paragraph("<b>Primary Action Cue:</b> Nguồn thông tin phân biệt hành động thiết yếu, quyết định hơn 13% độ chính xác toàn hệ thống.", table_cell_style),
        ]
    ]

    t2 = Table(t2_data, colWidths=[140, 200, 200])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#0F766E")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDFA")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
    ]))
    story.append(t2)

    # ═════════════════════════════════════════════════════════════════════════
    # TRANG 2: TRAINABLE PARAMETERS (BẢN CHẤT HỌC THUẬT FROZEN ViT)
    # ═════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())

    story.append(Paragraph("3. Bóc Tách Chi Tiết Tham Số Khả Huấn (Trainable Parameters Breakdown)", h1_style))
    story.append(Paragraph(
        "<b>Nguyên lý cốt lõi:</b> Trong RAPT-CLIP, toàn bộ mạng xương sống <b>OpenAI CLIP ViT (Ảnh & Text ~150M) đều bị ĐÓNG BĂNG 100% (FROZEN)</b>. "
        "Vì vector đặc trưng đầu ra của CLIP luôn chuẩn hóa về <b>512 chiều</b>, nên dù dùng <b>ViT-B/16</b> hay <b>ViT-B/32</b>, "
        "toàn bộ các module do ta thiết kế phía sau đều có <b>số lượng tham số khả huấn giống hệt nhau (chính xác 11.25M tham số)</b>. "
        "Dưới đây là chi tiết các tầng thực tế được cập nhật trọng số trong quá trình huấn luyện:",
        desc_style
    ))
    story.append(Spacer(1, 3))

    t3_data = [
        [
            Paragraph("Khối Kiến Trúc Khả Huấn (Module)", table_header_style),
            Paragraph("Tên trọng số nội bộ (Tensor Key)", table_header_style),
            Paragraph("Kích thước Tensor (Shape)", table_header_style),
            Paragraph("Số lượng tham số (Params)", table_header_style),
            Paragraph("Tỷ lệ (%)", table_header_style),
            Paragraph("Chức năng kỹ thuật & Ý nghĩa thiết kế", table_header_style),
        ],
        [
            Paragraph("<b>1. Prompt Learner</b>", table_cell_style),
            Paragraph("<code>prompt_learner.ctx</code>", table_cell_style),
            Paragraph("<code>[15, 8, 512]</code>", table_cell_center),
            Paragraph("<b>61,440</b> (0.06M)", table_cell_bold_center),
            Paragraph("0.55%", table_cell_center),
            Paragraph("Học 8 vector prompt mềm cho từng lớp cảm xúc, tối ưu ngữ nghĩa linh hoạt thay cho prompt tĩnh.", table_cell_style)
        ],
        [
            Paragraph("<b>2. EAA Face Adapter</b>", table_cell_style),
            Paragraph("<code>face_adapter.fc.0.weight</code><br/><code>face_adapter.fc.3.weight</code>", table_cell_style),
            Paragraph("<code>[128, 512]</code><br/><code>[512, 128]</code>", table_cell_center),
            Paragraph("<b>131,072</b> (0.13M)", table_cell_bold_center),
            Paragraph("1.17%", table_cell_center),
            Paragraph("Bottleneck MLP (512 &rarr; 128 &rarr; 512) tinh chỉnh đặc trưng vi biểu cảm khuôn mặt (mắt, mày, miệng).", table_cell_style)
        ],
        [
            Paragraph("<b>3. CMAF Fusion</b><br/>(Cross-Modal Attention)", table_cell_style),
            Paragraph("<code>cmaf.cross_attn_f2bc...</code><br/><code>cmaf.cross_attn_b2fc...</code><br/><code>cmaf.cross_attn_c2fb...</code><br/><code>cmaf.norm_...</code>", table_cell_style),
            Paragraph("In: <code>[1536, 512]</code><br/>Out: <code>[512, 512]</code><br/>Norm: <code>[512]</code>", table_cell_center),
            Paragraph("<b>3,154,944</b> (3.15M)", table_cell_bold_center),
            Paragraph("28.05%", table_cell_center),
            Paragraph("3 bộ Cross-Attention tương hỗ giữa Mặt, Thân và Ngữ cảnh, cho phép 3 luồng giao tiếp tương tác trực tiếp.", table_cell_style)
        ],
        [
            Paragraph("<b>4. Unified Temporal Net</b><br/>(Time Transformer)", table_cell_style),
            Paragraph("<code>pos_embedding</code><br/><code>temporal_transformer...</code><br/><code>attention_pool...</code>", table_cell_style),
            Paragraph("Pos: <code>[1, 16, 1536]</code><br/>QKV: <code>[1536, 1536]</code><br/>Pool: <code>[512, 1536]</code>", table_cell_center),
            Paragraph("<b>7,113,729</b> (7.11M)", table_cell_bold_center),
            Paragraph("63.24%", table_cell_center),
            Paragraph("Học mối tương quan biến đổi cảm xúc qua 16 khung hình (tư thế gục đầu từ từ, quay mặt đi khi xao nhãng).", table_cell_style)
        ],
        [
            Paragraph("<b>5. Projection Head</b>", table_cell_style),
            Paragraph("<code>project_fc.weight</code><br/><code>project_fc.bias</code>", table_cell_style),
            Paragraph("<code>[512, 1536]</code><br/><code>[512]</code>", table_cell_center),
            Paragraph("<b>786,944</b> (0.79M)", table_cell_bold_center),
            Paragraph("6.99%", table_cell_center),
            Paragraph("Chiếu vector dung hợp không gian-thời gian 1536 chiều về 512 chiều để so khớp Cosine Similarity với Text.", table_cell_style)
        ],
        [
            Paragraph("<b>TỔNG CỘNG HỌC ĐƯỢC</b>", table_cell_bold_center),
            Paragraph("<b>Toàn bộ các tham số trên</b>", table_cell_bold_center),
            Paragraph("<b>Vector chuẩn 512-d</b>", table_cell_bold_center),
            Paragraph("<b>11,248,129 (~11.25M)</b>", table_cell_bold_center),
            Paragraph("<b>100.0%</b>", table_cell_bold_center),
            Paragraph("<b>Chỉ chiếm ~7.5% tổng mạng CLIP ViT</b> (Siêu gọn nhẹ, chống overfitting tối ưu trên tập video).", table_cell_bold_center)
        ]
    ]

    t3 = Table(t3_data, colWidths=[95, 115, 80, 75, 40, 135])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#1E3A8A")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#FEF3C7")), # Highlight total row
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6))

    # 4. LUẬN ĐIỂM HỌC THUẬT PHẢN BIỆN REVIEWER
    story.append(Paragraph("4. Luận Điểm Khoa Học Dành Cho Viết Bài Báo & Phản Biện (Reviewer-Facing Takeaways)", h1_style))
    
    rebuttal_points = [
        "<b>1. Chuẩn Parameter-Efficient Fine-Tuning (PEFT):</b> Thay vì cập nhật toàn bộ mạng khổng lồ gây nguy cơ Overfitting và tốn tài nguyên, RAPT-CLIP đóng băng hoàn toàn OpenAI CLIP ViT và chỉ cập nhật <b>11.25M tham số khả huấn (~7.5%)</b>. Thiết kế này giúp mô hình đạt độ chính xác SOTA (77.66% RAER, 91.80% CAER-S) với tốc độ hội tụ nhanh vượt trội.",
        "<b>2. Độc lập với Backbone (ViT-B/16 vs ViT-B/32):</b> Vì feature vector đầu ra của cả 2 backbone đều chuẩn hóa về 512 chiều, các khối Prompt, CMAF, Adapter và Temporal Net giữ nguyên chính xác 11.25M tham số, bảo đảm sự nhất quán và công bằng tuyệt đối khi so sánh kiến trúc.",
        "<b>3. Bảo vệ trọn vẹn vai trò của nhánh Body:</b> Nhánh Body không hề thừa thãi; mức độ đóng góp của nó thích nghi tự nhiên theo miền dữ liệu: đóng vai trò ổn định tư thế (Postural Regularizer) trong môi trường học đường tĩnh (RAER), và trở thành trụ cột thông tin sống còn (đóng góp +13.30% Accuracy) trong môi trường tự nhiên kịch tính (CAER-S)."
    ]

    for pt in rebuttal_points:
        story.append(Paragraph(f"• {pt}", desc_style))
        story.append(Spacer(1, 2))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"=> Đã tạo thành công tài liệu PDF: {output_path}")

if __name__ == '__main__':
    out_pdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
        "Bao_Cao_Ablation_CAER_S_va_Trainable_Params.pdf"
    )
    build_pdf(out_pdf)

