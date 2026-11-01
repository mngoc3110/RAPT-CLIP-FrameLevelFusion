#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_prompt_summary_pdf.py
───────────────────────────────
Tạo tài liệu PDF tổng hợp 100% CHÍNH XÁC NGUYÊN VĂN từ file models/Text.py:
"Tổng Hợp & Phân Tích Prompt Engineering Thực Tế Trong Kiến Trúc RAPT-CLIP"
Áp dụng cho 4 bộ dữ liệu: RAER, CAER-S, DAiSEE, EMOTIC.

Tài liệu làm rõ 2 vai trò kiến trúc then chốt:
1. Phân loại chính (Prompt Learner / Main Classifier): Sử dụng prompt_ensemble
2. Định vị ngữ nghĩa tĩnh cho hàm mất mát (Mutual Information Loss - MI Loss): Sử dụng class_descriptor
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Import trực tiếp nguyên văn từ models/Text.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import models.Text as T

# Đăng ký font tiếng Việt chuẩn từ hệ thống Mac
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
    """Header và Footer tự động đánh số trang chuyên nghiệp."""
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
        
        # Header (từ trang 2 trở đi)
        if self._pageNumber > 1:
            self.drawString(36, 758, "RAPT-CLIP | Báo Cáo Kỹ Thuật: Tổng Hợp Prompt Chuẩn Xác Từ models/Text.py")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)

        # Footer
        page_text = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(576, 22, page_text)
        self.drawString(36, 22, "RAPT-CLIP Research Group • Tài Liệu Kỹ Thuật Text Prompts (Nguyên Văn Text.py)")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)
        self.restoreState()


def create_pdf(output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Style định dạng văn bản
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Arial-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Arial',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=7
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Arial-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"), # Navy
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Arial-Bold',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#0F766E"), # Teal
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Arial',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        fontName='Arial',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0F172A")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        fontName='Arial-Bold',
        fontSize=7.8,
        leading=10,
        textColor=colors.white,
        alignment=1 # Center
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        fontName='Arial',
        fontSize=7.3,
        leading=10,
        textColor=colors.HexColor("#1E293B")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Arial-Bold',
        fontSize=7.3,
        leading=10,
        textColor=colors.HexColor("#0F172A")
    )

    table_cell_mono = ParagraphStyle(
        'TableCellMono',
        fontName='Arial-Italic',
        fontSize=7.2,
        leading=9.8,
        textColor=colors.HexColor("#0369A1")
    )

    box_text_style = ParagraphStyle(
        'BoxText',
        fontName='Arial',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    story = []

    # ═════════════════════════════════════════════════════════════════
    # TRANG 1: VAI TRÒ KIẾN TRÚC & BỘ DỮ LIỆU RAER
    # ═════════════════════════════════════════════════════════════════
    story.append(Paragraph("BÁO CÁO PHÂN TÍCH & TỔNG HỢP PROMPT ENGINEERING THỰC TẾ", title_style))
    story.append(Paragraph(
        "<b>Tập tin gốc:</b> <code>models/Text.py</code> &nbsp;|&nbsp; "
        "<b>Áp dụng trong:</b> <code>models/Generate_Model.py</code>, <code>utils/builders.py</code>, <code>train_sh/*.sh</code><br/>"
        "<b>Bộ dữ liệu hỗ trợ:</b> RAER (5 lớp), CAER-S (7 lớp), DAiSEE (4/3 lớp), EMOTIC (26 lớp)",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceBefore=0, spaceAfter=6))

    # MỤC 1: CÁC VỊ TRÍ KIẾN TRÚC SỬ DỤNG PROMPT
    story.append(Paragraph("1. Kiến Trúc RAPT-CLIP Lấy Câu Prompt Nào Trong models/Text.py?", h1_style))
    story.append(Paragraph(
        "Trong mã nguồn RAPT-CLIP, các câu văn trong <code>models/Text.py</code> được triệu gọi ở <b>hai vị trí kiến trúc khác biệt hoàn toàn</b>:",
        body_style
    ))

    arch_box_data = [[
        Paragraph(
            "• <b>Vị trí 1 — Phân loại chính & Prompt Tuning (PromptLearner):</b><br/>"
            "Được nạp qua tham số <code>--text-type prompt_ensemble</code> (trong toàn bộ file <code>train_sh/*.sh</code> và <code>utils/builders.py</code>). "
            "Mô hình nạp biến <b><code>prompt_ensemble_*</code></b> (mỗi lớp chứa 2–8 câu đa dạng góc nhìn) để PromptLearner học các context token mềm.<br/>"
            "• <b>Vị trí 2 — Ràng buộc ngữ nghĩa tĩnh cho hàm mất mát (Mutual Information Loss - MI Loss):</b><br/>"
            "Được hardcode trực tiếp trong <code>models/Generate_Model.py</code> (dòng 71–89) làm <code>hand_crafted_prompts</code>. "
            "Mô hình nạp biến <b><code>class_descriptor_*</code></b> (hoặc <code>class_descriptor_5_only_face</code>) để cố định ngữ nghĩa sinh trắc học chuẩn, "
            "ngăn chặn hiện tượng trôi dạt ngữ nghĩa (semantic drift) khi học prompt.",
            box_text_style
        )
    ]]
    arch_table = Table(arch_box_data, colWidths=[540])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#93C5FD")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 6))

    # MỤC 2: BỘ DỮ LIỆU RAER (5 LỚP)
    story.append(Paragraph("2. Bộ Prompt Chuẩn Xác Cho Dữ Liệu RAER (5 Lớp Học Đường)", h1_style))
    story.append(Paragraph(
        "Đối với RAER, kiến trúc sử dụng <b><code>prompt_ensemble_5</code></b> làm đầu vào phân loại chính, và "
        "<b><code>class_descriptor_5_only_face</code></b> để tính MI Loss cho luồng Face:",
        body_style
    ))

    raer_classes = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
    raer_table_data = [
        [
            Paragraph("Lớp", table_header_style),
            Paragraph("Prompt Ensemble (Dùng Phân Loại Chính / PromptLearner)<br/><i>Biến: prompt_ensemble_5 (3 câu/lớp)</i>", table_header_style),
            Paragraph("Cố Định Cho MI Loss (Luồng Mặt)<br/><i>Biến: class_descriptor_5_only_face</i>", table_header_style)
        ]
    ]

    for i, cls_name in enumerate(raer_classes):
        # 3 câu ensemble
        ens_text = "<br/>".join([f"• \"{p}\"" for p in T.prompt_ensemble_5[i]])
        # 1 câu mi loss
        mi_text = f"\"{T.class_descriptor_5_only_face[i]}\""
        raer_table_data.append([
            Paragraph(f"<b>{cls_name}</b>", table_cell_bold),
            Paragraph(ens_text, table_cell_mono),
            Paragraph(mi_text, table_cell_style)
        ])

    raer_table = Table(raer_table_data, colWidths=[75, 275, 190])
    raer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(raer_table)
    story.append(Spacer(1, 4))

    # Các biến bổ trợ của RAER
    story.append(Paragraph("<b>Các biến bổ trợ khác của RAER trong Text.py:</b>", h2_style))
    story.append(Paragraph(
        "• <b><code>class_descriptor_5_only_body</code> (Đặc trưng cơ thể):</b> "
        f"Neutral: \"{T.class_descriptor_5_only_body[0]}\" | Confusion: \"{T.class_descriptor_5_only_body[2]}\"<br/>"
        "• <b><code>class_descriptor_5_au</code> (Dành cho cờ <code>--text-type au_guided_prompts</code>):</b> "
        f"Neutral: \"{T.class_descriptor_5_au[0]}\" | Confusion: \"{T.class_descriptor_5_au[2]}\"",
        body_style
    ))

    # ═════════════════════════════════════════════════════════════════
    # TRANG 2: BỘ DỮ LIỆU CAER-S (7 LỚP)
    # ═════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("3. Bộ Prompt Chuẩn Xác Cho Dữ Liệu CAER / CAER-S (7 Lớp)", h1_style))
    story.append(Paragraph(
        "Trong các script của CAER-S (ví dụ <code>train_sh/caer_s_cmaf.sh</code>):<br/>"
        "• <b>Hàm mất mát MI Loss:</b> Lấy trực tiếp <b><code>class_descriptor_caer</code></b> (7 câu mô tả biểu cảm).<br/>"
        "• <b>Phân loại chính / PromptLearner:</b> Lấy <b><code>prompt_ensemble_caer</code></b> (mỗi lớp gồm đúng <b>8 câu mô tả cực kỳ phong phú</b> "
        "kết hợp nét mặt, tư thế cơ thể và ngữ cảnh xung quanh):",
        body_style
    ))

    caer_classes = T.class_names_caer # ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    caer_table_data = [
        [
            Paragraph("Cảm xúc", table_header_style),
            Paragraph("Mô Tả Cố Định Cho MI Loss (Biến: class_descriptor_caer)", table_header_style),
            Paragraph("Trích Đoạn Các Câu Trong prompt_ensemble_caer (8 câu/lớp)", table_header_style)
        ]
    ]

    for i, cls_name in enumerate(caer_classes):
        desc_text = f"\"{T.class_descriptor_caer[i]}\""
        # Lấy 3/8 câu tiêu biểu minh họa
        ens_sample = "<br/>".join([
            f"1. \"{T.prompt_ensemble_caer[i][0]}\"",
            f"2. \"{T.prompt_ensemble_caer[i][1]}\"",
            f"3. \"{T.prompt_ensemble_caer[i][2]}\""
        ]) + f"<br/><i>(+ 5 câu chi tiết bối cảnh & cử chỉ khác trong Text.py)</i>"

        caer_table_data.append([
            Paragraph(f"<b>{cls_name}</b>", table_cell_bold),
            Paragraph(desc_text, table_cell_style),
            Paragraph(ens_sample, table_cell_mono)
        ])

    caer_table = Table(caer_table_data, colWidths=[65, 205, 270])
    caer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#7C3AED")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(caer_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Nhận xét kỹ thuật:</b> <code>prompt_ensemble_caer</code> được thiết kế 8 câu riêng biệt cho mỗi lớp, tích hợp rõ nét: "
        "Câu 1-2 mô tả cận cảnh khuôn mặt (Face); Câu 3 mô tả tư thế cơ thể (Body); Câu 4-8 mở rộng sang bối cảnh tương tác xã hội (Context).",
        body_style
    ))

    # ═════════════════════════════════════════════════════════════════
    # TRANG 3: BỘ DỮ LIỆU DAISEE (4 LỚP & 3 LỚP MERGED)
    # ═════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Bộ Prompt Chuẩn Xác Cho Dữ Liệu DAiSEE (Mức Độ Gắn Kết Học Tập)", h1_style))
    story.append(Paragraph(
        "Bộ dữ liệu DAiSEE (ghi hình webcam trực tuyến) được thiết kế xoay quanh ánh mắt (Gaze), mí mắt và độ tương tác với màn hình. "
        "Mã nguồn hỗ trợ cả cấu hình 4 lớp gốc và cấu hình 3 lớp gộp (<code>--daisee-merge-low</code>):",
        body_style
    ))

    story.append(Paragraph("A. Cấu Hình DAiSEE 4 Lớp Gốc (Very Low, Low, High, Very High)", h2_style))
    daisee_classes_4 = T.class_names_daisee
    daisee_4_data = [
        [
            Paragraph("Mức độ", table_header_style),
            Paragraph("Cố Định Cho MI Loss (class_descriptor_daisee)", table_header_style),
            Paragraph("Prompt Ensemble Cho PromptLearner (prompt_ensemble_daisee)", table_header_style)
        ]
    ]
    for i, cls_name in enumerate(daisee_classes_4):
        mi_txt = f"\"{T.class_descriptor_daisee[i]}\""
        ens_txt = "<br/>".join([f"• \"{p}\"" for p in T.prompt_ensemble_daisee[i]])
        daisee_4_data.append([
            Paragraph(f"<b>{cls_name}</b>", table_cell_bold),
            Paragraph(mi_txt, table_cell_style),
            Paragraph(ens_txt, table_cell_mono)
        ])

    daisee_4_table = Table(daisee_4_data, colWidths=[95, 235, 210])
    daisee_4_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EA580C")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(daisee_4_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("B. Cấu Hình DAiSEE 3 Lớp Gộp (Gộp Very Low + Low thành Low Engagement)", h2_style))
    story.append(Paragraph(
        "Nhằm giải quyết mất cân bằng lớp cực đoan (<1% mẫu Very Low), dự án triển khai bộ prompt 3 lớp gộp:",
        body_style
    ))

    daisee_classes_3 = T.class_names_daisee_3class
    daisee_3_data = [
        [
            Paragraph("Mức độ (3 Lớp)", table_header_style),
            Paragraph("Cố Định Cho MI Loss (class_descriptor_daisee_3class)", table_header_style),
            Paragraph("Prompt Ensemble (prompt_ensemble_daisee_3class)", table_header_style)
        ]
    ]
    for i, cls_name in enumerate(daisee_classes_3):
        mi_txt = f"\"{T.class_descriptor_daisee_3class[i]}\""
        ens_txt = "<br/>".join([f"• \"{p}\"" for p in T.prompt_ensemble_daisee_3class[i]])
        daisee_3_data.append([
            Paragraph(f"<b>{cls_name}</b>", table_cell_bold),
            Paragraph(mi_txt, table_cell_style),
            Paragraph(ens_txt, table_cell_mono)
        ])

    daisee_3_table = Table(daisee_3_data, colWidths=[110, 225, 205])
    daisee_3_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#C2410C")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(daisee_3_table)

    # ═════════════════════════════════════════════════════════════════
    # TRANG 4: BỘ DỮ LIỆU EMOTIC (26 LỚP) - PHẦN 1 (LỚP 1 ĐẾN 13)
    # ═════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("5. Bộ Prompt Chuẩn Xác Cho Dữ Liệu EMOTIC (26 Lớp Cảm Xúc Ngữ Cảnh) - Phần 1", h1_style))
    story.append(Paragraph(
        "EMOTIC là bài toán đa nhãn phức tạp (multi-label) gồm 26 lớp. "
        "Mô hình sử dụng nguyên văn <b><code>class_descriptor_emotic_26</code></b> trong <code>models/Text.py</code> "
        "làm định nghĩa hành vi chuẩn cho hàm mất mát MI Loss, đồng thời mở rộng qua <b><code>prompt_ensemble_emotic_26</code></b> "
        "(gồm 8 tiền tố góc máy cho mỗi lớp):",
        body_style
    ))

    emotic_classes = T.class_names_emotic
    emotic_data_p1 = [
        [
            Paragraph("STT", table_header_style),
            Paragraph("Tên Lớp (Class)", table_header_style),
            Paragraph("Định Nghĩa Hành Vi Cố Định Cho MI Loss (Nguyên văn: class_descriptor_emotic_26)", table_header_style)
        ]
    ]

    for i in range(13):
        cls_name = emotic_classes[i]
        desc_text = f"\"{T.class_descriptor_emotic_26[i]}\""
        emotic_data_p1.append([
            Paragraph(str(i+1), table_cell_bold),
            Paragraph(f"<b>{cls_name}</b>", table_cell_bold),
            Paragraph(desc_text, table_cell_style)
        ])

    emotic_table_p1 = Table(emotic_data_p1, colWidths=[30, 95, 415])
    emotic_table_p1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(emotic_table_p1)

    # ═════════════════════════════════════════════════════════════════
    # TRANG 5: BỘ DỮ LIỆU EMOTIC (26 LỚP) - PHẦN 2 (LỚP 14 ĐẾN 26) & KẾT LUẬN
    # ═════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("5. Bộ Prompt Cho Dữ Liệu EMOTIC (Tiếp Theo: Lớp 14 Đến 26 & Cấu Trúc Ensemble)", h1_style))

    emotic_data_p2 = [
        [
            Paragraph("STT", table_header_style),
            Paragraph("Tên Lớp (Class)", table_header_style),
            Paragraph("Định Nghĩa Hành Vi Cố Định Cho MI Loss (Nguyên văn: class_descriptor_emotic_26)", table_header_style)
        ]
    ]

    for i in range(13, 26):
        cls_name = emotic_classes[i]
        desc_text = f"\"{T.class_descriptor_emotic_26[i]}\""
        emotic_data_p2.append([
            Paragraph(str(i+1), table_cell_bold),
            Paragraph(f"<b>{cls_name}</b>", table_cell_bold),
            Paragraph(desc_text, table_cell_style)
        ])

    emotic_table_p2 = Table(emotic_data_p2, colWidths=[30, 95, 415])
    emotic_table_p2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0F766E")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(emotic_table_p2)
    story.append(Spacer(1, 6))

    # Cấu trúc prompt_ensemble_emotic_26
    story.append(Paragraph("<b>Quy luật cấu tạo 8 tiền tố trong prompt_ensemble_emotic_26:</b>", h2_style))
    story.append(Paragraph(
        "Mỗi lớp trong số 26 lớp được nhân bản thành 8 mẫu template thị giác khác nhau kết hợp định nghĩa ngữ nghĩa:<br/>"
        "1. <i>\"a photo of a person feeling [CLASS]: [định nghĩa]\"</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        "5. <i>\"a close-up of a person feeling [CLASS]: [định nghĩa]\"</i><br/>"
        "2. <i>\"a video of a person feeling [CLASS]: [định nghĩa]\"</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        "6. <i>\"a face of a person feeling [CLASS]: [định nghĩa]\"</i><br/>"
        "3. <i>\"a portrait of a person feeling [CLASS]: [định nghĩa]\"</i> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "
        "7. <i>\"a picture of a person feeling [CLASS]: [định nghĩa]\"</i><br/>"
        "4. <i>\"an image showing a person feeling [CLASS]: [định nghĩa]\"</i> "
        "8. <i>\"a shot of a person feeling [CLASS]: [định nghĩa]\"</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Hộp tóm tắt chốt lại
    summary_box_data = [[
        Paragraph(
            "<b>TỔNG KẾT QUY CHIẾU PROMPT TRONG CODEBASE:</b><br/>"
            "• <b>Không có câu prompt tự chế ngoài luồng:</b> Toàn bộ câu văn trong tài liệu này được import trực tiếp từ mã nguồn <code>models/Text.py</code>.<br/>"
            "• <b>Tách bạch rõ 2 nhiệm vụ:</b> Nhóm <code>prompt_ensemble_*</code> dùng cho PromptLearner học biểu diễn động; "
            "Nhóm <code>class_descriptor_*</code> dùng làm neo ngữ nghĩa tĩnh cho hàm mất mát MI Loss.",
            box_text_style
        )
    ]]
    summary_table = Table(summary_box_data, colWidths=[540])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor("#86EFAC")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)

    # Xuất tài liệu PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"===> Đã tạo thành công tài liệu PDF chuẩn xác: {output_path}")


if __name__ == '__main__':
    out_pdf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'LLM_Prompt_Engineering_Summary.pdf')
    create_pdf(out_pdf)
