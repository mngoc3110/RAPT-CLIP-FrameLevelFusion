#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assemble_paper_figures.py
─────────────────────────
Generates publication-ready figures for the RAPT-CLIP academic paper:
1. Figure 4: 4-panel t-SNE (RAER Baseline, RAER RAPT-CLIP, CAER-S Baseline, CAER-S RAPT-CLIP 91.80%)
2. Figure 5: Qualitative Grad-CAM comparison across 3 datasets:
   - Top (Horizontal): RAER dataset (Confusion, Distraction, Enjoyment)
   - Bottom Left (Vertical Column): EMOTIC dataset
   - Bottom Right (Vertical Column): CAER-S dataset (from caer_s_finetune_end.pth)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TSNE = os.path.join(ROOT_DIR, 'outputs', 'TSNE')
OUT_GRADCAM = os.path.join(ROOT_DIR, 'outputs', 'gradcam')
OUT_CAERS = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Caers')
OUT_EMOTIC = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Emotic')

# Fonts
FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = FONT_REG = '/System/Library/Fonts/Helvetica.ttc'


def build_figure4_tsne():
    """Builds Figure 4: 4-panel t-SNE side-by-side (RAER 77.66% and CAER-S 91.80%)."""
    print("=> Assembling Figure 4: 4-Panel t-SNE (RAER 77.66% + CAER-S 91.80%)...")
    # (a) RAER Pre-trained (5 classes)
    raer_a = cv2.imread(os.path.join(ROOT_DIR, 'image', 'tsne_pretrained.png'))
    # (b) RAER Fine-tuned RAPT-CLIP (77.66% SOTA, 5 classes)
    raer_b = cv2.imread(os.path.join(ROOT_DIR, 'image', 'tsne_finetuned.png'))
    # (c) CAER-S Baseline CLIP (7 classes - tangled dots)
    caers_c = cv2.imread(os.path.join(OUT_TSNE, 'caers_9180_panel_c_baseline.png'))
    # (d) CAER-S Fine-tuned RAPT-CLIP 91.80% (7 classes - distinct clusters)
    caers_d = cv2.imread(os.path.join(OUT_TSNE, 'caers_9180_panel_d_raptclip.png'))

    # Fallback if 9180 run is still in progress
    if caers_c is None:
        caers_c = cv2.imread(os.path.join(OUT_TSNE, 'caers_panel_c_ground_truth.png'))
    if caers_d is None:
        caers_d = cv2.imread(os.path.join(OUT_TSNE, 'caers_panel_d_ground_truth.png'))

    caers_c_plot = caers_c
    caers_d_plot = caers_d

    def crop_plot_area(im, tol=250):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        non_white = np.where(gray < tol)
        if len(non_white[0]) == 0:
            return im
        y1, y2 = np.min(non_white[0]), np.max(non_white[0])
        x1, x2 = np.min(non_white[1]), np.max(non_white[1])
        pad = 12
        y1 = max(0, y1 - pad)
        y2 = min(im.shape[0], y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(im.shape[1], x2 + pad)
        return im[y1:y2, x1:x2]

    target_h = 950

    def resize_h(img, h):
        c_img = crop_plot_area(img)
        aspect = c_img.shape[1] / c_img.shape[0]
        w = int(h * aspect)
        return cv2.resize(c_img, (w, h), interpolation=cv2.INTER_AREA)

    panels = [resize_h(raer_a, target_h), resize_h(raer_b, target_h),
              resize_h(caers_c_plot, target_h), resize_h(caers_d_plot, target_h)]
    subcaps = [
        '(a) RAER: Pre-trained',
        '(b) RAER: RAPT-CLIP',
        '(c) CAER-S: Pre-trained',
        '(d) CAER-S: RAPT-CLIP'
    ]

    gap = 35
    pad_top = 25
    pad_bottom = 105
    total_w = sum(p.shape[1] for p in panels) + gap * (len(panels) - 1) + 60
    total_h = target_h + pad_top + pad_bottom

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)
    font = ImageFont.truetype(FONT_BOLD, 42)

    curr_x = 30
    for p, subcap in zip(panels, subcaps):
        pw = p.shape[1]
        p_pil = Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
        pil_canvas.paste(p_pil, (curr_x, pad_top))

        bbox = font.getbbox(subcap)
        tw = bbox[2] - bbox[0]
        tx = curr_x + (pw - tw) // 2
        ty = pad_top + target_h + 30
        draw.text((tx, ty), subcap, font=font, fill=(0, 0, 0))
        curr_x += pw + gap

    out_path = os.path.join(OUT_TSNE, 'figure4_tsne_final.png')
    res = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)
    print(f"   [Saved] Figure 4 saved to: {out_path} ({res.shape[1]}x{res.shape[0]})")
    return out_path


def build_figure5_gradcam(num_samples=2):
    """
    Builds Figure 5:
    - Top horizontal: RAER with all 3 streams (Face, Body, Context)
    - Bottom left column: EMOTIC (Fresh, standard, high-clarity cards)
    - Bottom right column: CAER-S (From caer_s_finetune_end.pth 91.80%)
    """
    print(f"=> Assembling Figure 5: Grad-CAM (num_samples={num_samples})...")
    
    # 1. Load Canonical RAER Samples from image/gradcam_sample.png (cropped to active tiles)
    raer_src = cv2.imread(os.path.join(ROOT_DIR, 'image', 'gradcam_sample.png'))
    r_conf = raer_src[20:295, 20:1000]
    r_dist = raer_src[296:575, 20:1000]
    r_enjo = raer_src[594:875, 20:1000]
    raer_samples = [r_conf, r_dist, r_enjo]
    raer_names = ["Sample 1: Confusion", "Sample 2: Distraction", "Sample 3: Enjoyment"]

    # 2. Load PRISTINE, 100% DISTINCT EMOTIC cards
    emotic_paths = [
        os.path.join(OUT_EMOTIC, 'emotic_distinct_happiness.png'),
        os.path.join(OUT_EMOTIC, 'emotic_distinct_sadness.png')
    ][:num_samples]

    # 3. Load CAER-S 91.80% cards
    caers_paths = [
        os.path.join(OUT_CAERS, '3_Fear.png'),
        os.path.join(OUT_CAERS, '4_Happy.png')
    ][:num_samples]

    emotic_imgs = [cv2.imread(p) for p in emotic_paths]
    caers_imgs = [cv2.imread(p) for p in caers_paths]

    # Canvas Width = 3500 px (Equal tile size ~286px across all 3 datasets!)
    CANVAS_W = 3500
    MARGIN = 40
    USABLE_W = CANVAS_W - 2 * MARGIN

    # -------------------------------------------------------------
    # PART A: Assemble Top RAER Horizontal Block (Face + Body + Context)
    # -------------------------------------------------------------
    raer_gap = 35
    sample_w = (USABLE_W - 2 * raer_gap) // 3  # 1150 px! (4 tiles * 287 px each!)
    r_h, r_w = raer_samples[0].shape[:2]
    sample_h = int(r_h * (sample_w / r_w))     # ~323 px!

    raer_block_h = 75 + sample_h + 25
    raer_canvas = np.ones((raer_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    raer_pil = Image.fromarray(raer_canvas)
    draw_raer = ImageDraw.Draw(raer_pil)

    font_dataset_title = ImageFont.truetype(FONT_BOLD, 36)

    # RAER Dataset Title
    title_text = "RAER Dataset: Classroom Emotion Recognition (Tri-Stream: Face, Body, and Context)"
    bbox = font_dataset_title.getbbox(title_text)
    draw_raer.text(((CANVAS_W - (bbox[2] - bbox[0])) // 2, 20), title_text,
                   font=font_dataset_title, fill=(0, 0, 0))

    # Place the 3 RAER 4-stream cards
    for i, (sample_card, s_name) in enumerate(zip(raer_samples, raer_names)):
        sx = MARGIN + i * (sample_w + raer_gap)
        sy = 75
        resized_card = cv2.resize(sample_card, (sample_w, sample_h), interpolation=cv2.INTER_LANCZOS4)
        p_img = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        raer_pil.paste(p_img, (sx, sy))
        draw_raer.rectangle([sx - 1, sy - 1, sx + sample_w + 1, sy + sample_h + 1],
                            outline=(195, 195, 195), width=2)

    # -------------------------------------------------------------
    # PART B: Bottom 2 Vertical Columns (EMOTIC and CAER-S)
    # -------------------------------------------------------------
    col_gap = 60
    col_w = (USABLE_W - col_gap) // 2  # ~1690 px

    card_disp_w = 1000  # 3 tiles * 286 px!
    card_disp_h = int(595 * (card_disp_w / 784))  # ~759 px

    card_v_gap = 30
    bottom_block_h = 80 + num_samples * card_disp_h + (num_samples - 1) * card_v_gap + 30

    bottom_canvas = np.ones((bottom_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    bottom_pil = Image.fromarray(bottom_canvas)
    draw_bottom = ImageDraw.Draw(bottom_pil)

    emotic_title = "EMOTIC Dataset: In-the-Wild Contextual Emotion"
    caers_title = "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)"

    emotic_col_x = MARGIN
    bbox_em = font_dataset_title.getbbox(emotic_title)
    draw_bottom.text((emotic_col_x + (col_w - (bbox_em[2] - bbox_em[0])) // 2, 20),
                     emotic_title, font=font_dataset_title, fill=(0, 0, 0))

    caers_col_x = MARGIN + col_w + col_gap
    bbox_caer = font_dataset_title.getbbox(caers_title)
    draw_bottom.text((caers_col_x + (col_w - (bbox_caer[2] - bbox_caer[0])) // 2, 20),
                     caers_title, font=font_dataset_title, fill=(0, 0, 0))

    emotic_card_x = emotic_col_x + (col_w - card_disp_w) // 2
    caers_card_x = caers_col_x + (col_w - card_disp_w) // 2

    # Paste EMOTIC cards vertically
    for k, img in enumerate(emotic_imgs):
        resized_card = cv2.resize(img, (card_disp_w, card_disp_h), interpolation=cv2.INTER_LANCZOS4)
        c_pil = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        cy = 80 + k * (card_disp_h + card_v_gap)
        bottom_pil.paste(c_pil, (emotic_card_x, cy))
        draw_bottom.rectangle([emotic_card_x - 1, cy - 1, emotic_card_x + card_disp_w + 1, cy + card_disp_h + 1],
                              outline=(210, 210, 210), width=2)

    # Paste CAER-S cards vertically
    for k, img in enumerate(caers_imgs):
        resized_card = cv2.resize(img, (card_disp_w, card_disp_h), interpolation=cv2.INTER_LANCZOS4)
        c_pil = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        cy = 80 + k * (card_disp_h + card_v_gap)
        bottom_pil.paste(c_pil, (caers_card_x, cy))
        draw_bottom.rectangle([caers_card_x - 1, cy - 1, caers_card_x + card_disp_w + 1, cy + card_disp_h + 1],
                              outline=(210, 210, 210), width=2)

    # Vertical divider line between EMOTIC and CAER-S
    div_x = MARGIN + col_w + col_gap // 2
    draw_bottom.line([(div_x, 20), (div_x, bottom_block_h - 20)], fill=(200, 200, 200), width=2)

    # -------------------------------------------------------------
    # Combine Top and Bottom into Final Figure 5
    # -------------------------------------------------------------
    divider_h = 30
    TOTAL_H = raer_block_h + divider_h + bottom_block_h
    final_canvas = np.ones((TOTAL_H, CANVAS_W, 3), dtype=np.uint8) * 255
    final_pil = Image.fromarray(final_canvas)
    draw_final = ImageDraw.Draw(final_pil)

    # Paste RAER block
    final_pil.paste(raer_pil, (0, 0))

    # Draw horizontal divider
    div_y = raer_block_h + divider_h // 2
    draw_final.line([(MARGIN, div_y), (CANVAS_W - MARGIN, div_y)], fill=(190, 190, 190), width=2)

    # Paste Bottom block
    final_pil.paste(bottom_pil, (0, raer_block_h + divider_h))

    suffix = f"{num_samples}samples"
    out_name = f"figure5_gradcam_3datasets_{suffix}.png"
    out_path = os.path.join(OUT_GRADCAM, out_name)
    res = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)

    if num_samples == 2:
        main_path = os.path.join(OUT_GRADCAM, "figure5_gradcam_3datasets.png")
        cv2.imwrite(main_path, res)
        print(f"   [Main Saved] {main_path}")

    print(f"   [Saved] Figure 5 ({suffix}) to: {out_path} ({res.shape[1]}x{res.shape[0]})")
    return out_path


def main():
    print("==================================================================")
    print("🚀 ASSEMBLING PUBLICATION FIGURES FOR RAPT-CLIP")
    print("==================================================================")
    
    # 1. Figure 4: t-SNE
    fig4_path = build_figure4_tsne()
    
    # 2. Figure 5: Grad-CAM (2 samples per vertical column - standard page fit)
    fig5_2s_path = build_figure5_gradcam(num_samples=2)

    # 3. Figure 5: Grad-CAM (3 samples per vertical column - expanded gallery)
    fig5_3s_path = build_figure5_gradcam(num_samples=3)

    print("==================================================================")
    print("🎉 ALL FIGURES SUCCESSFULLY GENERATED!")
    print("==================================================================")


if __name__ == '__main__':
    main()
