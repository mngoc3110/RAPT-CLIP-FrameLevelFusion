#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assemble_large_raer_figure5.py
──────────────────────────────
Assembles Figure 5 with:
1. ENLARGED RAER Top Section:
   - Scales up the 3 canonical RAER samples (Confusion, Distraction, Enjoyment)
   - Crops tight active bounding boxes to eliminate empty borders
   - Enlarges the tiles significantly so that expressions, body posture, and context are prominent
2. RE-SELECTED EMOTIC Cards:
   - Uses verified single-subject cards with dramatic emotional contrast:
     - Card 1: emotic_sample_excitement.png (Excitement, 96%)
     - Card 2: emotic_sample_sadness.png (Sadness, 95%)
3. CAER-S 91.80% Cards:
   - Card 1: 3_Fear.png (Fear, 89.4%)
   - Card 2: 4_Happy.png (Happy, 92.4%)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_GRADCAM = os.path.join(ROOT_DIR, 'outputs', 'gradcam')
OUT_CAERS = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Caers')
OUT_EMOTIC = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Emotic')
os.makedirs(OUT_GRADCAM, exist_ok=True)

FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = FONT_REG = '/System/Library/Fonts/Helvetica.ttc'


def assemble_figure5_large():
    print("=> Assembling Figure 5 with Enlarged RAER and Re-selected EMOTIC...")

    # 1. Load Canonical RAER Samples from image/gradcam_sample.png
    raer_src = cv2.imread(os.path.join(ROOT_DIR, 'image', 'gradcam_sample.png'))
    if raer_src is None:
        raise FileNotFoundError("Missing image/gradcam_sample.png")

    # Crop each canonical row tightly to its active content (remove outer white margins)
    # Confusion: y: 20..295, x: 20..999
    # Distraction: y: 297..575, x: 20..999
    # Enjoyment: y: 595..875, x: 20..999
    r_conf = raer_src[20:295, 20:1000]
    r_dist = raer_src[296:575, 20:1000]
    r_enjo = raer_src[594:875, 20:1000]
    raer_samples = [r_conf, r_dist, r_enjo]

    # 2. Load NEW, verified EMOTIC cards (Excitement & Sadness)
    emotic_paths = [
        os.path.join(OUT_EMOTIC, 'emotic_sample_excitement.png'),
        os.path.join(OUT_EMOTIC, 'emotic_sample_sadness.png')
    ]
    emotic_imgs = [cv2.imread(p) for p in emotic_paths]

    # 3. Load CAER-S 91.80% cards
    caers_paths = [
        os.path.join(OUT_CAERS, '3_Fear.png'),
        os.path.join(OUT_CAERS, '4_Happy.png')
    ]
    caers_imgs = [cv2.imread(p) for p in caers_paths]

    # Canvas Configuration
    CANVAS_W = 2500
    MARGIN = 30
    USABLE_W = CANVAS_W - 2 * MARGIN

    font_dataset_title = ImageFont.truetype(FONT_BOLD, 28)
    font_sub_label = ImageFont.truetype(FONT_BOLD, 16)

    # -------------------------------------------------------------
    # PART A: Enlarged Top RAER Horizontal Block
    # -------------------------------------------------------------
    raer_gap = 20
    sample_w = (USABLE_W - 2 * raer_gap) // 3  # ~800 px per sample! (4 tiles * 200px each)
    r_h, r_w = raer_samples[0].shape[:2]
    sample_h = int(r_h * (sample_w / r_w))     # ~225 px

    raer_block_h = 62 + sample_h + 20          # ~307 px
    raer_canvas = np.ones((raer_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    raer_pil = Image.fromarray(raer_canvas)
    draw_raer = ImageDraw.Draw(raer_pil)

    title_text = "RAER Dataset: Classroom Emotion Recognition (Tri-Stream: Face, Body, and Context)"
    bbox = font_dataset_title.getbbox(title_text)
    draw_raer.text(((CANVAS_W - (bbox[2] - bbox[0])) // 2, 18), title_text,
                   font=font_dataset_title, fill=(0, 0, 0))

    # Paste the 3 canonical RAER samples with larger display size
    for i, sample_card in enumerate(raer_samples):
        sx = MARGIN + i * (sample_w + raer_gap)
        sy = 62
        resized_card = cv2.resize(sample_card, (sample_w, sample_h), interpolation=cv2.INTER_LANCZOS4)
        p_img = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        raer_pil.paste(p_img, (sx, sy))
        draw_raer.rectangle([sx - 1, sy - 1, sx + sample_w + 1, sy + sample_h + 1],
                            outline=(195, 195, 195), width=1)

    # -------------------------------------------------------------
    # PART B: Bottom 2 Vertical Columns (EMOTIC and CAER-S)
    # -------------------------------------------------------------
    col_gap = 40
    col_w = (USABLE_W - col_gap) // 2
    card_orig_h, card_orig_w = emotic_imgs[0].shape[:2]
    card_disp_w = col_w
    card_disp_h = int(card_orig_h * (card_disp_w / card_orig_w))

    card_v_gap = 25
    num_samples = len(emotic_imgs)
    bottom_block_h = 65 + num_samples * card_disp_h + (num_samples - 1) * card_v_gap + 25

    bottom_canvas = np.ones((bottom_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    bottom_pil = Image.fromarray(bottom_canvas)
    draw_bottom = ImageDraw.Draw(bottom_pil)

    emotic_title = "EMOTIC Dataset: In-the-Wild Contextual Emotion"
    caers_title = "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)"

    emotic_x = MARGIN
    bbox_em = font_dataset_title.getbbox(emotic_title)
    draw_bottom.text((emotic_x + (col_w - (bbox_em[2] - bbox_em[0])) // 2, 16),
                     emotic_title, font=font_dataset_title, fill=(0, 0, 0))

    caers_x = MARGIN + col_w + col_gap
    bbox_caer = font_dataset_title.getbbox(caers_title)
    draw_bottom.text((caers_x + (col_w - (bbox_caer[2] - bbox_caer[0])) // 2, 16),
                     caers_title, font=font_dataset_title, fill=(0, 0, 0))

    # Paste EMOTIC cards vertically
    for k, img in enumerate(emotic_imgs):
        resized_card = cv2.resize(img, (card_disp_w, card_disp_h), interpolation=cv2.INTER_LANCZOS4)
        c_pil = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        cy = 62 + k * (card_disp_h + card_v_gap)
        bottom_pil.paste(c_pil, (emotic_x, cy))
        draw_bottom.rectangle([emotic_x - 1, cy - 1, emotic_x + card_disp_w + 1, cy + card_disp_h + 1],
                              outline=(210, 210, 210), width=1)

    # Paste CAER-S cards vertically
    for k, img in enumerate(caers_imgs):
        resized_card = cv2.resize(img, (card_disp_w, card_disp_h), interpolation=cv2.INTER_LANCZOS4)
        c_pil = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        cy = 62 + k * (card_disp_h + card_v_gap)
        bottom_pil.paste(c_pil, (caers_x, cy))
        draw_bottom.rectangle([caers_x - 1, cy - 1, caers_x + card_disp_w + 1, cy + card_disp_h + 1],
                              outline=(210, 210, 210), width=1)

    # Vertical divider line
    div_x = MARGIN + col_w + col_gap // 2
    draw_bottom.line([(div_x, 15), (div_x, bottom_block_h - 15)], fill=(200, 200, 200), width=2)

    # -------------------------------------------------------------
    # PART C: Combine Top and Bottom into Final Figure 5
    # -------------------------------------------------------------
    divider_h = 28
    TOTAL_H = raer_block_h + divider_h + bottom_block_h
    final_canvas = np.ones((TOTAL_H, CANVAS_W, 3), dtype=np.uint8) * 255
    final_pil = Image.fromarray(final_canvas)
    draw_final = ImageDraw.Draw(final_pil)

    # Paste RAER block
    final_pil.paste(raer_pil, (0, 0))

    # Horizontal divider line
    div_y = raer_block_h + divider_h // 2
    draw_final.line([(MARGIN, div_y), (CANVAS_W - MARGIN, div_y)], fill=(190, 190, 190), width=2)

    # Paste Bottom block
    final_pil.paste(bottom_pil, (0, raer_block_h + divider_h))

    out_path = os.path.join(OUT_GRADCAM, "figure5_gradcam_3datasets.png")
    res = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)
    print(f"🎉 Saved Figure 5 to: {out_path} ({res.shape[1]}x{res.shape[0]})")

    # Save preview thumbnail
    preview_p = os.path.join(OUT_GRADCAM, "figure5_gradcam_preview.png")
    preview_img = cv2.resize(res, (1100, int(res.shape[0] * (1100 / res.shape[1]))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(preview_p, preview_img)

    brain_dir = '/Users/macbook/.gemini/antigravity-ide/brain/58cc08cc-748f-41a5-bb2e-3b1cf4d4782d'
    cv2.imwrite(os.path.join(brain_dir, "figure5_gradcam_3datasets.png"), res)
    cv2.imwrite(os.path.join(brain_dir, "figure5_gradcam_preview.png"), preview_img)
    print(f"📸 Saved preview thumbnail to: {preview_p}")

    return out_path


if __name__ == '__main__':
    assemble_figure5_large()
