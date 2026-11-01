#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assemble_perfect_figure5.py
───────────────────────────
Builds the definitive Figure 5 meeting ALL user specifications:
1. RAER Top Section:
   - Every tile is displayed at the EXACT SAME SIZE as EMOTIC (Tile size ~ 285x285 px).
   - Shows the 3 canonical samples: Confusion, Distraction, Enjoyment.
   - All 4 tiles per sample: [Original | Body Stream | Face Stream | Context Stream].
2. EMOTIC Bottom-Left Section:
   - Uses the brand new distinct cards (emotic_distinct_happiness.png and emotic_distinct_sadness.png).
   - Body Stream and Context Stream are 100% DISTINCT (Body is localized crop, Context is full wide scene).
3. CAER-S Bottom-Right Section:
   - From caer_s_finetune_end.pth (91.80%): 3_Fear.png and 4_Happy.png.
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


def assemble_perfect_figure5():
    print("=> Assembling Perfect Figure 5 (RAER tiles equal size to EMOTIC tiles, Distinct Context)...")

    # 1. Load Canonical RAER Samples from image/gradcam_sample.png
    raer_src = cv2.imread(os.path.join(ROOT_DIR, 'image', 'gradcam_sample.png'))
    if raer_src is None:
        raise FileNotFoundError("Missing image/gradcam_sample.png")

    # Crop tightly to the 4 tiles of each canonical row
    r_conf = raer_src[20:295, 20:1000]
    r_dist = raer_src[296:575, 20:1000]
    r_enjo = raer_src[594:875, 20:1000]
    raer_samples = [r_conf, r_dist, r_enjo]

    # 2. Load NEW, 100% DISTINCT EMOTIC cards
    emotic_paths = [
        os.path.join(OUT_EMOTIC, 'emotic_distinct_happiness.png'),
        os.path.join(OUT_EMOTIC, 'emotic_distinct_sadness.png')
    ]
    emotic_imgs = [cv2.imread(p) for p in emotic_paths]

    # 3. Load CAER-S 91.80% cards
    caers_paths = [
        os.path.join(OUT_CAERS, '3_Fear.png'),
        os.path.join(OUT_CAERS, '4_Happy.png')
    ]
    caers_imgs = [cv2.imread(p) for p in caers_paths]

    # -------------------------------------------------------------
    # Geometry & Dimensions:
    # Canvas Width = 3500 px.
    # Each tile in RAER = ~285 px wide!
    # Each tile in EMOTIC / CAER-S = ~285 px wide!
    # -------------------------------------------------------------
    CANVAS_W = 3500
    MARGIN = 40
    USABLE_W = CANVAS_W - 2 * MARGIN

    font_dataset_title = ImageFont.truetype(FONT_BOLD, 36)

    # PART A: RAER Top Section
    raer_gap = 35
    sample_w = (USABLE_W - 2 * raer_gap) // 3  # 1150 px! (4 tiles * 287 px each!)
    r_h, r_w = raer_samples[0].shape[:2]
    sample_h = int(r_h * (sample_w / r_w))     # ~323 px!

    raer_block_h = 75 + sample_h + 25          # ~423 px
    raer_canvas = np.ones((raer_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    raer_pil = Image.fromarray(raer_canvas)
    draw_raer = ImageDraw.Draw(raer_pil)

    title_text = "RAER Dataset: Classroom Emotion Recognition (Tri-Stream: Face, Body, and Context)"
    bbox = font_dataset_title.getbbox(title_text)
    draw_raer.text(((CANVAS_W - (bbox[2] - bbox[0])) // 2, 20), title_text,
                   font=font_dataset_title, fill=(0, 0, 0))

    for i, sample_card in enumerate(raer_samples):
        sx = MARGIN + i * (sample_w + raer_gap)
        sy = 75
        resized_card = cv2.resize(sample_card, (sample_w, sample_h), interpolation=cv2.INTER_LANCZOS4)
        p_img = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        raer_pil.paste(p_img, (sx, sy))
        draw_raer.rectangle([sx - 1, sy - 1, sx + sample_w + 1, sy + sample_h + 1],
                            outline=(195, 195, 195), width=2)

    # PART B: Bottom 2 Columns (EMOTIC and CAER-S)
    col_gap = 60
    col_w = (USABLE_W - col_gap) // 2  # ~1690 px
    
    # We display each card so its tiles are ~285 px (matching RAER tiles!)
    # In each card (784x595), 3 tiles occupy ~692 px (224*3 + 20).
    # If card_disp_w = 1000 px: tile size is 224 * (1000/784) = 286 px!
    card_disp_w = 1000
    card_disp_h = int(595 * (card_disp_w / 784))  # ~759 px

    card_v_gap = 30
    num_samples = len(emotic_imgs)
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

    # Center the cards inside each column
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

    # Vertical divider line
    div_x = MARGIN + col_w + col_gap // 2
    draw_bottom.line([(div_x, 20), (div_x, bottom_block_h - 20)], fill=(200, 200, 200), width=2)

    # PART C: Combine Top and Bottom into Final Figure 5
    divider_h = 35
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
    print(f"🎉 Saved Perfect Figure 5 to: {out_path} ({res.shape[1]}x{res.shape[0]})")

    # Save preview thumbnail
    preview_p = os.path.join(OUT_GRADCAM, "figure5_gradcam_preview.png")
    preview_img = cv2.resize(res, (1200, int(res.shape[0] * (1200 / res.shape[1]))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(preview_p, preview_img)

    brain_dir = '/Users/macbook/.gemini/antigravity-ide/brain/58cc08cc-748f-41a5-bb2e-3b1cf4d4782d'
    cv2.imwrite(os.path.join(brain_dir, "figure5_gradcam_3datasets.png"), res)
    cv2.imwrite(os.path.join(brain_dir, "figure5_gradcam_preview.png"), preview_img)
    print(f"📸 Saved preview thumbnail to: {preview_p}")

    return out_path


if __name__ == '__main__':
    assemble_perfect_figure5()
