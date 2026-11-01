#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'gradcam')
OUT_CAERS = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Caers')
os.makedirs(OUT_DIR, exist_ok=True)

FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = FONT_REG = '/System/Library/Fonts/Helvetica.ttc'


def build_unified_card(orig_tiles, cam_tiles, col_headers, title_sample, true_text, pred_text, card_w=880, card_h=620):
    """
    Builds a pristine publication-quality card:
    orig_tiles: list of 3 images [Face, Body, Context]
    cam_tiles: list of 3 images [Face, Body, Context]
    """
    canvas = np.ones((card_h, card_w, 3), dtype=np.uint8) * 255
    tile_s = 240
    left_m = 95
    gap = 12
    cols_x = [left_m + i * (tile_s + gap) for i in range(3)]
    y_row1 = 60
    y_row2 = 60 + tile_s + gap

    for x, tile in zip(cols_x, orig_tiles):
        canvas[y_row1:y_row1 + tile_s, x:x + tile_s] = cv2.resize(tile, (tile_s, tile_s), interpolation=cv2.INTER_LANCZOS4)

    for x, tile in zip(cols_x, cam_tiles):
        canvas[y_row2:y_row2 + tile_s, x:x + tile_s] = cv2.resize(tile, (tile_s, tile_s), interpolation=cv2.INTER_LANCZOS4)

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_col = ImageFont.truetype(FONT_BOLD, 17)
    font_side = ImageFont.truetype(FONT_REG, 15)
    font_title = ImageFont.truetype(FONT_BOLD, 18)
    font_meta = ImageFont.truetype(FONT_REG, 15)

    for header, cx in zip(col_headers, cols_x):
        bbox = font_col.getbbox(header)
        tw = bbox[2] - bbox[0]
        draw.text((cx + (tile_s - tw) // 2, 22), header, font=font_col, fill=(0, 0, 0))

    orig_bbox = font_side.getbbox('Original')
    draw.text(((left_m - (orig_bbox[2] - orig_bbox[0])) // 2, y_row1 + tile_s // 2 - 8),
              'Original', font=font_side, fill=(0, 0, 0))
    grad_bbox = font_side.getbbox('GradCAM')
    draw.text(((left_m - (grad_bbox[2] - grad_bbox[0])) // 2, y_row2 + tile_s // 2 - 8),
              'GradCAM', font=font_side, fill=(0, 0, 0))

    draw.text((left_m, y_row2 + tile_s + 12), title_sample, font=font_title, fill=(160, 0, 0))
    draw.text((left_m, y_row2 + tile_s + 38), f"True: {true_text}", font=font_meta, fill=(0, 110, 0))
    draw.text((left_m, y_row2 + tile_s + 62), f"Pred: {pred_text}", font=font_meta, fill=(0, 0, 160))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def build_figure_2columns_canonical():
    """
    Builds the figure with:
    - Left Column: RAER dataset with the 3 canonical classroom samples:
      [Original | Body Stream | Face Stream | Context Stream]
    - Right Column: CAER-S dataset (91.80%) with 2 video-frame samples:
      Fear and Happy (or 3 samples)
    """
    print("=> Building Figure 5: 2 Columns (RAER Column vs CAER-S Column - Canonical)...")
    raer_src = cv2.imread(os.path.join(ROOT_DIR, 'image', 'gradcam_sample.png'))

    # Extract 3 canonical samples tiles (each row has: Original, Body, Face, Context)
    # y: 98..332, 355..588, 598..832
    # x for tiles: 20..250, 265..495, 510..740, 755..985
    r_conf = raer_src[98:332, 18:986]
    r_dist = raer_src[355:588, 18:986]
    r_enjo = raer_src[598:832, 18:986]

    caers_fear = cv2.imread(os.path.join(OUT_CAERS, '3_Fear.png'))
    caers_happy = cv2.imread(os.path.join(OUT_CAERS, '4_Happy.png'))

    # We want a 2-column layout:
    # Column 1 width ~ 1100 px, Column 2 width ~ 1100 px
    # Total Canvas width = 2360 px
    CANVAS_W = 2360
    COL_W = 1100
    MARGIN = 50
    GAP = 60
    col1_x = MARGIN
    col2_x = MARGIN + COL_W + GAP

    # RAER rows: 3 rows.
    # Height of each row if scaled to width 1040 px:
    # 234 * (1040 / 968) ~ 251 px
    raer_w = 1040
    raer_h = int(234 * (raer_w / 968))
    raer_imgs = [r_conf, r_dist, r_enjo]
    raer_subtitles = ["Sample 1: Confusion", "Sample 2: Distraction", "Sample 3: Enjoyment"]

    # CAER-S cards: 2 cards.
    # Scaled to width 940 px:
    # 595 * (940 / 784) ~ 713 px
    caers_w = 940
    caers_h = int(595 * (caers_w / 784))
    caers_imgs = [caers_fear, caers_happy]

    # Calculate heights
    # Header height: 110 px
    # RAER content height:
    # 3 samples * (raer_h + 30 title/gap) + headers = 3 * (251 + 35) + 60 = ~920 px
    # To balance nicely with CAER-S (2 * 713 + 30 = 1456 px):
    # Let's adjust spacing or tile sizing so both columns look balanced and elegant!
    # If CAER-S has 2 cards:
    caers_block_h = 2 * caers_h + 35  # ~1460 px
    # In RAER, if we give each sample generous spacing and individual subtitles:
    raer_gap = (caers_block_h - 3 * raer_h) // 3
    if raer_gap < 20:
        raer_gap = 25

    TOTAL_H = 110 + caers_block_h + 60

    canvas = np.ones((TOTAL_H, CANVAS_W, 3), dtype=np.uint8) * 255
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    font_main_title = ImageFont.truetype(FONT_BOLD, 28)
    font_col_title = ImageFont.truetype(FONT_BOLD, 24)
    font_tile_header = ImageFont.truetype(FONT_BOLD, 17)
    font_sample_title = ImageFont.truetype(FONT_BOLD, 17)

    # Column 1 Header: RAER
    t1 = "RAER Dataset: Classroom Emotion Recognition (Tri-Stream)"
    bbox1 = font_col_title.getbbox(t1)
    tw1 = bbox1[2] - bbox1[0]
    draw.text((col1_x + (COL_W - tw1) // 2, 35), t1, font=font_col_title, fill=(0, 0, 0))

    # Column 2 Header: CAER-S
    t2 = "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)"
    bbox2 = font_col_title.getbbox(t2)
    tw2 = bbox2[2] - bbox2[0]
    draw.text((col2_x + (COL_W - tw2) // 2, 35), t2, font=font_col_title, fill=(0, 0, 0))

    # Divider line between columns
    div_x = col1_x + COL_W + GAP // 2
    draw.line([(div_x, 25), (div_x, TOTAL_H - 35)], fill=(210, 210, 210), width=2)

    # RAER Stream Column Headers: Original | Body Stream | Face Stream | Context Stream
    headers_raer = ["Original", "Body Stream", "Face Stream", "Context Stream"]
    tile_w_raer = raer_w // 4
    rx_start = col1_x + (COL_W - raer_w) // 2
    for idx, hdr in enumerate(headers_raer):
        hx = rx_start + idx * tile_w_raer
        h_bbox = font_tile_header.getbbox(hdr)
        hw = h_bbox[2] - h_bbox[0]
        draw.text((hx + (tile_w_raer - hw) // 2, 80), hdr, font=font_tile_header, fill=(30, 30, 30))

    # Place RAER 3 Samples
    curr_ry = 115
    for img, s_title in zip(raer_imgs, raer_subtitles):
        resized_r = cv2.resize(img, (raer_w, raer_h), interpolation=cv2.INTER_LANCZOS4)
        r_pil = Image.fromarray(cv2.cvtColor(resized_r, cv2.COLOR_BGR2RGB))
        pil_canvas.paste(r_pil, (rx_start, curr_ry))
        draw.rectangle([rx_start - 1, curr_ry - 1, rx_start + raer_w + 1, curr_ry + raer_h + 1],
                       outline=(200, 200, 200), width=2)
        curr_ry += raer_h + raer_gap

    # Place CAER-S 2 Cards
    cx_start = col2_x + (COL_W - caers_w) // 2
    curr_cy = 95
    for c_img in caers_imgs:
        resized_c = cv2.resize(c_img, (caers_w, caers_h), interpolation=cv2.INTER_LANCZOS4)
        c_pil = Image.fromarray(cv2.cvtColor(resized_c, cv2.COLOR_BGR2RGB))
        pil_canvas.paste(c_pil, (cx_start, curr_cy))
        draw.rectangle([cx_start - 1, curr_cy - 1, cx_start + caers_w + 1, curr_cy + caers_h + 1],
                       outline=(205, 205, 205), width=2)
        curr_cy += caers_h + 35

    out_path = os.path.join(OUT_DIR, 'figure5_raer_caers_columns_canonical.png')
    res = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)
    print(f"   [Saved Canonical] {out_path} ({res.shape[1]}x{res.shape[0]})")
    return out_path


def build_figure_2columns_standardized_cards():
    """
    Builds the figure with 2 completely standardized columns:
    Both RAER and CAER-S use the exact same 2x3 cards!
    """
    print("=> Building Figure 5: 2 Columns (Standardized 2x3 Cards across both)...")
    raer_conf = cv2.imread(os.path.join(OUT_DIR, 'raer_tristream_confusion.png'))
    raer_dist = cv2.imread(os.path.join(OUT_DIR, 'raer_tristream_distraction.png'))
    raer_enjo = cv2.imread(os.path.join(OUT_DIR, 'raer_tristream_enjoyment.png'))

    caers_fear = cv2.imread(os.path.join(OUT_CAERS, '3_Fear.png'))
    caers_happy = cv2.imread(os.path.join(OUT_CAERS, '4_Happy.png'))

    # 2 samples each for perfect symmetry:
    # RAER: Confusion and Enjoyment (highest contrast)
    # CAER-S: Fear and Happy
    raer_imgs = [raer_conf, raer_enjo]
    caers_imgs = [caers_fear, caers_happy]

    CANVAS_W = 2100
    COL_W = 960
    MARGIN = 45
    GAP = 50
    col1_x = MARGIN
    col2_x = MARGIN + COL_W + GAP

    card_w = 900
    orig_h, orig_w = raer_imgs[0].shape[:2]
    card_h = int(orig_h * (card_w / orig_w))  # ~683 px

    TOTAL_H = 80 + 2 * card_h + 35 + 50

    canvas = np.ones((TOTAL_H, CANVAS_W, 3), dtype=np.uint8) * 255
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    font_col_title = ImageFont.truetype(FONT_BOLD, 24)

    # Column 1 Header: RAER
    t1 = "RAER Dataset: Classroom Emotion Recognition (Tri-Stream)"
    bbox1 = font_col_title.getbbox(t1)
    tw1 = bbox1[2] - bbox1[0]
    draw.text((col1_x + (COL_W - tw1) // 2, 28), t1, font=font_col_title, fill=(0, 0, 0))

    # Column 2 Header: CAER-S
    t2 = "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)"
    bbox2 = font_col_title.getbbox(t2)
    tw2 = bbox2[2] - bbox2[0]
    draw.text((col2_x + (COL_W - tw2) // 2, 28), t2, font=font_col_title, fill=(0, 0, 0))

    # Divider line between columns
    div_x = col1_x + COL_W + GAP // 2
    draw.line([(div_x, 20), (div_x, TOTAL_H - 30)], fill=(210, 210, 210), width=2)

    # Place RAER Cards
    rx_start = col1_x + (COL_W - card_w) // 2
    curr_ry = 80
    for img in raer_imgs:
        resized_r = cv2.resize(img, (card_w, card_h), interpolation=cv2.INTER_LANCZOS4)
        r_pil = Image.fromarray(cv2.cvtColor(resized_r, cv2.COLOR_BGR2RGB))
        pil_canvas.paste(r_pil, (rx_start, curr_ry))
        draw.rectangle([rx_start - 1, curr_ry - 1, rx_start + card_w + 1, curr_ry + card_h + 1],
                       outline=(205, 205, 205), width=2)
        curr_ry += card_h + 35

    # Place CAER-S Cards
    cx_start = col2_x + (COL_W - card_w) // 2
    curr_cy = 80
    for img in caers_imgs:
        resized_c = cv2.resize(img, (card_w, card_h), interpolation=cv2.INTER_LANCZOS4)
        c_pil = Image.fromarray(cv2.cvtColor(resized_c, cv2.COLOR_BGR2RGB))
        pil_canvas.paste(c_pil, (cx_start, curr_cy))
        draw.rectangle([cx_start - 1, curr_cy - 1, cx_start + card_w + 1, curr_cy + card_h + 1],
                       outline=(205, 205, 205), width=2)
        curr_cy += card_h + 35

    out_path = os.path.join(OUT_DIR, 'figure5_raer_caers_columns_unified_2x3.png')
    res = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)
    print(f"   [Saved Standardized 2x3] {out_path} ({res.shape[1]}x{res.shape[0]})")
    return out_path


def main():
    p1 = build_figure_2columns_canonical()
    p2 = build_figure_2columns_standardized_cards()
    print("Done!")

if __name__ == '__main__':
    main()
