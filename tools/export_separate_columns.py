#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'gradcam')
os.makedirs(OUT_DIR, exist_ok=True)

from assemble_figure5_raer_caers_dual_column import (
    get_perfect_raer_samples,
    get_caers_samples,
    FONT_BOLD,
    FONT_REG
)

def build_single_dataset_figure(samples, title, out_filename, num_samples=3):
    samples = samples[:num_samples]
    tile_size = 240
    tile_gap = 12
    col_w = 4 * tile_size + 3 * tile_gap  # 996 px
    margin_x = 40

    canvas_w = 2 * margin_x + col_w
    pad_top = 35
    title_h = 42
    header_h = 32
    row_meta_h = 38
    row_gap = 26

    sample_row_h = tile_size + row_meta_h
    content_h = num_samples * sample_row_h + (num_samples - 1) * row_gap
    canvas_h = pad_top + title_h + header_h + content_h + 30

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    font_title = ImageFont.truetype(FONT_BOLD, 22)
    font_stream_header = ImageFont.truetype(FONT_BOLD, 16)
    font_meta_bold = ImageFont.truetype(FONT_BOLD, 15)

    # Title
    bbox_t = font_title.getbbox(title)
    tw = bbox_t[2] - bbox_t[0]
    draw.text(((canvas_w - tw) // 2, pad_top), title, font=font_title, fill=(0, 0, 0))

    # Stream headers
    stream_names = ['Original', 'Body Stream', 'Face Stream', 'Context Stream']
    y_headers = pad_top + title_h + 8
    for idx, sname in enumerate(stream_names):
        tx = margin_x + idx * (tile_size + tile_gap)
        bbox = font_stream_header.getbbox(sname)
        sw = bbox[2] - bbox[0]
        draw.text((tx + (tile_size - sw) // 2, y_headers), sname, font=font_stream_header, fill=(30, 30, 30))

    y_content_start = y_headers + header_h

    curr_y = y_content_start
    for s in samples:
        tiles = [s['orig'], s['body'], s['face'], s['ctx']]
        for idx, tile in enumerate(tiles):
            tx = margin_x + idx * (tile_size + tile_gap)
            resized_tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_LANCZOS4)
            tile_pil = Image.fromarray(cv2.cvtColor(resized_tile, cv2.COLOR_BGR2RGB))
            pil_canvas.paste(tile_pil, (tx, curr_y))
            draw.rectangle([tx - 1, curr_y - 1, tx + tile_size, curr_y + tile_size],
                           outline=(205, 205, 205), width=1)
        
        meta_y = curr_y + tile_size + 8
        meta_str_prefix = f"Sample: {s['title']}  |  "
        meta_true_str = f"True: {s['true']}  |  "
        meta_pred_str = f"Pred: {s['pred']}"

        full_meta = meta_str_prefix + meta_true_str + meta_pred_str
        bbox_m = font_meta_bold.getbbox(full_meta)
        mw = bbox_m[2] - bbox_m[0]
        mx = margin_x + (col_w - mw) // 2

        draw.text((mx, meta_y), meta_str_prefix, font=font_meta_bold, fill=(160, 0, 0))
        w1 = font_meta_bold.getbbox(meta_str_prefix)[2] - font_meta_bold.getbbox(meta_str_prefix)[0]
        draw.text((mx + w1, meta_y), meta_true_str, font=font_meta_bold, fill=(0, 120, 0))
        w2 = font_meta_bold.getbbox(meta_true_str)[2] - font_meta_bold.getbbox(meta_true_str)[0]
        draw.text((mx + w1 + w2, meta_y), meta_pred_str, font=font_meta_bold, fill=(0, 0, 160))

        curr_y += sample_row_h + row_gap

    out_path = os.path.join(OUT_DIR, out_filename)
    res = cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, res)
    print(f"Saved: {out_path} ({res.shape[1]}x{res.shape[0]})")
    return out_path

def main():
    raer_samples = get_perfect_raer_samples()
    caers_samples = get_caers_samples()

    # 3 samples each
    build_single_dataset_figure(raer_samples, "RAER Dataset: Classroom Emotion Recognition (Tri-Stream)", "figure_raer_gradcam.png", num_samples=3)
    build_single_dataset_figure(caers_samples, "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)", "figure_caers_gradcam.png", num_samples=3)

    # 2 samples each
    build_single_dataset_figure(raer_samples, "RAER Dataset: Classroom Emotion Recognition (Tri-Stream)", "figure_raer_gradcam_2samples.png", num_samples=2)
    build_single_dataset_figure(caers_samples, "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)", "figure_caers_gradcam_2samples.png", num_samples=2)

if __name__ == '__main__':
    main()
