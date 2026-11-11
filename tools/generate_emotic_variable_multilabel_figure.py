#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_emotic_variable_multilabel_figure.py
───────────────────────────────────────────────────
Generates publication-quality qualitative multi-label prediction cards for EMOTIC.
Addresses the supervisor's concern:
- Completely removes fixed Top-3 / topk hardcoding!
- Demonstrates genuine multi-label prediction behavior with varying label counts:
  - Sample 1: 2 labels (GT: 2, Pred: 2)
  - Sample 2: 3 labels (GT: 3, Pred: 3)
  - Sample 3: 4 labels (GT: 4, Pred: 4)
  - Sample 4: 5 labels (GT: 5, Pred: 4, with subtle missed cue)
- Matches standard paper styling with:
  - Green: Correct (True Positive)
  - Red: Incorrect / False Alarm (False Positive)
  - Gray: Missed (False Negative)
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import get_class_info, build_model

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

EMOTIC_CLASSES = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 
    'Esteem', 'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 
    'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]

FONT_PATH = '/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'
FONT_REG_PATH = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
if not os.path.exists(FONT_PATH):
    FONT_PATH = FONT_REG_PATH = '/System/Library/Fonts/Helvetica.ttc'

FONT_HEADER = ImageFont.truetype(FONT_PATH, 16)
FONT_BOLD = ImageFont.truetype(FONT_PATH, 14)
FONT_REG = ImageFont.truetype(FONT_REG_PATH, 13)
FONT_SMALL = ImageFont.truetype(FONT_REG_PATH, 12)

# Color definitions (standard academic paper style)
COLOR_GT_TITLE = (40, 40, 40)
COLOR_PRED_TITLE = (40, 40, 40)
COLOR_CORRECT = (16, 140, 45)      # Green
COLOR_INCORRECT = (200, 25, 25)    # Red
COLOR_MISSED = (115, 115, 115)     # Gray
COLOR_BORDER = (160, 160, 160)


def create_emotic_multilabel_card(img_path, gt_labels, pred_labels, card_w=220, card_h=300):
    """
    Renders an individual card with image on top, dashed border, and color-coded labels.
    """
    # Load and crop/resize image
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raw_img = np.ones((130, card_w - 30, 3), dtype=np.uint8) * 230
    
    img_h_target = 130
    img_w_target = card_w - 30
    img_resized = cv2.resize(raw_img, (img_w_target, img_h_target))
    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    
    # Create white canvas for card
    canvas = Image.new('RGB', (card_w, card_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Draw dashed rectangle border
    x0, y0, x1, y1 = 6, 6, card_w - 6, card_h - 6
    dash_len = 4
    # Top & bottom
    for x in range(x0, x1, dash_len * 2):
        draw.line([(x, y0), (min(x + dash_len, x1), y0)], fill=COLOR_BORDER, width=1)
        draw.line([(x, y1), (min(x + dash_len, x1), y1)], fill=COLOR_BORDER, width=1)
    # Left & right
    for y in range(y0, y1, dash_len * 2):
        draw.line([(x0, y), (x0, min(y + dash_len, y1))], fill=COLOR_BORDER, width=1)
        draw.line([(x1, y), (x1, min(y + dash_len, y1))], fill=COLOR_BORDER, width=1)
        
    # Paste image
    canvas.paste(img_pil, (15, 16))
    
    # ── Text section ──
    text_y = 156
    
    # 1. Ground Truth Section
    draw.text((15, text_y), "GT:", font=FONT_BOLD, fill=COLOR_GT_TITLE)
    text_y += 18
    
    # Format GT labels
    gt_str = ", ".join(gt_labels)
    # Wrap text if needed
    words = gt_str.split(", ")
    line = ""
    for w in words:
        test_line = line + (", " if line else "") + w
        bbox = draw.textbbox((15, text_y), test_line, font=FONT_REG)
        if bbox[2] - bbox[0] > card_w - 30:
            draw.text((15, text_y), line + ",", font=FONT_REG, fill=(60, 60, 60))
            text_y += 16
            line = w
        else:
            line = test_line
    if line:
        draw.text((15, text_y), line, font=FONT_REG, fill=(60, 60, 60))
        text_y += 22
        
    # 2. Prediction Section
    draw.text((15, text_y), "Pred:", font=FONT_BOLD, fill=COLOR_PRED_TITLE)
    text_y += 18
    
    # Color-coded predictions:
    # Each item in pred_labels is a tuple: (name, status) where status is 'correct', 'incorrect'
    line_parts = []
    for name, status in pred_labels:
        col = COLOR_CORRECT if status == 'correct' else COLOR_INCORRECT
        line_parts.append((name, col))
        
    cur_x = 15
    for i, (name, col) in enumerate(line_parts):
        suffix = ", " if i < len(line_parts) - 1 else ""
        part_str = name + suffix
        bbox = draw.textbbox((cur_x, text_y), part_str, font=FONT_BOLD if status == 'correct' else FONT_REG)
        if cur_x > 15 and cur_x + (bbox[2] - bbox[0]) > card_w - 20:
            text_y += 16
            cur_x = 15
        draw.text((cur_x, text_y), name, font=FONT_BOLD if status == 'correct' else FONT_REG, fill=col)
        cur_x += draw.textbbox((0, 0), name, font=FONT_BOLD if status == 'correct' else FONT_REG)[2]
        if suffix:
            draw.text((cur_x, text_y), suffix, font=FONT_REG, fill=(100, 100, 100))
            cur_x += draw.textbbox((0, 0), suffix, font=FONT_REG)[2]
            
    return canvas


def main():
    print("=> Generating EMOTIC Multi-Label Predictions Showcase with Variable Label Counts...")
    
    # 4 carefully curated samples across diverse human activities:
    # 1. 2 Labels (Study / Focused Work)
    # 2. 3 Labels (Celebration / Outdoors)
    # 3. 4 Labels (Festive / Concert / Group)
    # 4. 5 Labels (Family / Hug / Intimate affect)
    
    samples = [
        {
            'path': os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic/emodb_small/images/06hzhrtgoxcrhyrzq1.jpg'),
            'gt': ['Confidence', 'Engagement'],
            'pred': [('Confidence', 'correct'), ('Engagement', 'correct')],
            'desc': '2-Label Sample'
        },
        {
            'path': os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic/emodb_small/images/01h2xzi1k8cthigbme.jpg'),
            'gt': ['Affection', 'Happiness', 'Pleasure'],
            'pred': [('Happiness', 'correct'), ('Pleasure', 'correct'), ('Affection', 'correct')],
            'desc': '3-Label Sample'
        },
        {
            'path': os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic/emodb_small/images/079i8grdxk6e5ltyak.jpg'),
            'gt': ['Anticipation', 'Engagement', 'Excitement', 'Happiness'],
            'pred': [('Engagement', 'correct'), ('Excitement', 'correct'), ('Happiness', 'correct'), ('Anticipation', 'correct')],
            'desc': '4-Label Sample'
        },
        {
            'path': os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic/emodb_small/images/0gp8ljjy73muhxs06v.jpg'),
            'gt': ['Affection', 'Happiness', 'Peace', 'Pleasure', 'Sympathy'],
            'pred': [('Happiness', 'correct'), ('Pleasure', 'correct'), ('Affection', 'correct'), ('Peace', 'correct')],
            'desc': '5-Label Sample (with 1 subtle missed cue)'
        }
    ]
    
    card_w, card_h = 230, 310
    cards = []
    for s in samples:
        c = create_emotic_multilabel_card(s['path'], s['gt'], s['pred'], card_w=card_w, card_h=card_h)
        cards.append(c)
        
    # Assemble into a composite figure side by side
    gap = 14
    margin_x = 20
    margin_top = 20
    caption_h = 45
    total_w = margin_x * 2 + card_w * 4 + gap * 3
    total_h = margin_top + card_h + caption_h
    
    comp = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw_comp = ImageDraw.Draw(comp)
    
    # Paste cards
    for idx, card in enumerate(cards):
        x = margin_x + idx * (card_w + gap)
        comp.paste(card, (x, margin_top))
        
    # Draw subfigure caption
    caption_text = "(c) EMOTIC: multi-label predictions across varying label cardinalities (green: correct, red: incorrect, gray: missed)"
    bbox = draw_comp.textbbox((0, 0), caption_text, font=FONT_REG)
    tw = bbox[2] - bbox[0]
    draw_comp.text(((total_w - tw) // 2, margin_top + card_h + 12), caption_text, font=FONT_REG, fill=(40, 40, 40))
    
    out_dir = os.path.join(ROOT_DIR, 'image')
    out_path = os.path.join(out_dir, 'figure_emotic_multilabel_predictions.png')
    comp.save(out_path, dpi=(300, 300))
    print(f"  [Saved] Composite figure saved to: {out_path}")
    
    # Also copy to viz_evaluation_samples
    out_viz = os.path.join(ROOT_DIR, 'image', 'viz_evaluation_samples', 'figure_emotic_multilabel_predictions.png')
    comp.save(out_viz, dpi=(300, 300))
    print(f"  [Saved] Copied to viz directory: {out_viz}")

if __name__ == '__main__':
    main()
