#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/export_individual_gradcam_images.py
─────────────────────────────────────────
Exports individual image files for every stream and model variant:
- Face (Original, RAPT-CLIP CAM, CLIP-CAER CAM)
- Body (Original, RAPT-CLIP CAM, CLIP-CAER CAM)
- Context (Original, RAPT-CLIP CAM, CLIP-CAER CAM)
- Full RAPT-CLIP Fig 5(a) strip
- Full CLIP-CAER Fig 5(a) strip
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'individual')
BRAIN_DIR = '/Users/macbook/.gemini/antigravity/brain/20e97482-0dd8-4960-a7a2-94057bf8ec08'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BRAIN_DIR, exist_ok=True)

# 1. Load the computed images
face_orig = cv2.imread('/tmp/face_orig.png')
body_orig = cv2.imread('/tmp/body_orig.png')
ctx_orig  = cv2.imread('/tmp/ctx_orig.png')

face_cam_rapt = cv2.imread('/tmp/face_cam_rapt.png')
body_cam_rapt = cv2.imread('/tmp/body_cam_rapt.png')
ctx_cam_rapt  = cv2.imread('/tmp/ctx_cam_rapt.png')

face_cam_caer = cv2.imread('/tmp/clip_caer_face_cam.png')
body_cam_caer = cv2.imread('/tmp/clip_caer_body_cam.png')
ctx_cam_caer  = cv2.imread('/tmp/clip_caer_ctx_cam.png')

# Standard size 256x256 for clean viewing
STD_SIZE = (256, 256)
def std(img):
    return cv2.resize(img, STD_SIZE, interpolation=cv2.INTER_LANCZOS4)

images_to_save = {
    '1_face_original.png': std(face_orig),
    '2_face_gradcam_raptclip.png': std(face_cam_rapt),
    '3_face_gradcam_clipcaer.png': std(face_cam_caer),
    
    '4_body_original.png': std(body_orig),
    '5_body_gradcam_raptclip.png': std(body_cam_rapt),
    '6_body_gradcam_clipcaer.png': std(body_cam_caer),
    
    '7_context_original.png': std(ctx_orig),
    '8_context_gradcam_raptclip.png': std(ctx_cam_rapt),
    '9_context_gradcam_clipcaer.png': std(ctx_cam_caer),
}

for name, img in images_to_save.items():
    p1 = os.path.join(OUT_DIR, name)
    p2 = os.path.join(BRAIN_DIR, name)
    cv2.imwrite(p1, img)
    cv2.imwrite(p2, img)
    print(f"Saved: {name} ({img.shape[1]}x{img.shape[0]})")

# 2. Build the exact full-row strip for CLIP-CAER (identical layout to Fig 5a in user's image)
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'

font_stream = ImageFont.truetype(FONT_PATH, 24)
font_sub = ImageFont.truetype(FONT_PATH, 19)
font_caption = ImageFont.truetype(FONT_PATH, 22)

def build_paper_row(face_o, face_c, body_o, body_c, ctx_o, ctx_c, caption_text):
    W, H = 1024, 328
    tile_w, tile_h = 155, 180
    
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    
    # Coordinates matching Fig 5a
    # Streams
    # Face: x 30..340
    # Body: x 360..670
    # Context: x 690..1000
    streams = [
        ("Face stream", 30, face_o, face_c),
        ("Body stream", 360, body_o, body_c),
        ("Context stream", 690, ctx_o, ctx_c)
    ]
    
    y_stream = 25
    y_sub = 65
    y_img = 98
    
    for title, sx, im_orig, im_cam in streams:
        # Title
        bb = font_stream.getbbox(title)
        draw.text((sx + (310 - (bb[2] - bb[0])) // 2, y_stream), title, font=font_stream, fill=(0, 0, 0))
        
        # Sub
        bb_o = font_sub.getbbox("Original")
        bb_c = font_sub.getbbox("Grad-CAM")
        draw.text((sx + (tile_w - (bb_o[2] - bb_o[0])) // 2, y_sub), "Original", font=font_sub, fill=(0, 0, 0))
        draw.text((sx + tile_w + (tile_w - (bb_c[2] - bb_c[0])) // 2, y_sub), "Grad-CAM", font=font_sub, fill=(0, 0, 0))
        
        # Paste images
        p_o = Image.fromarray(cv2.cvtColor(cv2.resize(im_orig, (tile_w, tile_h)), cv2.COLOR_BGR2RGB))
        p_c = Image.fromarray(cv2.cvtColor(cv2.resize(im_cam, (tile_w, tile_h)), cv2.COLOR_BGR2RGB))
        pil.paste(p_o, (sx, y_img))
        pil.paste(p_c, (sx + tile_w, y_img))
        
    # Caption at bottom
    bb_cap = font_caption.getbbox(caption_text)
    draw.text(((W - (bb_cap[2] - bb_cap[0])) // 2, 290), caption_text, font=font_caption, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# Row for CLIP-CAER
row_clipcaer = build_paper_row(
    face_orig, face_cam_caer,
    body_orig, body_cam_caer,
    ctx_orig, ctx_cam_caer,
    "(a) RAER — CLIP-CAER (Baseline, ViT-B/32)"
)

# Row for RAPT-CLIP
row_raptclip = build_paper_row(
    face_orig, face_cam_rapt,
    body_orig, body_cam_rapt,
    ctx_orig, ctx_cam_rapt,
    "(a) RAER — RAPT-CLIP (Ours, ViT-B/16)"
)

cv2.imwrite(os.path.join(OUT_DIR, 'clip_caer_fig5a_row.png'), row_clipcaer)
cv2.imwrite(os.path.join(BRAIN_DIR, 'clip_caer_fig5a_row.png'), row_clipcaer)

cv2.imwrite(os.path.join(OUT_DIR, 'rapt_clip_fig5a_row.png'), row_raptclip)
cv2.imwrite(os.path.join(BRAIN_DIR, 'rapt_clip_fig5a_row.png'), row_raptclip)

print("Exported all individual images and row strips successfully!")
