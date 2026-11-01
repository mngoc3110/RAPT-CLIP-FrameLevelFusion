#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/prepare_latex_gradcam_tiles.py
───────────────────────────────────
Prepares 9 perfectly clean, high-resolution, unpadded image tiles
specifically for LaTeX subfigure inclusion:
- Face:    [face_original.png, face_gradcam_rapt.png, face_gradcam_clipcaer.png]
- Body:    [body_original.png, body_gradcam_rapt.png, body_gradcam_clipcaer.png]
- Context: [context_original.png, context_gradcam_rapt.png, context_gradcam_clipcaer.png]

Saved to:
1. latex_figures/
2. <appDataDir>/brain/20e97482-0dd8-4960-a7a2-94057bf8ec08/latex_tiles/
"""

import os
import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_DIR = os.path.join(ROOT_DIR, 'latex_figures')
BRAIN_TILES_DIR = '/Users/macbook/.gemini/antigravity/brain/20e97482-0dd8-4960-a7a2-94057bf8ec08/latex_tiles'
BRAIN_ROOT = '/Users/macbook/.gemini/antigravity/brain/20e97482-0dd8-4960-a7a2-94057bf8ec08'

os.makedirs(LATEX_DIR, exist_ok=True)
os.makedirs(BRAIN_TILES_DIR, exist_ok=True)

# 1. Load the clean extracted images
fo = cv2.imread('/tmp/face_orig.png')
bo = cv2.imread('/tmp/body_orig.png')
co = cv2.imread('/tmp/ctx_orig.png')

f_rapt = cv2.imread('/tmp/face_cam_rapt.png')
b_rapt = cv2.imread('/tmp/body_cam_rapt.png')
c_rapt = cv2.imread('/tmp/ctx_cam_rapt.png')

f_caer = cv2.imread('/tmp/clip_caer_face_cam.png')
b_caer = cv2.imread('/tmp/clip_caer_body_cam.png')
c_caer = cv2.imread('/tmp/clip_caer_ctx_cam.png')

# 2. Rescale all tiles to exact 300x300 for crisp, uniform LaTeX subfigures
TARGET_SIZE = (300, 300)
def clean_tile(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

tiles = {
    # Face Stream
    'face_original.png': clean_tile(fo),
    'face_gradcam_rapt.png': clean_tile(f_rapt),
    'face_gradcam_clipcaer.png': clean_tile(f_caer),

    # Body Stream
    'body_original.png': clean_tile(bo),
    'body_gradcam_rapt.png': clean_tile(b_rapt),
    'body_gradcam_clipcaer.png': clean_tile(b_caer),

    # Context Stream
    'context_original.png': clean_tile(co),
    'context_gradcam_rapt.png': clean_tile(c_rapt),
    'context_gradcam_clipcaer.png': clean_tile(c_caer),
}

print("=> Exporting 9 clean LaTeX tiles...")
for name, img in tiles.items():
    p_latex = os.path.join(LATEX_DIR, name)
    p_brain = os.path.join(BRAIN_TILES_DIR, name)
    p_root = os.path.join(BRAIN_ROOT, name)
    cv2.imwrite(p_latex, img)
    cv2.imwrite(p_brain, img)
    cv2.imwrite(p_root, img) # for direct embedding in markdown
    print(f"  ✓ Saved: {name} (300x300 px)")

print(f"\nAll files saved successfully to:\n  • {LATEX_DIR}\n  • {BRAIN_TILES_DIR}")
