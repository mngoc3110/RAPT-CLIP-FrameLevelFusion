#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TILES = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'latex_tiles')
os.makedirs(OUT_TILES, exist_ok=True)

from assemble_figure5_raer_caers_dual_column import (
    get_perfect_raer_samples,
    get_caers_samples
)

raer_samples = get_perfect_raer_samples()
caers_samples = get_caers_samples()

# 1. RAER tiles
# Confusion
cv2.imwrite(os.path.join(OUT_TILES, 'confusion_org.png'), raer_samples[0]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'confusion_body.png'), raer_samples[0]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'confusion_face.png'), raer_samples[0]['face'])

# Distraction
cv2.imwrite(os.path.join(OUT_TILES, 'distraction_org.png'), raer_samples[1]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'distraction_body.png'), raer_samples[1]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'distraction_face.png'), raer_samples[1]['face'])

# Enjoyment
cv2.imwrite(os.path.join(OUT_TILES, 'enjoyment_org.png'), raer_samples[2]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'enjoyment_body.png'), raer_samples[2]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'enjoyment_face.png'), raer_samples[2]['face'])

# 2. CAER-S tiles
# Fear
cv2.imwrite(os.path.join(OUT_TILES, 'Fear1_org.png'), caers_samples[0]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'Fear1_body.png'), caers_samples[0]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'Fear1_face.png'), caers_samples[0]['face'])

# Happy
cv2.imwrite(os.path.join(OUT_TILES, 'Happy1_org.png'), caers_samples[1]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'Happy1_body.png'), caers_samples[1]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'Happy1_face.png'), caers_samples[1]['face'])

# Neutral / Sad
cv2.imwrite(os.path.join(OUT_TILES, 'Neutral1_org.png'), caers_samples[2]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'Neutral1_body.png'), caers_samples[2]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'Neutral1_face.png'), caers_samples[2]['face'])

# Also save with Sad name
cv2.imwrite(os.path.join(OUT_TILES, 'Sad1_org.png'), caers_samples[2]['orig'])
cv2.imwrite(os.path.join(OUT_TILES, 'Sad1_body.png'), caers_samples[2]['body'])
cv2.imwrite(os.path.join(OUT_TILES, 'Sad1_face.png'), caers_samples[2]['face'])

print(f"Successfully exported all 18+ individual tile images into: {OUT_TILES}")
