#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_tsne_caers.py
─────────────────
Fixes the Baseline CLIP t-SNE extraction for CAER-S:
Uses a fresh, untouched OpenAI CLIP ViT-B/16 instance (pure pre-trained weights)
so that baseline features show true severe entanglement across emotion categories.
"""

import os
import json
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.clip import clip

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('=> Device:', device)

CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CLASS_COLORS = ['#E64A19', '#8E24AA', '#1E88E5', '#FB8C00', '#546E7A', '#00ACC1', '#43A047']

# 1. Sample identical 1050 samples as before
random.seed(42)
samples_by_class = {i: [] for i in range(7)}
with open('CAER-S/annotations/test.txt') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            lbl = int(parts[1])
            if 0 <= lbl < 7:
                samples_by_class[lbl].append((parts[0], lbl))

sampled = []
for c in range(7):
    pool = samples_by_class[c]
    random.shuffle(pool)
    sampled.extend(pool[:150])

print(f'=> Sampled {len(sampled)} test images across 7 classes.')

# 2. Load FRESH, UNTOUCHED OpenAI CLIP ViT-B/16 (PURE BASELINE)
print("=> Loading Pure Pre-trained CLIP ViT-B/16 (No fine-tuning)...")
raw_clip, _ = clip.load('ViT-B/16', device=device, jit=False)
raw_clip = raw_clip.float().eval()

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

def prep_raw(img_bgr):
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    return (t - mean) / std

pre_feats = []
lbls = []

print("=> Extracting true baseline CLIP features...")
with torch.no_grad():
    for rel_p, lbl in tqdm(sampled):
        p = os.path.join('CAER-S', rel_p)
        if not os.path.exists(p):
            continue
        raw = cv2.imread(p)
        if raw is None:
            continue
        t_img = prep_raw(raw).type(raw_clip.dtype)
        feat = raw_clip.encode_image(t_img)
        feat = (feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)).squeeze(0).cpu().numpy()
        pre_feats.append(feat)
        lbls.append(lbl)

pre_feats = np.array(pre_feats)
lbls = np.array(lbls)
print('=> Pure Baseline feature extraction complete! Shape:', pre_feats.shape)

# Run t-SNE for pure Baseline CLIP
print("=> Computing t-SNE for pure Baseline CLIP...")
tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000)
emb_pre = tsne.fit_transform(pre_feats)

plt.figure(figsize=(7, 6), dpi=300)
for idx, (cname, color) in enumerate(zip(CAER_CLASSES, CLASS_COLORS)):
    mask = (lbls == idx)
    plt.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=color, label=cname, alpha=0.75, s=25, edgecolors='none')
plt.title('Baseline CLIP', fontsize=14, fontweight='bold', pad=10)
plt.legend(loc='lower left', frameon=True, fontsize=8, markerscale=1.4)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
out_baseline = 'outputs/TSNE/caers_9180_panel_c_baseline.png'
plt.savefig(out_baseline, dpi=300)
plt.close()

print(f"=> Saved true entangled Baseline CLIP plot to: {out_baseline}")
