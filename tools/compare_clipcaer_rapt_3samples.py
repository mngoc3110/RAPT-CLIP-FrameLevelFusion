#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/compare_clipcaer_rapt_3samples.py
───────────────────────────────────────
Generates full 3-class comparison on RAER (Confusion, Distraction, Enjoyment):
RAPT-CLIP (Ours, ViT-B/16) vs CLIP-CAER (Baseline, ViT-B/32) using clip-caer.pth.
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class RecorderMeter:
    pass
sys.modules['__main__'].RecorderMeter = RecorderMeter

from models.clip import clip
from models.Prompt_Learner import TextEncoder

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

# 1. Load CLIP-CAER
ckpt_path = os.path.join(ROOT_DIR, 'clip-caer.pth')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['state_dict']

clip_model, _ = clip.load('ViT-B/32', device=device)
clip_model = clip_model.float()

img_sd = {k.replace('image_encoder.', ''): v for k, v in sd.items() if k.startswith('image_encoder.')}
clip_model.visual.load_state_dict(img_sd, strict=False)
clip_model.visual.eval()

text_sd = {k.replace('text_encoder.', ''): v for k, v in sd.items() if k.startswith('text_encoder.')}
text_encoder = TextEncoder(clip_model).to(device)
text_encoder.load_state_dict(text_sd, strict=False)
text_encoder.eval()

ctx = sd['prompt_learner.ctx'].to(device)
prefix = sd['prompt_learner.token_prefix'].to(device)
suffix = sd['prompt_learner.token_suffix'].to(device)
full_prompt = torch.cat([prefix, ctx, suffix], dim=1)
tokenized_prompts = torch.zeros(5, 77, dtype=torch.long, device=device)
tokenized_prompts[:, -1] = 1

with torch.no_grad():
    text_features = text_encoder(full_prompt, tokenized_prompts)
    text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

# 2. Function to compute CAM with CLIP-CAER
def compute_cam(img_bgr, target_cls):
    h_orig, w_orig = img_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std

    v = clip_model.visual
    with torch.no_grad():
        x_conv = v.conv1(t)
        x_conv = x_conv.reshape(x_conv.shape[0], x_conv.shape[1], -1).permute(0, 2, 1)
        cls_token = v.class_embedding.to(t.dtype) + torch.zeros(t.shape[0], 1, x_conv.shape[-1], dtype=t.dtype, device=device)
        x_in = torch.cat([cls_token, x_conv], dim=1) + v.positional_embedding.to(t.dtype)
        x_pre = v.ln_pre(x_in)
        x_trans = v.transformer(x_pre.permute(1, 0, 2)).permute(1, 0, 2)
        spatial = x_trans[:, 1:, :]
        spatial_proj = v.ln_post(spatial) @ v.proj
        spatial_proj = spatial_proj / (spatial_proj.norm(dim=-1, keepdim=True) + 1e-6)

        sim = spatial_proj[0] @ text_features.t()
        target_sim = sim[:, target_cls]
        mean_sim = sim.mean(dim=1)
        disc_sim = target_sim - mean_sim

        cam_grid = disc_sim.cpu().numpy().reshape(7, 7)
        cam_grid = np.maximum(cam_grid, 0)
        if cam_grid.max() > 1e-6:
            cam_grid = cam_grid / cam_grid.max()

        cam_up = cv2.resize(cam_grid, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-6)
        heat = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(img_bgr, 0.52, heat, 0.48, 0)
        return blend

# 3. Load 3 samples from image/gradcam_sample.png
src = cv2.imread('image/gradcam_sample.png')
# Target classes: Confusion=2, Distraction=4, Enjoyment=1
sample_configs = [
    ('Confusion', 2, (98, 332), (510, 740)),
    ('Distraction', 4, (355, 588), (510, 740)),
    ('Enjoyment', 1, (598, 832), (510, 740))
]

print("=> Generating full comparison for 3 classes...")
# Build 3 sample rows
rows = []
for name, cls_idx, (y1, y2), (fx1, fx2) in sample_configs:
    body_scene = src[y1:y2, 18:250].copy()
    face_raw   = src[y1:y2-25, fx1:fx2].copy()
    face_orig  = cv2.resize(face_raw, (body_scene.shape[1], body_scene.shape[0]))
    
    # RAPT-CLIP CAMs from paper
    rapt_body_cam = src[y1:y2, 270:500].copy()
    rapt_face_cam = src[y1:y2, 770:1000].copy()
    
    # Compute CLIP-CAER CAMs
    caer_face_cam = compute_cam(face_orig, cls_idx)
    caer_body_cam = compute_cam(body_scene, cls_idx)
    
    rows.append({
        'name': name,
        'face_orig': face_orig,
        'body_orig': body_scene,
        'rapt_face': rapt_face_cam,
        'rapt_body': rapt_body_cam,
        'caer_face': caer_face_cam,
        'caer_body': caer_body_cam
    })

print(f"Computed CAMs for {len(rows)} emotion classes successfully.")
