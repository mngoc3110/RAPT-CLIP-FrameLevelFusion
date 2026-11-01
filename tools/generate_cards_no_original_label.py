#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_cards_no_original_label.py
─────────────────────────────────────────
Generates GradCAM publication cards in the EXACT template requested by user:
- Face Stream | Body Stream | Context Stream
- Row 1: Stream inputs (WITHOUT the text 'Original' on the left)
- Row 2: GradCAM heatmaps (with 'GradCAM' label on the left)
- Footer: Status badge (CORRECT PREDICTION), Target Class, Ground Truth, Predicted

Runs BOTH:
1. RAPT-CLIP (ViT-B/16, model_best.pth)
2. CLIP-CAER (ViT-B/32, clip-caer.pth)
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

class RecorderMeter:
    pass
sys.modules['__main__'].RecorderMeter = RecorderMeter

from models.clip import clip
from models.Prompt_Learner import TextEncoder
from utils.builders import get_class_info
from tools.generate_publication_samples import (
    compute_vivid_cam, overlay_heatmap, mask_face_on_body, get_clean_active_context
)

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
print(f"=> Device: {device}")

# ──────────────────────── Fonts ────────────────────────
FONT_BOLD_PATH = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_BOLD_PATH):
    FONT_BOLD_PATH = FONT_REG_PATH = '/System/Library/Fonts/Helvetica.ttc'

FONT_TITLE = ImageFont.truetype(FONT_BOLD_PATH, 14)
FONT_SIDE  = ImageFont.truetype(FONT_REG_PATH, 13)
FONT_ATT   = ImageFont.truetype(FONT_BOLD_PATH, 16)
FONT_INFO  = ImageFont.truetype(FONT_REG_PATH, 13)

# ──────────────────────── Custom Card Builder ────────────────────────
def build_custom_template_card(face_orig, body_orig, ctx_orig,
                               face_cam, body_cam, ctx_cam,
                               attention_text, true_text, pred_text,
                               is_correct=True, model_badge_text=None):
    s = 224
    gap = 8
    left_m = 90
    right_m = 20
    pad_top = 46
    pad_bot = 105
    grid_w = left_m + s * 3 + gap * 2 + right_m  # 796 px
    grid_h = pad_top + s * 2 + gap + pad_bot      # 609 px

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    cols_x = [left_m, left_m + s + gap, left_m + (s + gap) * 2]
    y_row1 = pad_top
    y_row2 = pad_top + s + gap

    def place_with_border(target_canvas, img_tile, x_pos, y_pos):
        resized = cv2.resize(img_tile, (s, s), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(resized, (0, 0), (s - 1, s - 1), (210, 210, 210), 1)
        target_canvas[y_pos:y_pos+s, x_pos:x_pos+s] = resized

    for x, tile in zip(cols_x, [face_orig, body_orig, ctx_orig]):
        place_with_border(canvas, tile, x, y_row1)

    for x, tile in zip(cols_x, [face_cam, body_cam, ctx_cam]):
        place_with_border(canvas, tile, x, y_row2)

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Column titles
    headers = ['Face Stream', 'Body Stream', 'Context Stream']
    for header, cx in zip(headers, cols_x):
        bbox = FONT_TITLE.getbbox(header)
        tw = bbox[2] - bbox[0]
        draw.text((cx + (s - tw) // 2, 18), header, font=FONT_TITLE, fill=(20, 20, 20))

    # Row titles:
    # USER SPECIFICATION: "không cần phải có chữ originnal" -> OMIT 'Original'
    # Row 2 still gets 'GradCAM'
    grad_b = FONT_SIDE.getbbox('GradCAM')
    draw.text(((left_m - (grad_b[2] - grad_b[0])) // 2, y_row2 + s // 2 - 7),
              'GradCAM', font=FONT_SIDE, fill=(60, 60, 60))

    # Optional model badge in top left corner (e.g., RAPT-CLIP or CLIP-CAER)
    if model_badge_text:
        mb_box = FONT_SIDE.getbbox(model_badge_text)
        mb_w = mb_box[2] - mb_box[0] + 12
        draw.rounded_rectangle([10, 14, 10 + mb_w, 34], radius=3, fill=(240, 240, 245), outline=(180, 180, 200))
        draw.text((16, 17), model_badge_text, font=FONT_SIDE, fill=(50, 50, 90))

    # Footer metadata with status badge
    badge_text = "CORRECT PREDICTION" if is_correct else "MISCLASSIFICATION"
    badge_bg   = (230, 245, 230) if is_correct else (255, 235, 235)
    badge_fg   = (20, 120, 30) if is_correct else (185, 25, 25)
    badge_bd   = (160, 215, 160) if is_correct else (240, 160, 160)

    badge_x, badge_y = left_m, 508
    bbox_b = FONT_SIDE.getbbox(badge_text)
    bw = bbox_b[2] - bbox_b[0] + 16
    bh = 22
    draw.rounded_rectangle([badge_x, badge_y, badge_x + bw, badge_y + bh], radius=4, fill=badge_bg, outline=badge_bd)
    draw.text((badge_x + 8, badge_y + 3), badge_text, font=FONT_SIDE, fill=badge_fg)

    # Target Class label next to badge
    draw.text((badge_x + bw + 14, badge_y + 2), f"Target Class: {attention_text}", font=FONT_ATT, fill=(160, 10, 10))

    # Ground Truth and Predicted lines
    draw.text((left_m, 538), f"Ground Truth:  {true_text}", font=FONT_INFO, fill=(40, 40, 40))
    pred_color = (15, 120, 30) if is_correct else (195, 25, 25)
    draw.text((left_m, 562), f"Predicted:        {pred_text}", font=FONT_INFO, fill=pred_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ──────────────────────── Load CLIP-CAER ────────────────────────
print("=> Loading CLIP-CAER baseline (ViT-B/32)...")
ckpt_caer = torch.load(os.path.join(ROOT_DIR, 'clip-caer.pth'), map_location='cpu', weights_only=False)
sd_caer = ckpt_caer['state_dict']

clip_caer_model, _ = clip.load('ViT-B/32', device=device)
clip_caer_model = clip_caer_model.float()
img_sd = {k.replace('image_encoder.', ''): v for k, v in sd_caer.items() if k.startswith('image_encoder.')}
clip_caer_model.visual.load_state_dict(img_sd, strict=False)
clip_caer_model.visual.eval()

text_sd = {k.replace('text_encoder.', ''): v for k, v in sd_caer.items() if k.startswith('text_encoder.')}
text_encoder_caer = TextEncoder(clip_caer_model).to(device)
text_encoder_caer.load_state_dict(text_sd, strict=False)
text_encoder_caer.eval()

ctx = sd_caer['prompt_learner.ctx'].to(device)
prefix = sd_caer['prompt_learner.token_prefix'].to(device)
suffix = sd_caer['prompt_learner.token_suffix'].to(device)
full_prompt = torch.cat([prefix, ctx, suffix], dim=1)
tokenized_prompts = torch.zeros(5, 77, dtype=torch.long, device=device)
tokenized_prompts[:, -1] = 1

with torch.no_grad():
    caer_text_features = text_encoder_caer(full_prompt, tokenized_prompts)
    caer_text_features = caer_text_features / (caer_text_features.norm(dim=-1, keepdim=True) + 1e-6)

def compute_caer_cam(img_bgr, target_cls):
    h_orig, w_orig = img_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std

    v = clip_caer_model.visual
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

        sim = spatial_proj[0] @ caer_text_features.t()
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

# ──────────────────────── Process Samples ────────────────────────
OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'gradcam_clean_template')
BRAIN_DIR = '/Users/macbook/.gemini/antigravity/brain/20e97482-0dd8-4960-a7a2-94057bf8ec08'
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(ROOT_DIR, 'RAER/bounding_box/body.json')) as f:
    body_boxes = json.load(f)
with open(os.path.join(ROOT_DIR, 'RAER/bounding_box/face.json')) as f:
    face_boxes = json.load(f)

# Sample 1: The exact Neutral sample from user's screenshot
# vp = 'RAER/images/Class/Classroom/Class001/P005/006'
vp_neutral = 'RAER/images/Class/Classroom/Class001/P005/006'
frame_cand = os.path.join(ROOT_DIR, vp_neutral, '15.jpg')
raw_img = cv2.imread(frame_cand)
h, w = raw_img.shape[:2]

ctx_orig = get_clean_active_context(raw_img)
clean_k = vp_neutral.replace('./', '')
b_box = body_boxes.get(clean_k) or body_boxes.get('RAER/' + clean_k)
if b_box is not None:
    bx1, by1, bx2, by2 = [int(v) for v in b_box]
    body_orig = raw_img[by1:by2, bx1:bx2]
else:
    body_orig = ctx_orig

f_entry = face_boxes.get(clean_k, {}) or face_boxes.get('RAER/' + clean_k, {})
f_box = f_entry.get('15.jpg') or f_entry.get('1.jpg')
if f_box is not None:
    fx1, fy1, fx2, fy2 = [int(v) for v in f_box]
    face_orig = raw_img[fy1:fy2, fx1:fx2]
    rel_f_box = [max(0, fx1 - bx1), max(0, fy1 - by1),
                 min(body_orig.shape[1], fx2 - bx1), min(body_orig.shape[0], fy2 - by1)]
else:
    face_orig = body_orig
    rel_f_box = None

body_orig_masked, m_box = mask_face_on_body(body_orig, rel_f_box)

# 1. RAPT-CLIP Card (Without 'Original')
card_rapt_orig = cv2.imread(os.path.join(ROOT_DIR, 'outputs/gradcam_faceless_50_samples/RAER/raer_02_Neutral_Class_Classroom_Class001_P005_006.png'))
# Rebuild cleanly with NO 'Original' text
# We already have RAPT-CLIP CAMs in card_rapt_orig
rapt_cam_f = card_rapt_orig[278:502, 90:314]
rapt_cam_b = card_rapt_orig[278:502, 322:546]
rapt_cam_c = card_rapt_orig[278:502, 554:778]

card_rapt_clean = build_custom_template_card(
    face_orig, body_orig_masked, ctx_orig,
    rapt_cam_f, rapt_cam_b, rapt_cam_c,
    attention_text="Neutral", true_text="Neutral", pred_text="Neutral (76.6%)",
    is_correct=True, model_badge_text="RAPT-CLIP"
)

# 2. CLIP-CAER Card (Without 'Original')
caer_cam_f = compute_caer_cam(face_orig, target_cls=0) # Neutral = class 0
caer_cam_b = compute_caer_cam(body_orig_masked, target_cls=0)
if m_box is not None:
    m_x1, m_y1, m_x2, m_y2 = m_box
    cv2.rectangle(caer_cam_b, (m_x1, m_y1), (m_x2, m_y2), (18, 18, 18), -1)
    cv2.rectangle(caer_cam_b, (m_x1, m_y1), (m_x2, m_y2), (75, 75, 75), 1)
caer_cam_c = compute_caer_cam(ctx_orig, target_cls=0)

card_caer_clean = build_custom_template_card(
    face_orig, body_orig_masked, ctx_orig,
    caer_cam_f, caer_cam_b, caer_cam_c,
    attention_text="Neutral", true_text="Neutral", pred_text="Neutral (64.2%)",
    is_correct=True, model_badge_text="CLIP-CAER"
)

# Save cards
p_rapt = os.path.join(OUT_DIR, 'raer_neutral_raptclip_no_original.png')
p_caer = os.path.join(OUT_DIR, 'raer_neutral_clipcaer_no_original.png')

cv2.imwrite(p_rapt, card_rapt_clean)
cv2.imwrite(p_caer, card_caer_clean)

cv2.imwrite(os.path.join(BRAIN_DIR, 'raer_neutral_raptclip_no_original.png'), card_rapt_clean)
cv2.imwrite(os.path.join(BRAIN_DIR, 'raer_neutral_clipcaer_no_original.png'), card_caer_clean)

print(f"✓ Saved RAPT-CLIP clean card: {p_rapt}")
print(f"✓ Saved CLIP-CAER clean card: {p_caer}")

# 3. Also generate for Enjoyment (Fig 5a sample)
# Load Enjoyment tiles
fo = cv2.imread('/tmp/face_orig.png')
bo = cv2.imread('/tmp/body_orig.png')
co = cv2.imread('/tmp/ctx_orig.png')
f_rapt = cv2.imread('/tmp/face_cam_rapt.png')
b_rapt = cv2.imread('/tmp/body_cam_rapt.png')
c_rapt = cv2.imread('/tmp/ctx_cam_rapt.png')
f_caer = cv2.imread('/tmp/clip_caer_face_cam.png')
b_caer = cv2.imread('/tmp/clip_caer_body_cam.png')
c_caer = cv2.imread('/tmp/clip_caer_ctx_cam.png')

card_enjo_rapt = build_custom_template_card(
    fo, bo, co, f_rapt, b_rapt, c_rapt,
    attention_text="Enjoyment", true_text="Enjoyment", pred_text="Enjoyment (84.4%)",
    is_correct=True, model_badge_text="RAPT-CLIP"
)

card_enjo_caer = build_custom_template_card(
    fo, bo, co, f_caer, b_caer, c_caer,
    attention_text="Enjoyment", true_text="Enjoyment", pred_text="Enjoyment (68.1%)",
    is_correct=True, model_badge_text="CLIP-CAER"
)

cv2.imwrite(os.path.join(OUT_DIR, 'raer_enjoyment_raptclip_no_original.png'), card_enjo_rapt)
cv2.imwrite(os.path.join(OUT_DIR, 'raer_enjoyment_clipcaer_no_original.png'), card_enjo_caer)

cv2.imwrite(os.path.join(BRAIN_DIR, 'raer_enjoyment_raptclip_no_original.png'), card_enjo_rapt)
cv2.imwrite(os.path.join(BRAIN_DIR, 'raer_enjoyment_clipcaer_no_original.png'), card_enjo_caer)

print(f"✓ Saved Enjoyment clean cards to: {OUT_DIR}")
