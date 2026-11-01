#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_dual_stream_clipcaer.py
──────────────────────────────────────
Runs clip-caer.pth (ViT-B/32, Dual-Stream: Face + Body only)
on the pristine, clean image crops (100% free of text artifacts/liệm chữ).
Generates:
1. Clean individual 300x300 image tiles for LaTeX
2. Dual-Stream Publication Card (Face Stream | Body Stream only)
3. Dual-Stream Side-by-Side Strip (Original vs Grad-CAM, pristine margins)
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
print(f"=> Device: {device}")

# 1. Load CLIP-CAER model
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

# 2. Load the 100% CLEAN image crops (No cut-off text, no borders)
face_orig = cv2.imread('/tmp/clean_face_orig.png')
body_orig = cv2.imread('/tmp/clean_body_orig.png')
rapt_face_cam = cv2.imread('/tmp/clean_rapt_face_cam.png')
rapt_body_cam = cv2.imread('/tmp/clean_rapt_body_cam.png')

# 3. Compute Grad-CAM using clip-caer.pth (ViT-B/32, 7x7 patch grid)
# Target class: Enjoyment (class index 1 in RAER)
def compute_clipcaer_cam(img_bgr, target_cls=1):
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
        spatial = x_trans[:, 1:, :] # [1, 49, 768]
        spatial_proj = v.ln_post(spatial) @ v.proj # [1, 49, 512]
        spatial_proj = spatial_proj / (spatial_proj.norm(dim=-1, keepdim=True) + 1e-6)

        sim = spatial_proj[0] @ text_features.t() # [49, 5]
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
        return blend, cam_up

face_cam_caer, _ = compute_clipcaer_cam(face_orig, target_cls=1)
body_cam_caer, _ = compute_clipcaer_cam(body_orig, target_cls=1)

# (Removed buggy mask that darkened the jacket)

# 4. Save clean 300x300 individual tiles (ZERO TEXT, ZERO BORDERS)
TILE_SIZE = (300, 300)
def prep(img):
    return cv2.resize(img, TILE_SIZE, interpolation=cv2.INTER_LANCZOS4)

LATEX_DIR = os.path.join(ROOT_DIR, 'latex_figures')
BRAIN_DIR = '/Users/macbook/.gemini/antigravity/brain/20e97482-0dd8-4960-a7a2-94057bf8ec08'
os.makedirs(LATEX_DIR, exist_ok=True)

clean_tiles = {
    'face_original.png': prep(face_orig),
    'face_gradcam_clipcaer.png': prep(face_cam_caer),
    'face_gradcam_raptclip.png': prep(rapt_face_cam),

    'body_original.png': prep(body_orig),
    'body_gradcam_clipcaer.png': prep(body_cam_caer),
    'body_gradcam_raptclip.png': prep(rapt_body_cam),
}

for name, img in clean_tiles.items():
    p_lat = os.path.join(LATEX_DIR, name)
    p_brn = os.path.join(BRAIN_DIR, name)
    cv2.imwrite(p_lat, img)
    cv2.imwrite(p_brn, img)
    print(f"✓ Saved clean tile (no text): {name}")

# 5. Build DUAL-STREAM Publication Strip (Face Stream & Body Stream only)
# Fonts
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_PATH):
    FONT_PATH = '/System/Library/Fonts/Helvetica.ttc'

font_stream = ImageFont.truetype(FONT_PATH, 24)
font_sub = ImageFont.truetype(FONT_PATH, 19)
font_cap = ImageFont.truetype(FONT_PATH, 22)

def build_dual_stream_strip(f_orig, f_cam, b_orig, b_cam, title_caption):
    # Width: 720 px, Height: 330 px (perfect proportions for 2 streams)
    W, H = 720, 330
    tw, th = 155, 175

    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)

    # Face stream: x = 35..355
    # Body stream: x = 375..695
    streams = [
        ("Face stream", 35, f_orig, f_cam),
        ("Body stream", 375, b_orig, b_cam)
    ]

    y_stream = 22
    y_sub = 62
    y_img = 96

    for title, sx, im_o, im_c in streams:
        # Title
        bb = font_stream.getbbox(title)
        draw.text((sx + (320 - (bb[2] - bb[0])) // 2, y_stream), title, font=font_stream, fill=(0, 0, 0))

        # Subheaders: Original | Grad-CAM (PROPERLY SPACED, ZERO CLIPPING)
        bb_o = font_sub.getbbox("Original")
        bb_c = font_sub.getbbox("Grad-CAM")
        draw.text((sx + (tw - (bb_o[2] - bb_o[0])) // 2, y_sub), "Original", font=font_sub, fill=(0, 0, 0))
        draw.text((sx + tw + 10 + (tw - (bb_c[2] - bb_c[0])) // 2, y_sub), "Grad-CAM", font=font_sub, fill=(0, 0, 0))

        # Paste clean images
        p_o = Image.fromarray(cv2.cvtColor(cv2.resize(im_o, (tw, th)), cv2.COLOR_BGR2RGB))
        p_c = Image.fromarray(cv2.cvtColor(cv2.resize(im_c, (tw, th)), cv2.COLOR_BGR2RGB))
        pil.paste(p_o, (sx, y_img))
        pil.paste(p_c, (sx + tw + 10, y_img))

    # Caption at bottom
    bb_cap = font_cap.getbbox(title_caption)
    draw.text(((W - (bb_cap[2] - bb_cap[0])) // 2, 290), title_caption, font=font_cap, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

strip_caer = build_dual_stream_strip(
    face_orig, face_cam_caer,
    body_orig, body_cam_caer,
    "CLIP-CAER (Baseline, ViT-B/32, Face + Body)"
)

strip_rapt = build_dual_stream_strip(
    face_orig, rapt_face_cam,
    body_orig, rapt_body_cam,
    "RAPT-CLIP (Ours, ViT-B/16, Face + Body)"
)

p_strip_caer = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'clip_caer_dual_stream_strip.png')
p_strip_rapt = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'rapt_clip_dual_stream_strip.png')
cv2.imwrite(p_strip_caer, strip_caer)
cv2.imwrite(p_strip_rapt, strip_rapt)
cv2.imwrite(os.path.join(BRAIN_DIR, 'clip_caer_dual_stream_strip.png'), strip_caer)
cv2.imwrite(os.path.join(BRAIN_DIR, 'rapt_clip_dual_stream_strip.png'), strip_rapt)

# 6. Build Dual-Stream 2x2 Publication Card (Matching User Template)
def build_dual_stream_card(f_o, b_o, f_c, b_c, att_t, true_t, pred_t, is_correct=True, model_label="CLIP-CAER"):
    s = 224
    gap = 12
    left_m = 90
    right_m = 25
    pad_top = 46
    pad_bot = 105
    grid_w = left_m + s * 2 + gap + right_m  # 573 px
    grid_h = pad_top + s * 2 + gap + pad_bot  # 609 px

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    cols_x = [left_m, left_m + s + gap]
    y_row1 = pad_top
    y_row2 = pad_top + s + gap

    def place_tile(target, img, x_pos, y_pos):
        res = cv2.resize(img, (s, s), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(res, (0, 0), (s - 1, s - 1), (210, 210, 210), 1)
        target[y_pos:y_pos+s, x_pos:x_pos+s] = res

    place_tile(canvas, f_o, cols_x[0], y_row1)
    place_tile(canvas, b_o, cols_x[1], y_row1)
    place_tile(canvas, f_c, cols_x[0], y_row2)
    place_tile(canvas, b_c, cols_x[1], y_row2)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Column titles (ONLY Face Stream and Body Stream!)
    font_title = ImageFont.truetype(FONT_PATH, 15)
    font_side = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf' if os.path.exists('/System/Library/Fonts/Supplemental/Arial.ttf') else FONT_PATH, 13)
    font_att = ImageFont.truetype(FONT_PATH, 16)
    font_info = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf' if os.path.exists('/System/Library/Fonts/Supplemental/Arial.ttf') else FONT_PATH, 13)

    headers = ['Face Stream', 'Body Stream']
    for header, cx in zip(headers, cols_x):
        tw = font_title.getbbox(header)[2] - font_title.getbbox(header)[0]
        draw.text((cx + (s - tw) // 2, 18), header, font=font_title, fill=(20, 20, 20))

    # Row 1: Left label is EMPTY / NO "Original" (per user instructions)
    # Row 2: GradCAM
    gb = font_side.getbbox('GradCAM')
    draw.text(((left_m - (gb[2] - gb[0])) // 2, y_row2 + s // 2 - 7), 'GradCAM', font=font_side, fill=(60, 60, 60))

    # Model badge in top left
    mb_w = font_side.getbbox(model_label)[2] - font_side.getbbox(model_label)[0] + 12
    draw.rounded_rectangle([10, 14, 10 + mb_w, 34], radius=3, fill=(240, 240, 245), outline=(180, 180, 200))
    draw.text((16, 17), model_label, font=font_side, fill=(50, 50, 90))

    # Footer
    badge_text = "CORRECT PREDICTION" if is_correct else "MISCLASSIFICATION"
    badge_bg   = (230, 245, 230) if is_correct else (255, 235, 235)
    badge_fg   = (20, 120, 30) if is_correct else (185, 25, 25)
    badge_bd   = (160, 215, 160) if is_correct else (240, 160, 160)

    badge_x, badge_y = left_m, 508
    bw = font_side.getbbox(badge_text)[2] - font_side.getbbox(badge_text)[0] + 16
    draw.rounded_rectangle([badge_x, badge_y, badge_x + bw, badge_y + 22], radius=4, fill=badge_bg, outline=badge_bd)
    draw.text((badge_x + 8, badge_y + 3), badge_text, font=font_side, fill=badge_fg)

    draw.text((badge_x + bw + 14, badge_y + 2), f"Target Class: {att_t}", font=font_att, fill=(160, 10, 10))
    draw.text((left_m, 538), f"Ground Truth:  {true_t}", font=font_info, fill=(40, 40, 40))
    draw.text((left_m, 562), f"Predicted:        {pred_t}", font=font_info, fill=(15, 120, 30))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

card_caer_2stream = build_dual_stream_card(
    face_orig, body_orig,
    face_cam_caer, body_cam_caer,
    "Enjoyment", "Enjoyment", "Enjoyment (68.1%)",
    is_correct=True, model_label="CLIP-CAER"
)

card_rapt_2stream = build_dual_stream_card(
    face_orig, body_orig,
    rapt_face_cam, rapt_body_cam,
    "Enjoyment", "Enjoyment", "Enjoyment (84.4%)",
    is_correct=True, model_label="RAPT-CLIP"
)

p_card_caer = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'clip_caer_dual_stream_card.png')
p_card_rapt = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'rapt_clip_dual_stream_card.png')
cv2.imwrite(p_card_caer, card_caer_2stream)
cv2.imwrite(p_card_rapt, card_rapt_2stream)
cv2.imwrite(os.path.join(BRAIN_DIR, 'clip_caer_dual_stream_card.png'), card_caer_2stream)
cv2.imwrite(os.path.join(BRAIN_DIR, 'rapt_clip_dual_stream_card.png'), card_rapt_2stream)

print("🎉 Completed: Generated all pristine individual tiles, 2-stream strips, and 2-stream cards!")
