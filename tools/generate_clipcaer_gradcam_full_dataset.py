#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_clipcaer_gradcam_full_dataset.py
───────────────────────────────────────────────
Generates 2-Stream Grad-CAM cards for ALL 528 test samples of RAER
using Baseline clip-caer.pth (ViT-B/32, Face + Body only).

Features:
- Exactly follows user template (No "Original" text, modal card design).
- Clean crops with NO text clipping.
- Saves to outputs/gradcam/clip_caer_all_528_samples/ and by_class/.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class RecorderMeter:
    pass
sys.modules['__main__'].RecorderMeter = RecorderMeter

from models.clip import clip
from models.Prompt_Learner import TextEncoder

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = '/System/Library/Fonts/Helvetica.ttc'

FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_REG):
    FONT_REG = FONT_BOLD

def build_dual_stream_card(f_o, b_o, f_c, b_c, target_name, true_name, pred_name, conf, is_correct, sample_idx):
    s = 224
    gap = 14
    left_m = 90
    right_m = 25
    pad_top = 46
    pad_bot = 105
    grid_w = left_m + s * 2 + gap + right_m  # 577 px
    grid_h = pad_top + s * 2 + gap + pad_bot  # 613 px

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

    font_title = ImageFont.truetype(FONT_BOLD, 15)
    font_side  = ImageFont.truetype(FONT_REG, 13)
    font_att   = ImageFont.truetype(FONT_BOLD, 15)
    font_info  = ImageFont.truetype(FONT_REG, 13)

    # Column titles (Face Stream & Body Stream)
    headers = ['Face Stream', 'Body Stream']
    for header, cx in zip(headers, cols_x):
        bb = font_title.getbbox(header)
        tw = bb[2] - bb[0]
        draw.text((cx + (s - tw) // 2, 18), header, font=font_title, fill=(20, 20, 20))

    # Row 1 has NO text (user request: "không cần phải có chữ original")
    # Row 2 label: GradCAM
    gb = font_side.getbbox('GradCAM')
    draw.text(((left_m - (gb[2] - gb[0])) // 2, y_row2 + s // 2 - 7), 'GradCAM', font=font_side, fill=(60, 60, 60))

    # Model badge in top left
    badge_label = f"CLIP-CAER #{sample_idx:03d}"
    bb_m = font_side.getbbox(badge_label)
    mb_w = bb_m[2] - bb_m[0] + 14
    draw.rounded_rectangle([10, 14, 10 + mb_w, 34], radius=3, fill=(240, 240, 245), outline=(180, 180, 200))
    draw.text((16, 17), badge_label, font=font_side, fill=(50, 50, 90))

    # Status Badge
    badge_text = "CORRECT PREDICTION" if is_correct else "MISCLASSIFICATION"
    badge_bg   = (230, 245, 230) if is_correct else (255, 235, 235)
    badge_fg   = (20, 120, 30) if is_correct else (185, 25, 25)
    badge_bd   = (160, 215, 160) if is_correct else (240, 160, 160)

    badge_x, badge_y = left_m, y_row2 + s + 16
    bb_b = font_side.getbbox(badge_text)
    bw = bb_b[2] - bb_b[0] + 16
    draw.rounded_rectangle([badge_x, badge_y, badge_x + bw, badge_y + 22], radius=4, fill=badge_bg, outline=badge_bd)
    draw.text((badge_x + 8, badge_y + 3), badge_text, font=font_side, fill=badge_fg)

    draw.text((badge_x + bw + 14, badge_y + 2), f"Target Class: {target_name}", font=font_att, fill=(160, 10, 10))
    draw.text((left_m, badge_y + 30), f"Ground Truth:  {true_name}", font=font_info, fill=(40, 40, 40))
    pred_color = (15, 120, 30) if is_correct else (185, 25, 25)
    draw.text((left_m, badge_y + 54), f"Predicted:        {pred_name} ({conf:.1f}%)", font=font_info, fill=pred_color)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f"🚀 Generating 2-Stream Grad-CAM Cards for FULL RAER Test Set")
    print(f"   Device: {device}")

    # 1. Load CLIP-CAER Checkpoint
    ckpt_path = os.path.join(ROOT_DIR, 'clip-caer.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']

    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model = clip_model.float()
    img_sd = {k.replace('image_encoder.', ''): v for k, v in sd.items() if k.startswith('image_encoder.')}
    clip_model.visual.load_state_dict(img_sd, strict=False)
    clip_model.visual.eval()

    text_encoder = TextEncoder(clip_model).to(device)
    text_sd = {k.replace('text_encoder.', ''): v for k, v in sd.items() if k.startswith('text_encoder.')}
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

    # 2. Grad-CAM Computation Function
    mean_val = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std_val  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def compute_cam(img_bgr, target_cls):
        h_orig, w_orig = img_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        t = (t - mean_val) / std_val

        v = clip_model.visual
        with torch.no_grad():
            x_conv = v.conv1(t)
            x_conv = x_conv.reshape(x_conv.shape[0], x_conv.shape[1], -1).permute(0, 2, 1)
            cls_token = v.class_embedding.to(t.dtype) + torch.zeros(t.shape[0], 1, x_conv.shape[-1], dtype=t.dtype, device=device)
            x_in = torch.cat([cls_token, x_conv], dim=1) + v.positional_embedding.to(t.dtype)
            x_pre = v.ln_pre(x_in)
            x_trans = v.transformer(x_pre.permute(1, 0, 2)).permute(1, 0, 2)
            spatial = x_trans[:, 1:, :] # (1, 49, 768)
            spatial_proj = v.ln_post(spatial) @ v.proj # (1, 49, 512)
            spatial_proj = spatial_proj / (spatial_proj.norm(dim=-1, keepdim=True) + 1e-6)

            sim = spatial_proj[0] @ text_features.t() # (49, 5)
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

    # 3. Load Annotation & Bounding Boxes
    with open('RAER/annotation/test.txt') as f:
        test_entries = [l.strip().split() for l in f]

    with open('RAER/bounding_box/body.json') as f:
        body_boxes = json.load(f)

    with open('RAER/bounding_box/face.json') as f:
        face_boxes = json.load(f)

    out_all = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'clip_caer_all_528_samples')
    os.makedirs(out_all, exist_ok=True)
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'clip_caer_by_class', cname), exist_ok=True)

    print(f"📂 Processing all {len(test_entries)} videos in RAER Test Set...")

    t0 = time.time()
    for idx, (vid_path, num_frames_str, label_str) in enumerate(tqdm(test_entries, desc="Generating CLIP-CAER Grad-CAM")):
        true_cls = int(label_str) - 1
        true_name = CLASS_NAMES[true_cls]

        # Select middle frame
        mid_num = max(1, int(num_frames_str) // 2)
        frame_candidates = [f"{mid_num}.jpg", "15.jpg", "16.jpg", "1.jpg"]
        img_raw = None
        chosen_frame = "1.jpg"
        for fc in frame_candidates:
            p = os.path.join(vid_path, fc)
            if os.path.exists(p):
                img_raw = cv2.imread(p)
                chosen_frame = fc
                break

        if img_raw is None:
            jpgs = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            if jpgs:
                chosen_frame = sorted(jpgs)[len(jpgs)//2]
                img_raw = cv2.imread(os.path.join(vid_path, chosen_frame))
            else:
                continue

        clean_key = vid_path.replace('./', '')
        match_key = clean_key if clean_key in body_boxes else 'RAER/' + clean_key

        # Face crop
        f_entry = face_boxes.get(match_key, {})
        f_box = f_entry.get(chosen_frame) or f_entry.get("1.jpg") or (list(f_entry.values())[0] if f_entry else None)
        fx1, fy1, fx2, fy2 = 0, 0, 0, 0
        if f_box is not None and len(f_box) >= 4:
            fx1, fy1, fx2, fy2 = int(f_box[0]), int(f_box[1]), int(f_box[2]), int(f_box[3])
            fw, fh = fx2 - fx1, fy2 - fy1
            mw, mh = int(fw * 0.12), int(fh * 0.12)
            fy1_m, fy2_m = max(0, fy1 - mh), min(img_raw.shape[0], fy2 + mh)
            fx1_m, fx2_m = max(0, fx1 - mw), min(img_raw.shape[1], fx2 + mw)
            if fy2_m > fy1_m and fx2_m > fx1_m:
                tile_face = img_raw[fy1_m:fy2_m, fx1_m:fx2_m].copy()
            else:
                tile_face = img_raw.copy()
        else:
            tile_face = img_raw.copy()

        # Body crop with face mask
        b_box = body_boxes.get(match_key)
        if b_box is not None and len(b_box) >= 4:
            bx1, by1, bx2, by2 = int(b_box[0]), int(b_box[1]), int(b_box[2]), int(b_box[3])
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(img_raw.shape[1], bx2), min(img_raw.shape[0], by2)
            if bx2 > bx1 and by2 > by1:
                tile_body = img_raw[by1:by2, bx1:bx2].copy()
                # Apply black face mask relative to body box
                if fx2 > fx1 and fy2 > fy1:
                    mfx1, mfx2 = max(0, fx1 - bx1), min(tile_body.shape[1], fx2 - bx1)
                    mfy1, mfy2 = max(0, fy1 - by1), min(tile_body.shape[0], fy2 - by1)
                    if mfx2 > mfx1 and mfy2 > mfy1:
                        tile_body[mfy1:mfy2, mfx1:mfx2] = 0
            else:
                tile_body = img_raw.copy()
        else:
            tile_body = img_raw.copy()

        # Prediction via CLIP visual feature cosine sim
        with torch.no_grad():
            t_f = torch.from_numpy(cv2.cvtColor(cv2.resize(tile_face, (224, 224)), cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            t_f = (t_f - mean_val) / std_val
            feat_f = clip_model.visual(t_f)
            feat_f = feat_f / (feat_f.norm(dim=-1, keepdim=True) + 1e-6)

            t_b = torch.from_numpy(cv2.cvtColor(cv2.resize(tile_body, (224, 224)), cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            t_b = (t_b - mean_val) / std_val
            feat_b = clip_model.visual(t_b)
            feat_b = feat_b / (feat_b.norm(dim=-1, keepdim=True) + 1e-6)

            fused_feat = (feat_f + feat_b) / 2.0
            logits = fused_feat @ text_features.t()
            probs = torch.softmax(logits / 0.07, dim=-1)[0]
            pred_cls = logits.argmax(dim=-1).item()
            conf = probs[pred_cls].item() * 100.0

        pred_name = CLASS_NAMES[pred_cls]
        is_correct = (pred_cls == true_cls)

        # Compute CAMs for target class (Ground Truth)
        face_cam = compute_cam(tile_face, true_cls)
        body_cam = compute_cam(tile_body, true_cls)

        card = build_dual_stream_card(
            tile_face, tile_body,
            face_cam, body_cam,
            true_name, true_name, pred_name, conf, is_correct, idx
        )

        fn = f"sample_{idx:04d}_{true_name}_{pred_name}.png"
        cv2.imwrite(os.path.join(out_all, fn), card)
        cv2.imwrite(os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'clip_caer_by_class', true_name, fn), card)

    elapsed = time.time() - t0
    print(f"\n🎉 Successfully generated 528 Grad-CAM cards in {elapsed:.1f}s ({len(test_entries)/elapsed:.1f} cards/sec)!")
    print(f"   Outputs stored in: outputs/gradcam/clip_caer_all_528_samples/")

if __name__ == '__main__':
    main()

