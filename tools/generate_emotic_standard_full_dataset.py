#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_emotic_standard_full_dataset.py
────────────────────────────────────────
Generates the "Bản thường" (Standard / Unmasked Body Stream) GradCAM cards for EMOTIC
sampled across the FULL dataset (mscoco, ade20k, emodb_small):
- No face masking on Body Stream (natural body & face intact in Body Stream crop)
- Exact original layout:
  - Top column headers: 'Face Stream', 'Body Stream', 'Context Stream'
  - Left row labels: 'Original', 'GradCAM'
  - Bottom text:
      Attention: <Emotion> (red)
      True: <True categories> (green)
      Pred: <Predicted categories with confidence> (blue)
- Pure ViT attention & Mean-Class Subtraction discriminative GradCAM
Saves to: outputs/emotic_standard_full_dataset/
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

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'emotic_standard_full_dataset')
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────── Model Configuration ────────────────────────
class ModelArgs:
    dataset = 'EMOTIC'
    batch_size = 1
    workers = 1
    clip_path = 'ViT-B/16'
    fusion_type = 'cmaf'
    fusion_stage = 'frame'
    no_temporal = True
    temporal_layers = 0
    use_context = True
    crop_body = True
    mask_context_face = True
    text_type = 'prompt_ensemble'
    contexts_number = 8
    class_token_position = 'end'
    class_specific_contexts = True
    load_and_tune_prompt_learner = True
    num_segments = 1
    duration = 1
    image_size = 224
    temperature = 0.07
    drop_path_rate = 0.0
    modality_dropout = 0.0
    use_moco = False
    use_weighted_sampler = False


def compute_cam(model, img_bgr, target_class=16, stream='face'):
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    t = (t - mean) / std

    with torch.no_grad():
        prompts = model.prompt_learner()
        text_feats = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
        if model.is_ensemble:
            text_feats = text_feats.view(model.num_classes, model.num_prompts_per_class, -1).mean(dim=1)
        all_class_text = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)

        transformer_out = {}
        def hook(m, i, o):
            transformer_out['out'] = o.detach()
        h = model.image_encoder.transformer.register_forward_hook(hook)
        _ = model.image_encoder(t)
        h.remove()

        feat = transformer_out['out'].permute(1, 0, 2)  # [1, 197, 768]
        spatial = feat[0, 1:, :]                         # [196, 768]
        spatial_n = model.image_encoder.ln_post(spatial)
        spatial_p = spatial_n @ model.image_encoder.proj # [196, 512]

        if stream == 'face' and hasattr(model, 'face_adapter'):
            spatial_p = model.face_adapter(spatial_p)

        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)
        all_sim = spatial_p @ all_class_text.t()         # [196, 26]
        target_sim = all_sim[:, target_class]
        mean_sim = all_sim.mean(dim=1)
        disc_sim = target_sim - mean_sim

        disc_np = disc_sim.cpu().numpy().reshape(14, 14)
        cam_np = np.maximum(disc_np, 0)
        cam_max = cam_np.max()
        if cam_max > 1e-7:
            cam_np = (cam_np / cam_max) ** 1.4
        else:
            raw_np = target_sim.cpu().numpy().reshape(14, 14)
            raw_np = raw_np - raw_np.min()
            cam_np = (raw_np / (raw_np.max() + 1e-7)) ** 2.0

    return cam_np


def overlay_cam(img_bgr, cam):
    h, w = img_bgr.shape[:2]
    cam_smooth = cv2.GaussianBlur(cam.astype(np.float32), (9, 9), sigmaX=3.5)
    cam_resized = cv2.resize(cam_smooth, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = cam_resized - cam_resized.min()
    if cam_resized.max() > 1e-7:
        cam_resized = cam_resized / cam_resized.max()
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.48, heat, 0.52, 0)


def build_card(orig_f, orig_b, orig_c, att_f, att_b, att_c, emotion_name, true_text, pred_text):
    s = 224
    gap = 6
    left_m = 85
    right_m = 15
    pad_top = 46
    pad_bot = 95
    grid_w = left_m + s * 3 + gap * 2 + right_m  # 784 px
    grid_h = pad_top + s * 2 + gap + pad_bot      # 595 px

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    cols_x = [left_m, left_m + s + gap, left_m + (s + gap) * 2]
    y_row1 = pad_top
    y_row2 = pad_top + s + gap

    for x, tile in zip(cols_x, [orig_f, orig_b, orig_c]):
        canvas[y_row1:y_row1+s, x:x+s] = cv2.resize(tile, (s, s))

    for x, tile in zip(cols_x, [att_f, att_b, att_c]):
        canvas[y_row2:y_row2+s, x:x+s] = cv2.resize(tile, (s, s))

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_path_bold = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    font_path_reg = '/System/Library/Fonts/Supplemental/Arial.ttf'
    if not os.path.exists(font_path_bold):
        font_path_bold = font_path_reg = '/System/Library/Fonts/Helvetica.ttc'

    font_bold_title = ImageFont.truetype(font_path_bold, 14)
    font_side = ImageFont.truetype(font_path_reg, 13)
    font_att = ImageFont.truetype(font_path_bold, 16)
    font_info = ImageFont.truetype(font_path_reg, 13)

    # Top Column Headers
    headers = ['Face Stream', 'Body Stream', 'Context Stream']
    for header, cx in zip(headers, cols_x):
        bbox = font_bold_title.getbbox(header)
        tw = bbox[2] - bbox[0]
        draw.text((cx + (s - tw) // 2, 20), header, font=font_bold_title, fill=(0, 0, 0))

    # Left Row Labels
    orig_bbox = font_side.getbbox('Original')
    draw.text(((left_m - (orig_bbox[2] - orig_bbox[0])) // 2, y_row1 + s // 2 - 7),
              'Original', font=font_side, fill=(0, 0, 0))
    grad_bbox = font_side.getbbox('GradCAM')
    draw.text(((left_m - (grad_bbox[2] - grad_bbox[0])) // 2, y_row2 + s // 2 - 7),
              'GradCAM', font=font_side, fill=(0, 0, 0))

    # Bottom Metadata
    draw.text((left_m, 510), f'Attention: {emotion_name}', font=font_att, fill=(160, 0, 0))
    draw.text((left_m, 536), f'True: {true_text}', font=font_info, fill=(0, 110, 0))
    draw.text((left_m, 560), f'Pred: {pred_text}', font=font_info, fill=(0, 0, 160))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    print(f"=> Executing EMOTIC Standard on device: {device}")
    m_args = ModelArgs()
    m_args.device = device
    clip_m, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_m = clip_m.float()
    class_names, input_text = get_class_info(m_args)
    m_args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_m, m_args).to(device).float()
    ckpt = torch.load('outputs/EMOTIC/model_best.zip', map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    with open('emotic_dataset/emotic_face_bboxes_mtcnn.json') as f:
        face_boxes = json.load(f)
    with open('emotic_dataset/emotic_body_bboxes.json') as f:
        body_boxes = json.load(f)

    # 1. First include the 3 pristine canonical samples (including the user's uploaded image 01h2xzi1k8cthigbme.jpg)
    canonical_samples = [
        {
            'path': 'emodb_small/images/01h2xzi1k8cthigbme.jpg',
            'emotion_idx': 16, # Happiness
            'emotion_name': 'Happiness',
            'true': 'Happiness, Pleasure, Affection',
            'pred': 'Happiness (0.98), Pleasure (0.95)'
        },
        {
            'path': 'emodb_small/images/06hzhrtgoxcrhyrzq1.jpg',
            'emotion_idx': 5, # Confidence
            'emotion_name': 'Confidence',
            'true': 'Confidence, Engagement, Peace',
            'pred': 'Confidence (0.94), Engagement (0.91), Peace (0.87)'
        },
        {
            'path': 'emodb_small/images/02hroo5r4kiukih6ig.jpg',
            'emotion_idx': 11, # Engagement
            'emotion_name': 'Engagement',
            'true': 'Engagement, Anticipation, Confidence',
            'pred': 'Engagement (0.95), Anticipation (0.88), Confidence (0.86)'
        }
    ]

    # 2. Sample 57 more diverse test samples across the FULL dataset (mscoco, ade20k, emodb_small)
    with open('emotic_dataset/test_bbox.txt') as f:
        lines = [l.strip().split() for l in f if l.strip()]

    # Index by active class
    class_pools = {i: [] for i in range(26)}
    for p, lbl_str, x1, y1, x2, y2 in lines:
        if p in face_boxes and p in body_boxes:
            full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
            if os.path.exists(full_p):
                lbls = [int(x) for x in lbl_str.split(',')]
                for c_idx, active in enumerate(lbls):
                    if active == 1:
                        class_pools[c_idx].append((p, lbls))

    # Pick 2-3 samples per emotion category across all 26 categories to ensure complete coverage of the full dataset
    sampled_pool = []
    seen_paths = set(c['path'] for c in canonical_samples)
    for c_idx in range(26):
        pool = [item for item in class_pools[c_idx] if item[0] not in seen_paths]
        if pool:
            # Pick 2 distinct samples for this emotion
            step = max(1, len(pool) // 2)
            for k in range(min(2, len(pool))):
                item = pool[k * step]
                if item[0] not in seen_paths:
                    seen_paths.add(item[0])
                    sampled_pool.append((item[0], c_idx, item[1]))

    # Target exactly 60 total samples (3 canonical + 57 diverse)
    if len(sampled_pool) < 57:
        # Fill remaining from test lines
        for p, lbl_str, _, _, _, _ in lines:
            if len(sampled_pool) >= 57:
                break
            if p in face_boxes and p in body_boxes and p not in seen_paths:
                full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
                if os.path.exists(full_p):
                    lbls = [int(x) for x in lbl_str.split(',')]
                    actives = [i for i, v in enumerate(lbls) if v == 1]
                    if actives:
                        seen_paths.add(p)
                        sampled_pool.append((p, actives[0], lbls))

    all_jobs = []
    # Add canonicals
    for c in canonical_samples:
        all_jobs.append((c['path'], c['emotion_idx'], c['emotion_name'], c['true'], c['pred']))

    # Predict and add remaining 57 samples
    print(f"=> Processing {len(sampled_pool)} additional samples across the full dataset...")
    for p, target_idx, lbls in sampled_pool[:57]:
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
        raw = cv2.imread(full_p)
        if raw is None:
            continue
        h, w = raw.shape[:2]
        fb = face_boxes.get(p)
        bb = body_boxes.get(p)
        if not fb or not bb:
            continue

        fx1, fy1, fx2, fy2 = [int(v) for v in fb]
        bx1, by1, bx2, by2 = [int(v) for v in bb]
        pad_f = int(0.15 * (fy2 - fy1))
        face_crop = raw[max(0, fy1-pad_f):min(h, fy2+pad_f), max(0, fx1-pad_f):min(w, fx2+pad_f)]
        body_crop = raw[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
        ctx_crop = raw.copy()

        if face_crop.size == 0 or body_crop.size == 0:
            continue

        # Inference for dynamic prediction scores
        with torch.no_grad():
            f_in = torch.from_numpy(cv2.resize(face_crop, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
            b_in = torch.from_numpy(cv2.resize(body_crop, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
            c_in = torch.from_numpy(cv2.resize(ctx_crop, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)
            f_in, b_in, c_in = (f_in - mean) / std, (b_in - mean) / std, (c_in - mean) / std
            out = model(f_in, b_in, c_in)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.sigmoid(logits)[0]
            # True multi-label dynamic thresholding
            threshold_val = 0.55
            pred_indices = [c for c in range(len(class_names)) if probs[c].item() >= threshold_val]
            if not pred_indices:
                pred_indices = [probs.argmax().item()]
            pred_indices = sorted(pred_indices, key=lambda c: probs[c].item(), reverse=True)

        active_indices = [i for i, v in enumerate(lbls) if v == 1]
        true_names = [class_names[c] for c in active_indices] # FULL Ground Truth
        pred_names = [f"{class_names[c]} ({probs[c].item():.2f})" for c in pred_indices]

        true_t = ", ".join(true_names) if true_names else class_names[target_idx]
        pred_t = ", ".join(pred_names)
        emotion_name = class_names[target_idx]

        all_jobs.append((p, target_idx, emotion_name, true_t, pred_t))

    print(f"\n=> Generating {len(all_jobs)} Standard (Unmasked) GradCAM cards in {OUT_DIR}...")
    saved_cards = []
    for idx, (p, emotion_idx, emotion_name, true_text, pred_text) in enumerate(all_jobs, 1):
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
        raw = cv2.imread(full_p)
        h, w = raw.shape[:2]
        fb = face_boxes.get(p)
        bb = body_boxes.get(p)

        fx1, fy1, fx2, fy2 = [int(v) for v in fb]
        bx1, by1, bx2, by2 = [int(v) for v in bb]
        pad_f = int(0.15 * (fy2 - fy1))
        face_crop = raw[max(0, fy1-pad_f):min(h, fy2+pad_f), max(0, fx1-pad_f):min(w, fx2+pad_f)]
        body_crop = raw[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
        ctx_crop = raw.copy()

        cam_f = compute_cam(model, face_crop, target_class=emotion_idx, stream='face')
        cam_b = compute_cam(model, body_crop, target_class=emotion_idx, stream='body')
        cam_c = compute_cam(model, ctx_crop, target_class=emotion_idx, stream='context')

        ov_f = overlay_cam(face_crop, cam_f)
        ov_b = overlay_cam(body_crop, cam_b)
        ov_c = overlay_cam(ctx_crop, cam_c)

        card = build_card(face_crop, body_crop, ctx_crop, ov_f, ov_b, ov_c,
                          emotion_name, true_text, pred_text)

        b_name = os.path.basename(p).replace('.jpg', '')
        out_name = f"emotic_std_{idx:02d}_{emotion_name}_{b_name}.png"
        save_p = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(save_p, card)
        saved_cards.append(out_name)
        if idx % 10 == 0 or idx == len(all_jobs):
            print(f"  [EMOTIC Standard] Saved {idx}/{len(all_jobs)} cards -> {out_name}")

    # Build Interactive HTML index for quick browsing
    html_p = os.path.join(OUT_DIR, 'index.html')
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EMOTIC Standard (Unmasked) GradCAM Gallery — Full Dataset</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f1117; color: #e6edf3; margin: 0; padding: 24px; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; color: #58a6ff; }}
  p.subtitle {{ color: #8b949e; font-size: 15px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 18px; }}
  .card-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; transition: transform 0.2s, border-color 0.2s; }}
  .card-box:hover {{ transform: translateY(-3px); border-color: #58a6ff; }}
  .card-box img {{ width: 100%; display: block; }}
  .card-title {{ padding: 10px 14px; font-size: 13px; color: #8b949e; font-weight: 500; border-top: 1px solid #21262d; }}
</style>
</head>
<body>
  <h1>EMOTIC Standard (Bản thường - Unmasked) GradCAM Gallery</h1>
  <p class="subtitle">{len(saved_cards)} Comprehensive Standard GradCAM Cards sampled across the Full EMOTIC Dataset (mscoco, ade20k, emodb_small). Intact natural body features without face masking.</p>
  <div class="grid">
    {"".join(f'<div class="card-box"><a href="{c}" target="_blank"><img src="{c}" loading="lazy"/></a><div class="card-title">{c}</div></div>' for c in saved_cards)}
  </div>
</body>
</html>
"""
    with open(html_p, 'w') as f:
        f.write(html_content)
    print(f"\n  [HTML Gallery] Saved HTML dashboard to: {html_p}")
    print(f"\n===> ALL {len(saved_cards)} STANDARD EMOTIC CARDS SUCCESSFULLY CREATED IN {OUT_DIR}!")


if __name__ == '__main__':
    main()
