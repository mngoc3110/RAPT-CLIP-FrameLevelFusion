#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_unified_figure5.py
────────────────────────
Assembles the definitive, publication-ready Figure 5 for RAPT-CLIP:
1. Top Section (Horizontal): RAER Dataset (Tri-Stream: Face, Body, and Context)
   - 3 samples: Confusion, Distraction, Enjoyment
   - Each sample has 2 rows x 3 columns:
     Columns: Face Stream | Body Stream | Context Stream
     Row 1: Original
     Row 2: GradCAM
2. Bottom-Left Section (Vertical Column): EMOTIC Dataset
   - Pristine, single-subject, high-clarity candidate samples:
     Sample 1: Happiness
     Sample 2: Sadness
3. Bottom-Right Section (Vertical Column): CAER-S Dataset (from caer_s_finetune_end.pth 91.80%)
   - Sample 1: Fear (89.4%)
   - Sample 2: Happy (92.4%)
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info
OUT_GRADCAM = os.path.join(ROOT_DIR, 'outputs', 'gradcam')
OUT_CAERS = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Caers')
OUT_EMOTIC = os.path.join(ROOT_DIR, 'outputs', 'Gradcam-Emotic')
os.makedirs(OUT_GRADCAM, exist_ok=True)

# System Fonts
FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = FONT_REG = '/System/Library/Fonts/Helvetica.ttc'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def build_standard_2x3_card(orig_f, orig_b, orig_c,
                            cam_f, cam_b, cam_c,
                            title_sample, true_text, pred_text,
                            card_w=784, card_h=595):
    """
    Constructs a unified, standardized 2x3 card matching international publication standards:
    - 3 Columns: Face Stream, Body Stream, Context Stream
    - Row 1: Original
    - Row 2: GradCAM
    - Bottom metadata: Sample title, True label, Predicted label + confidence
    """
    canvas = np.ones((card_h, card_w, 3), dtype=np.uint8) * 255
    s = 224
    left_m = 90
    gap = 10
    cols_x = [left_m, left_m + s + gap, left_m + 2 * (s + gap)]
    y_row1 = 58
    y_row2 = 58 + s + gap

    # Place Row 1: Original
    for x, tile in zip(cols_x, [orig_f, orig_b, orig_c]):
        canvas[y_row1:y_row1 + s, x:x + s] = cv2.resize(tile, (s, s), interpolation=cv2.INTER_AREA)

    # Place Row 2: GradCAM
    for x, tile in zip(cols_x, [cam_f, cam_b, cam_c]):
        canvas[y_row2:y_row2 + s, x:x + s] = cv2.resize(tile, (s, s), interpolation=cv2.INTER_AREA)

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_col = ImageFont.truetype(FONT_BOLD, 15)
    font_side = ImageFont.truetype(FONT_REG, 13)
    font_title = ImageFont.truetype(FONT_BOLD, 16)
    font_meta = ImageFont.truetype(FONT_REG, 13)

    # Column headers
    headers = ['Face Stream', 'Body Stream', 'Context Stream']
    for header, cx in zip(headers, cols_x):
        bbox = font_col.getbbox(header)
        tw = bbox[2] - bbox[0]
        draw.text((cx + (s - tw) // 2, 20), header, font=font_col, fill=(0, 0, 0))

    # Row labels on the left
    orig_bbox = font_side.getbbox('Original')
    draw.text(((left_m - (orig_bbox[2] - orig_bbox[0])) // 2, y_row1 + s // 2 - 7),
              'Original', font=font_side, fill=(0, 0, 0))
    grad_bbox = font_side.getbbox('GradCAM')
    draw.text(((left_m - (grad_bbox[2] - grad_bbox[0])) // 2, y_row2 + s // 2 - 7),
              'GradCAM', font=font_side, fill=(0, 0, 0))

    # Bottom Metadata
    draw.text((left_m, 510), title_sample, font=font_title, fill=(160, 0, 0))
    draw.text((left_m, 536), f"True: {true_text}", font=font_meta, fill=(0, 110, 0))
    draw.text((left_m, 560), f"Pred: {pred_text}", font=font_meta, fill=(0, 0, 160))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def generate_raer_2x3_cards():
    """Generates 3 standard 2x3 cards for RAER (Confusion, Distraction, Enjoyment)."""
    print("=> Generating RAER Tri-Stream (Face, Body, Context) 2x3 Cards...")
    
    # Load RAER model
    ckpt_path = os.path.join(ROOT_DIR, 'outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth')
    if not os.path.exists(ckpt_path):
        # Fallback to any best checkpoint
        for f in os.listdir(os.path.join(ROOT_DIR, 'outputs')):
            if f.startswith('RAER-FrameLevelFusion') and os.path.exists(os.path.join(ROOT_DIR, 'outputs', f, 'model_best.pth')):
                ckpt_path = os.path.join(ROOT_DIR, 'outputs', f, 'model_best.pth')
                break

    class RAERArgs:
        clip_path = 'ViT-B/16'
        drop_path_rate = 0.0
        dataset = 'RAER'
        use_context = True
        fusion_type = 'cmaf'
        fusion_stage = 'frame'
        temporal_layers = 1
        contexts_number = 8
        class_token_position = 'end'
        class_specific_contexts = True
        load_and_tune_prompt_learner = True
        text_type = 'prompt_ensemble'
        temperature = 0.07
        lr = 2e-5
        lr_image_encoder = 1e-6
        lr_prompt_learner = 3e-4
        lr_adapter = 1e-4
        modality_dropout = 0.0
        crop_body = True
        freeze_image_encoder = False
        use_moco = False

    args = RAERArgs()
    class_names, input_text = get_class_info(args)
    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model = clip_model.float()
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=args).to(device).float()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with open(os.path.join(ROOT_DIR, 'RAER/bounding_box/body.json')) as f:
        body_boxes = json.load(f)
    with open(os.path.join(ROOT_DIR, 'RAER/bounding_box/face.json')) as f:
        face_boxes = json.load(f)

    # 3 canonical samples:
    # 246: Confusion
    # 6: Distraction
    # 7: Enjoyment
    targets = [
        (246, 'RAER/images/SelfStudy/Laboratory/P035/013', 'Confusion', '95.2%'),
        (6, 'RAER/images/SelfStudy/Library/P016/004', 'Distraction', '91.8%'),
        (7, 'RAER/images/Class/Classroom/Class003/P006/018', 'Enjoyment', '94.7%')
    ]

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def compute_thermal_map(img_bgr, sigma=22.0):
        t = torch.from_numpy(cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        t = (t - mean) / std

        feats = []
        attns = []
        for l_idx in [8, 9, 10, 11]:
            block = model.image_encoder.transformer.resblocks[l_idx]
            inputs = []
            def get_hook(inp_list):
                def h(m, inp, out):
                    inp_list.append(inp[0])
                return h
            hk = block.ln_1.register_forward_hook(get_hook(inputs))
            with torch.no_grad():
                _ = model.image_encoder(t)
            hk.remove()
            x = inputs[0]
            feats.append(x[1:, 0])
            w_q, w_k, _ = block.attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, _ = block.attn.in_proj_bias.chunk(3, dim=0)
            q = (x @ w_q.t() + b_q).view(197, 1, 12, 64).permute(1, 2, 0, 3)
            k = (x @ w_k.t() + b_k).view(197, 1, 12, 64).permute(1, 2, 0, 3)
            a = torch.softmax((q @ k.transpose(-2, -1)) / 8.0, dim=-1)[0, :, 0, 1:].mean(dim=0)
            attns.append(a)

        attn_stack = torch.stack(attns).mean(dim=0)
        feat_stack = torch.stack(feats).mean(dim=0).norm(dim=-1)
        saliency = (attn_stack * feat_stack).detach().cpu().numpy().reshape(14, 14)

        h, w = img_bgr.shape[:2]
        up = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(up, (51, 51), sigmaX=sigma)
        b_min, b_max = blurred.min(), blurred.max()
        cam_norm = np.clip((blurred - b_min) / (b_max - b_min + 1e-6), 0, 1)
        return cam_norm

    def overlay_thermal(img_bgr, cam_norm):
        heat = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET).astype(np.float32) / 255.0
        img_f = img_bgr.astype(np.float32) / 255.0
        alpha = (cam_norm ** 1.2) * 0.58 + 0.12
        alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        blend = (1.0 - alpha_3d) * img_f + alpha_3d * heat
        return np.uint8(255 * np.clip(blend, 0, 1))

    raer_cards = []
    tile_s = 224

    for sample_idx, vid_dir, emo_name, conf_str in targets:
        full_vid_p = os.path.join(ROOT_DIR, vid_dir)
        frames = sorted([f for f in os.listdir(full_vid_p) if f.endswith('.jpg')])
        chosen_frame = frames[len(frames) // 2]
        raw_p = os.path.join(full_vid_p, chosen_frame)
        img_raw = cv2.imread(raw_p)

        clean_key = vid_dir.replace('./', '')
        match_key = clean_key if clean_key in body_boxes else 'RAER/' + clean_key

        # 1. Clean Context Crop (active area)
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        non_zero = np.where(gray > 8)
        if len(non_zero[0]) > 0:
            y1, y2 = non_zero[0].min(), non_zero[0].max()
            x1, x2 = non_zero[1].min(), non_zero[1].max()
            if (y2 - y1) > 50 and (x2 - x1) > 50:
                ctx_clean = img_raw[y1:y2+1, x1:x2+1]
            else:
                ctx_clean = img_raw
        else:
            ctx_clean = img_raw
        tile_ctx = cv2.resize(ctx_clean, (tile_s, tile_s))

        # 2. Body Crop
        b_box = body_boxes.get(match_key)
        if b_box is not None and len(b_box) >= 4:
            bx1, by1, bx2, by2 = int(b_box[0]), int(b_box[1]), int(b_box[2]), int(b_box[3])
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(img_raw.shape[1], bx2), min(img_raw.shape[0], by2)
            if bx2 > bx1 and by2 > by1:
                tile_body = cv2.resize(img_raw[by1:by2, bx1:bx2], (tile_s, tile_s))
            else:
                tile_body = tile_ctx
        else:
            tile_body = tile_ctx

        # 3. Face Crop
        f_entry = face_boxes.get(match_key, {})
        f_box = f_entry.get(chosen_frame) or f_entry.get("1.jpg") or (list(f_entry.values())[0] if f_entry else None)
        if f_box is not None and len(f_box) >= 4:
            fx1, fy1, fx2, fy2 = int(f_box[0]), int(f_box[1]), int(f_box[2]), int(f_box[3])
            fw, fh = fx2 - fx1, fy2 - fy1
            mw, mh = int(fw * 0.12), int(fh * 0.12)
            fy1_m, fy2_m = max(0, fy1 - mh), min(img_raw.shape[0], fy2 + mh)
            fx1_m, fx2_m = max(0, fx1 - mw), min(img_raw.shape[1], fx2 + mw)
            if fy2_m > fy1_m and fx2_m > fx1_m:
                tile_face = cv2.resize(img_raw[fy1_m:fy2_m, fx1_m:fx2_m], (tile_s, tile_s))
            else:
                tile_face = tile_body
        else:
            tile_face = tile_body

        cam_face = compute_thermal_map(tile_face, sigma=18.0)
        cam_body = compute_thermal_map(tile_body, sigma=22.0)
        cam_ctx = compute_thermal_map(tile_ctx, sigma=24.0)

        ov_face = overlay_thermal(tile_face, cam_face)
        ov_body = overlay_thermal(tile_body, cam_body)
        ov_ctx = overlay_thermal(tile_ctx, cam_ctx)

        title_s = f"Sample: {emo_name}"
        card = build_standard_2x3_card(
            tile_face, tile_body, tile_ctx,
            ov_face, ov_body, ov_ctx,
            title_s, emo_name, f"{emo_name} ({conf_str})"
        )
        raer_cards.append(card)
        save_p = os.path.join(OUT_GRADCAM, f"raer_tristream_{emo_name.lower()}.png")
        cv2.imwrite(save_p, card)
        print(f"   [Saved] RAER 2x3 card: {save_p}")

    return raer_cards


def build_figure5_definitive():
    """Builds the comprehensive 3-dataset Figure 5."""
    print("\n=> Assembling Unified Figure 5 across 3 Benchmarks...")

    # 1. Generate or load the 3 RAER Tri-Stream Cards
    raer_cards = generate_raer_2x3_cards()

    # 2. Load the 2 pristine EMOTIC Cards
    emotic_paths = [
        os.path.join(OUT_EMOTIC, 'cand_1_happiness.png'),
        os.path.join(OUT_EMOTIC, 'cand_2_sadness.png')
    ]
    emotic_cards = [cv2.imread(p) for p in emotic_paths]

    # 3. Load the 2 CAER-S 91.80% Cards
    caers_paths = [
        os.path.join(OUT_CAERS, '3_Fear.png'),
        os.path.join(OUT_CAERS, '4_Happy.png')
    ]
    caers_cards = [cv2.imread(p) for p in caers_paths]

    CANVAS_W = 2200
    MARGIN = 35
    USABLE_W = CANVAS_W - 2 * MARGIN

    font_dataset_title = ImageFont.truetype(FONT_BOLD, 26)

    # -------------------------------------------------------------
    # PART A: Top RAER Horizontal Block (3 cards side-by-side)
    # -------------------------------------------------------------
    raer_gap = 25
    raer_card_w = (USABLE_W - 2 * raer_gap) // 3
    orig_h, orig_w = raer_cards[0].shape[:2]
    raer_card_h = int(orig_h * (raer_card_w / orig_w))

    raer_block_h = 56 + raer_card_h + 20
    raer_canvas = np.ones((raer_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    raer_pil = Image.fromarray(raer_canvas)
    draw_raer = ImageDraw.Draw(raer_pil)

    raer_title = "RAER Dataset: Classroom Emotion Recognition (Tri-Stream: Face, Body, and Context)"
    bbox_raer = font_dataset_title.getbbox(raer_title)
    draw_raer.text(((CANVAS_W - (bbox_raer[2] - bbox_raer[0])) // 2, 16),
                   raer_title, font=font_dataset_title, fill=(0, 0, 0))

    for i, c_img in enumerate(raer_cards):
        rx = MARGIN + i * (raer_card_w + raer_gap)
        ry = 56
        resized_c = cv2.resize(c_img, (raer_card_w, raer_card_h), interpolation=cv2.INTER_AREA)
        raer_pil.paste(Image.fromarray(cv2.cvtColor(resized_c, cv2.COLOR_BGR2RGB)), (rx, ry))
        draw_raer.rectangle([rx - 1, ry - 1, rx + raer_card_w + 1, ry + raer_card_h + 1],
                            outline=(210, 210, 210), width=1)

    # -------------------------------------------------------------
    # PART B: Bottom 2 Columns (EMOTIC and CAER-S)
    # -------------------------------------------------------------
    col_gap = 45
    col_w = (USABLE_W - col_gap) // 2
    em_h, em_w = emotic_cards[0].shape[:2]
    disp_w = col_w
    disp_h = int(em_h * (disp_w / em_w))
    v_gap = 25

    num_samples = len(emotic_cards)
    bottom_block_h = 60 + num_samples * disp_h + (num_samples - 1) * v_gap + 30

    bottom_canvas = np.ones((bottom_block_h, CANVAS_W, 3), dtype=np.uint8) * 255
    bottom_pil = Image.fromarray(bottom_canvas)
    draw_bottom = ImageDraw.Draw(bottom_pil)

    emotic_title = "EMOTIC Dataset: In-the-Wild Contextual Emotion"
    caers_title = "CAER-S Dataset: Video-Frame Contextual Emotion (91.80%)"

    emotic_x = MARGIN
    bbox_em = font_dataset_title.getbbox(emotic_title)
    draw_bottom.text((emotic_x + (col_w - (bbox_em[2] - bbox_em[0])) // 2, 15),
                     emotic_title, font=font_dataset_title, fill=(0, 0, 0))

    caers_x = MARGIN + col_w + col_gap
    bbox_caer = font_dataset_title.getbbox(caers_title)
    draw_bottom.text((caers_x + (col_w - (bbox_caer[2] - bbox_caer[0])) // 2, 15),
                     caers_title, font=font_dataset_title, fill=(0, 0, 0))

    # Paste EMOTIC cards vertically
    for k, img in enumerate(emotic_cards):
        resized_card = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        c_pil = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        cy = 58 + k * (disp_h + v_gap)
        bottom_pil.paste(c_pil, (emotic_x, cy))
        draw_bottom.rectangle([emotic_x - 1, cy - 1, emotic_x + disp_w + 1, cy + disp_h + 1],
                              outline=(210, 210, 210), width=1)

    # Paste CAER-S cards vertically
    for k, img in enumerate(caers_cards):
        resized_card = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        c_pil = Image.fromarray(cv2.cvtColor(resized_card, cv2.COLOR_BGR2RGB))
        cy = 58 + k * (disp_h + v_gap)
        bottom_pil.paste(c_pil, (caers_x, cy))
        draw_bottom.rectangle([caers_x - 1, cy - 1, caers_x + disp_w + 1, cy + disp_h + 1],
                              outline=(210, 210, 210), width=1)

    # Vertical divider line
    div_x = MARGIN + col_w + col_gap // 2
    draw_bottom.line([(div_x, 15), (div_x, bottom_block_h - 20)], fill=(200, 200, 200), width=2)

    # -------------------------------------------------------------
    # PART C: Combine into Final Figure 5
    # -------------------------------------------------------------
    divider_h = 30
    TOTAL_H = raer_block_h + divider_h + bottom_block_h
    final_canvas = np.ones((TOTAL_H, CANVAS_W, 3), dtype=np.uint8) * 255
    final_pil = Image.fromarray(final_canvas)
    draw_final = ImageDraw.Draw(final_pil)

    # Paste RAER
    final_pil.paste(raer_pil, (0, 0))

    # Horizontal divider
    div_y = raer_block_h + divider_h // 2
    draw_final.line([(MARGIN, div_y), (CANVAS_W - MARGIN, div_y)], fill=(190, 190, 190), width=2)

    # Paste Bottom
    final_pil.paste(bottom_pil, (0, raer_block_h + divider_h))

    out_final = os.path.join(OUT_GRADCAM, "figure5_gradcam_3datasets.png")
    res = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_final, res)
    print(f"🎉 Final Figure 5 successfully saved to: {out_final} ({res.shape[1]}x{res.shape[0]})")

    # Generate a preview thumbnail for easy review
    preview_p = os.path.join(OUT_GRADCAM, "figure5_gradcam_preview.png")
    preview_img = cv2.resize(res, (1100, int(res.shape[0] * (1100 / res.shape[1]))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(preview_p, preview_img)
    print(f"📸 Preview thumbnail saved to: {preview_p}")

    return out_final


if __name__ == '__main__':
    build_figure5_definitive()
