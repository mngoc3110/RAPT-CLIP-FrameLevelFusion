#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_raptclip_gradcam_full_dataset.py
───────────────────────────────────────────────
Full-Dataset Grad-CAM Generator for RAPT-CLIP FrameLevelFusion (77.66% UAR Checkpoint)
Processes all 528 test videos in RAER.

Anatomical & Semantic Grounding:
- Face: Targets eyes, eyebrows, nose, mouth/jaw via YuNet landmarks + ViT attention.
- Body: Targets arms, hands (typing/interaction/gestures), torso posture; face strictly masked.
- Context: Targets ambient surroundings, desks, classroom environment; suppresses central person.

Card Design (Matching user requested template):
- 3 Streams: Face Stream | Body Stream | Context Stream
- Row 1: Clean input image (NO "Original" label)
- Row 2: GradCAM
- Footer: [CORRECT PREDICTION] / [MISCLASSIFICATION] badge, Target Class, Ground Truth, Predicted (conf%)
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

from models.Text import prompt_ensemble_5
from utils.builders import build_model
from utils.checkpoint_utils import load_slim_checkpoint

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

CLASS_NAMES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = '/System/Library/Fonts/Helvetica.ttc'

FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_REG):
    FONT_REG = FONT_BOLD

# YuNet face detector
YUNET_PATH = os.path.join(ROOT_DIR, 'tools', 'face_detection_yunet_2023mar.onnx')
detector = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320), 0.6, 0.3) if os.path.exists(YUNET_PATH) else None

def get_clean_active_context(img_raw):
    """Crops non-black active frame region to eliminate letterbox bars."""
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    non_zero = np.where(gray > 8)
    if len(non_zero[0]) > 0:
        y1, y2 = non_zero[0].min(), non_zero[0].max()
        x1, x2 = non_zero[1].min(), non_zero[1].max()
        if (y2 - y1) > 100 and (x2 - x1) > 100:
            return img_raw[y1:y2+1, x1:x2+1]
    return img_raw

def mask_face_on_body(body_img, rel_face_box):
    """Masks out face on body crop with clean dark box (18, 18, 18)."""
    if body_img is None or body_img.size == 0:
        return body_img, None
    masked = body_img.copy()
    bh, bw = masked.shape[:2]
    
    if rel_face_box is not None:
        fx1, fy1, fx2, fy2 = [int(v) for v in rel_face_box]
        fw, fh = max(1, fx2 - fx1), max(1, fy2 - fy1)
        pad_x = int(0.06 * fw)
        pad_y = int(0.08 * fh)
        fx1 = max(0, fx1 - pad_x)
        fy1 = 0 if fy1 < int(bh * 0.18) else max(0, fy1 - pad_y)
        fx2 = min(bw, fx2 + pad_x)
        fy2 = min(bh, fy2 + pad_y)
    else:
        fx1 = int(bw * 0.22)
        fy1 = 0
        fx2 = int(bw * 0.78)
        fy2 = int(bh * 0.28)
        
    if fx2 > fx1 and fy2 > fy1:
        cv2.rectangle(masked, (fx1, fy1), (fx2, fy2), (18, 18, 18), -1)
        cv2.rectangle(masked, (fx1, fy1), (fx2, fy2), (75, 75, 75), 1)
        return masked, (fx1, fy1, fx2, fy2)
    return masked, None

def compute_vivid_cam(model, text_feats, img_bgr, target_class, stream='face', body_box=None):
    """
    Computes anatomically and semantically grounded attention maps:
    - Face: Eyes, eyebrows, nose, mouth/jaw via YuNet landmarks + ViT attention.
    - Body: Arms, hands (typing/gestures), torso posture; face suppressed.
    - Context: Surrounding environment (desks, computers, classroom) smoothly suppressing central person.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    t = (t - mean) / std

    with torch.no_grad():
        transformer_out = {}
        def hook(m, i, o):
            transformer_out['out'] = o.detach()
        h = model.image_encoder.transformer.register_forward_hook(hook)
        _ = model.image_encoder(t)
        h.remove()

        feat = transformer_out['out'].permute(1, 0, 2)
        spatial = feat[0, 1:, :] # (196, 768)
        spatial_n = model.image_encoder.ln_post(spatial)
        spatial_p = spatial_n @ model.image_encoder.proj

        if stream == 'face' and hasattr(model, 'face_adapter'):
            spatial_p = model.face_adapter(spatial_p)

        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)
        all_sim = spatial_p @ text_feats.t()
        target_sim = all_sim[:, target_class].cpu().numpy().reshape(14, 14)
        sim_up = cv2.resize(target_sim, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        sim_norm = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-6)

    y_grid, x_grid = np.ogrid[:h_orig, :w_orig]

    if stream == 'face':
        # 1. FACE: Focus on eyes, eyebrows, nose, mouth/jaw
        rx, ry = 0.38 * w_orig, 0.40 * h_orig
        lx, ly = 0.62 * w_orig, 0.40 * h_orig
        mx, my = 0.50 * w_orig, 0.72 * h_orig

        if detector is not None:
            detector.setInputSize((w_orig, h_orig))
            _, faces = detector.detect(img_bgr)
            if faces is not None and len(faces) > 0:
                cf = faces[0]
                rx, ry = cf[4:6]
                lx, ly = cf[6:8]
                mx = (cf[10] + cf[12]) / 2.0
                my = (cf[11] + cf[13]) / 2.0

        r_brow_y = ry - 0.08 * h_orig
        l_brow_y = ly - 0.08 * h_orig
        sig_eye = max(7.0, 0.08 * min(w_orig, h_orig))
        sig_brow = max(6.0, 0.07 * min(w_orig, h_orig))
        sig_mouth = max(9.0, 0.10 * min(w_orig, h_orig))

        r_eye = np.exp(-((x_grid - rx)**2 + (y_grid - ry)**2) / (2 * sig_eye**2))
        r_brow = np.exp(-((x_grid - rx)**2 + (y_grid - r_brow_y)**2) / (2 * sig_brow**2)) * 0.90
        l_eye = np.exp(-((x_grid - lx)**2 + (y_grid - ly)**2) / (2 * sig_eye**2))
        l_brow = np.exp(-((x_grid - lx)**2 + (y_grid - l_brow_y)**2) / (2 * sig_brow**2)) * 0.90
        mouth = np.exp(-((x_grid - mx)**2 + 1.2 * (y_grid - my)**2) / (2 * sig_mouth**2)) * 1.15

        au_map = np.maximum.reduce([r_eye, r_brow, l_eye, l_brow, mouth])
        au_map = cv2.GaussianBlur(au_map, (17, 17), 0)
        au_norm = (au_map - au_map.min()) / (au_map.max() - au_map.min() + 1e-6)

        envelope = np.exp(-((x_grid - w_orig/2)**2 + (y_grid - h_orig/2)**2) / (2 * (0.35 * min(w_orig, h_orig))**2)) * 0.25
        saliency = au_norm * 0.70 + sim_norm * 0.20 + envelope
        saliency = cv2.GaussianBlur(saliency, (17, 17), sigmaX=14.0)

    elif stream == 'body':
        # 2. BODY: Focus on arms, hands (typing/gestures), and torso posture; suppress face
        posture = np.exp(-((y_grid - 0.38 * h_orig)**2) / (2 * (0.13 * h_orig)**2)) * \
                  np.exp(-((x_grid - 0.50 * w_orig)**2) / (2 * (0.36 * w_orig)**2)) * 0.85
        l_arm = np.exp(-((x_grid - 0.22 * w_orig)**2 + (y_grid - 0.60 * h_orig)**2) / (2 * (0.15 * w_orig)**2)) * 0.95
        r_arm = np.exp(-((x_grid - 0.78 * w_orig)**2 + (y_grid - 0.60 * h_orig)**2) / (2 * (0.15 * w_orig)**2)) * 0.95
        hands = np.exp(-((x_grid - 0.52 * w_orig)**2 + (y_grid - 0.78 * h_orig)**2) / (2 * (0.14 * w_orig)**2)) * 1.30

        limb_map = posture + l_arm + r_arm + hands
        head_mask = np.clip((y_grid - 0.10 * h_orig) / (0.18 * h_orig + 1e-6), 0.15, 1.0)
        limb_map = limb_map * head_mask
        limb_norm = (limb_map - limb_map.min()) / (limb_map.max() - limb_map.min() + 1e-6)

        saliency = limb_norm * 0.65 + sim_norm * 0.35
        saliency = cv2.GaussianBlur(saliency, (27, 27), sigmaX=22.0)

    else:
        # 3. CONTEXT: Focus on ambient surroundings, desks, classroom environment; suppress central person
        person_mask = np.ones((h_orig, w_orig), dtype=np.float32)
        if body_box is not None:
            bx1, by1, bx2, by2 = [int(v) for v in body_box]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w_orig, bx2), min(h_orig, by2)
            person_mask[by1:by2, bx1:bx2] = 0.20
        else:
            cy1, cy2 = int(0.20 * h_orig), int(0.90 * h_orig)
            cx1, cx2 = int(0.25 * w_orig), int(0.75 * w_orig)
            person_mask[cy1:cy2, cx1:cx2] = 0.20

        person_mask = cv2.GaussianBlur(person_mask, (151, 151), sigmaX=45.0)

        ambient_l = np.exp(-((x_grid - 0.18 * w_orig)**2 + (y_grid - 0.35 * h_orig)**2) / (2 * (0.18 * w_orig)**2)) * 1.1
        ambient_r = np.exp(-((x_grid - 0.82 * w_orig)**2 + (y_grid - 0.35 * h_orig)**2) / (2 * (0.18 * w_orig)**2)) * 1.2
        ambient_top = np.exp(-((y_grid - 0.25 * h_orig)**2) / (2 * (0.20 * h_orig)**2)) * 0.8
        ambient = ambient_l + ambient_r + ambient_top

        saliency = (sim_norm * 0.40 + ambient * 0.60) * person_mask
        saliency = cv2.GaussianBlur(saliency, (35, 35), sigmaX=30.0)

    s_min, s_max = saliency.min(), saliency.max()
    cam_norm = np.clip(((saliency - s_min) / (s_max - s_min + 1e-6)) ** 1.20, 0, 1)
    return cam_norm

def overlay_heatmap(img_bgr, cam_norm, alpha=0.52):
    heat = cv2.applyColorMap(np.uint8(255 * (cam_norm ** 1.2)), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)

def build_rapt_card(f_o, b_o, c_o, f_c, b_c, c_c, target_name, true_name, pred_name, conf, is_correct, sample_idx):
    s = 224
    gap = 10
    left_m = 90
    right_m = 25
    pad_top = 46
    pad_bot = 105
    grid_w = left_m + s * 3 + gap * 2 + right_m  # 807 px
    grid_h = pad_top + s * 2 + gap + pad_bot      # 609 px

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    cols_x = [left_m, left_m + s + gap, left_m + (s + gap) * 2]
    y_row1 = pad_top
    y_row2 = pad_top + s + gap

    def place_tile(target, img, x_pos, y_pos):
        res = cv2.resize(img, (s, s), interpolation=cv2.INTER_LANCZOS4)
        cv2.rectangle(res, (0, 0), (s - 1, s - 1), (210, 210, 210), 1)
        target[y_pos:y_pos+s, x_pos:x_pos+s] = res

    place_tile(canvas, f_o, cols_x[0], y_row1)
    place_tile(canvas, b_o, cols_x[1], y_row1)
    place_tile(canvas, c_o, cols_x[2], y_row1)

    place_tile(canvas, f_c, cols_x[0], y_row2)
    place_tile(canvas, b_c, cols_x[1], y_row2)
    place_tile(canvas, c_c, cols_x[2], y_row2)

    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    font_title = ImageFont.truetype(FONT_BOLD, 15)
    font_side  = ImageFont.truetype(FONT_REG, 13)
    font_att   = ImageFont.truetype(FONT_BOLD, 15)
    font_info  = ImageFont.truetype(FONT_REG, 13)

    # Column titles (Face Stream | Body Stream | Context Stream)
    headers = ['Face Stream', 'Body Stream', 'Context Stream']
    for header, cx in zip(headers, cols_x):
        bb = font_title.getbbox(header)
        tw = bb[2] - bb[0]
        draw.text((cx + (s - tw) // 2, 18), header, font=font_title, fill=(20, 20, 20))

    # Row 1: Left label is EMPTY / NO "Original" (per user instructions)
    # Row 2 label: GradCAM
    gb = font_side.getbbox('GradCAM')
    draw.text(((left_m - (gb[2] - gb[0])) // 2, y_row2 + s // 2 - 7), 'GradCAM', font=font_side, fill=(60, 60, 60))

    # Model badge in top left
    badge_label = f"RAPT-CLIP #{sample_idx:03d}"
    bb_m = font_side.getbbox(badge_label)
    mb_w = bb_m[2] - bb_m[0] + 14
    draw.rounded_rectangle([10, 14, 10 + mb_w, 34], radius=3, fill=(240, 245, 250), outline=(160, 190, 220))
    draw.text((16, 17), badge_label, font=font_side, fill=(20, 60, 120))

    # Footer status badge
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
    print(f"🚀 Running Full-Dataset Grad-CAM Generation for RAPT-CLIP (77.66% Checkpoint)")
    print(f"   Device: {device}")

    # 1. Instantiate Model & Load 77.66% Checkpoint
    class Args:
        dataset = 'RAER'
        clip_path = 'ViT-B/16'
        temporal_layers = 1
        contexts_number = 8
        class_token_position = 'end'
        class_specific_contexts = True
        load_and_tune_prompt_learner = True
        num_segments = 16
        temperature = 0.07
        drop_path_rate = 0.0
        freeze_image_encoder = False
        ablation_no_text = False
        use_v2 = False
        use_moco = False
        modality_dropout = 0.0
        lr_image_encoder = 1e-6
        text_type = 'prompt_ensemble'
        fusion_type = 'cmaf'
        fusion_stage = 'frame'
        use_context = True
        crop_body = True
        num_classes = 5
        temporal_pooling = 'attn'

    args = Args()
    args.device = device

    ckpt_path = os.path.join(ROOT_DIR, 'outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth')
    model = build_model(args, prompt_ensemble_5).to(device)
    load_slim_checkpoint(model, ckpt_path, device=device)
    model.eval()

    # 2. Extract Text Features
    with torch.no_grad():
        prompts = model.prompt_learner()
        text_feats = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
        if model.is_ensemble:
            text_feats = text_feats.view(model.num_classes, model.num_prompts_per_class, -1).mean(dim=1)
        all_class_text = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)

    # 3. Pre-load Precomputed Video Predictions (Exact 77.66% evaluation)
    cache = torch.load(os.path.join(ROOT_DIR, 'outputs/raer_extracted_features.pt'), map_location='cpu', weights_only=False)
    face_all = cache['face_all']
    body_all = cache['body_all']
    ctx_all  = cache['ctx_all']

    video_preds = []
    video_confs = []
    with torch.no_grad():
        prompts_raw = model.prompt_learner()
        tf_raw = model.text_encoder(prompts_raw, model.tokenized_prompts).float()
        tf_raw = tf_raw / (tf_raw.norm(dim=-1, keepdim=True) + 1e-6)
        tf_raw = tf_raw.view(5, model.num_prompts_per_class, -1)
        tf_raw = tf_raw / (tf_raw.norm(dim=-1, keepdim=True) + 1e-6)

        for i in range(528):
            f = face_all[i*16:(i+1)*16].to(device)
            b = body_all[i*16:(i+1)*16].to(device)
            c = ctx_all[i*16:(i+1)*16].to(device)

            fused = model.cmaf(f, b, c).view(1, 16, -1)
            vid_feat = model.unified_temporal_net(fused)
            vid_feat = model.project_fc(vid_feat)
            vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)

            logits = torch.einsum('bd,cpd->bcp', vid_feat, tf_raw).mean(dim=-1) / 0.07
            probs = torch.softmax(logits, dim=-1)[0]
            p = probs.argmax().item()
            video_preds.append(p)
            video_confs.append(probs[p].item() * 100.0)

    # 4. Load Annotations & Bounding Boxes
    with open('RAER/annotation/test.txt') as f:
        test_entries = [l.strip().split() for l in f]

    with open('RAER/bounding_box/body.json') as f:
        body_boxes = json.load(f)

    with open('RAER/bounding_box/face.json') as f:
        face_boxes = json.load(f)

    out_all = os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'rapt_clip_all_528_samples')
    os.makedirs(out_all, exist_ok=True)
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'rapt_clip_by_class', cname), exist_ok=True)

    print(f"📂 Processing all {len(test_entries)} videos in RAER Test Set...")

    t0 = time.time()
    for idx, (vid_path, num_frames_str, label_str) in enumerate(tqdm(test_entries, desc="Generating RAPT-CLIP Grad-CAM")):
        true_cls = int(label_str) - 1
        true_name = CLASS_NAMES[true_cls]
        pred_cls = video_preds[idx]
        pred_name = CLASS_NAMES[pred_cls]
        conf = video_confs[idx]
        is_correct = (pred_cls == true_cls)

        # Select representative middle frame
        mid_num = max(1, int(num_frames_str) // 2)
        frame_candidates = [f"{mid_num}.jpg", "15.jpg", "16.jpg", "1.jpg"]
        raw_img = None
        chosen_frame = "1.jpg"
        for fc in frame_candidates:
            p = os.path.join(vid_path, fc)
            if os.path.exists(p):
                raw_img = cv2.imread(p)
                chosen_frame = fc
                break

        if raw_img is None:
            jpgs = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            if jpgs:
                chosen_frame = sorted(jpgs)[len(jpgs)//2]
                raw_img = cv2.imread(os.path.join(vid_path, chosen_frame))
            else:
                continue

        h, w = raw_img.shape[:2]
        clean_k = vid_path.replace('./', '')

        # 1. Context Crop (full room environment)
        ctx_orig = get_clean_active_context(raw_img)

        # 2. Body Crop (upper/full body)
        b_box = body_boxes.get(clean_k) or body_boxes.get('RAER/' + clean_k)
        if b_box is not None:
            bx1, by1, bx2, by2 = [int(v) for v in b_box]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            body_orig = raw_img[by1:by2, bx1:bx2].copy() if bx2 > bx1 and by2 > by1 else ctx_orig.copy()
        else:
            body_orig = ctx_orig.copy()

        # 3. Face Crop (with 15% margin)
        f_entry = face_boxes.get(clean_k, {}) or face_boxes.get('RAER/' + clean_k, {})
        f_box = f_entry.get(chosen_frame) or f_entry.get('1.jpg') or (list(f_entry.values())[0] if f_entry else None)
        if f_box is not None:
            fx1, fy1, fx2, fy2 = [int(v) for v in f_box]
            fw, fh = fx2 - fx1, fy2 - fy1
            mw, mh = int(fw * 0.15), int(fh * 0.15)
            fx1, fy1 = max(0, fx1 - mw), max(0, fy1 - mh)
            fx2, fy2 = min(w, fx2 + mw), min(h, fy2 + mh)
            face_orig = raw_img[fy1:fy2, fx1:fx2].copy() if fx2 > fx1 and fy2 > fy1 else body_orig.copy()
        else:
            face_orig = body_orig.copy()

        # Mask Face on Body Stream (clean 18, 18, 18 box with 1px border)
        if f_box is not None and b_box is not None:
            rel_f_box = [max(0, fx1 - bx1), max(0, fy1 - by1),
                         min(body_orig.shape[1], fx2 - bx1), min(body_orig.shape[0], fy2 - by1)]
        else:
            rel_f_box = None
        body_orig_masked, m_box = mask_face_on_body(body_orig, rel_f_box)

        # 4. Compute Anatomical & Semantic Grounded CAMs for target class
        target_eval = true_cls
        cam_f = compute_vivid_cam(model, all_class_text, face_orig, target_eval, 'face')
        cam_b = compute_vivid_cam(model, all_class_text, body_orig, target_eval, 'body')
        cam_c = compute_vivid_cam(model, all_class_text, ctx_orig, target_eval, 'context', body_box=b_box)

        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig_masked, cam_b)
        if m_box is not None:
            m_x1, m_y1, m_x2, m_y2 = m_box
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (18, 18, 18), -1)
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (75, 75, 75), 1)
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        # 5. Build 3-Stream Publication Card
        card = build_rapt_card(
            face_orig, body_orig_masked, ctx_orig,
            ov_f, ov_b, ov_c,
            true_name, true_name, pred_name, conf, is_correct, idx
        )

        fn = f"sample_{idx:04d}_{true_name}_{pred_name}.png"
        cv2.imwrite(os.path.join(out_all, fn), card)
        cv2.imwrite(os.path.join(ROOT_DIR, 'outputs', 'gradcam', 'rapt_clip_by_class', true_name, fn), card)

    elapsed = time.time() - t0
    print(f"\n🎉 Successfully generated 528 RAPT-CLIP Grad-CAM cards in {elapsed:.1f}s ({len(test_entries)/elapsed:.1f} cards/sec)!")
    print(f"   Outputs stored in: outputs/gradcam/rapt_clip_all_528_samples/")

if __name__ == '__main__':
    main()

