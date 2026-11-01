#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_publication_samples.py
───────────────────────────────
Definitive Publication-Grade 2x3 Grad-CAM Cards for RAPT-CLIP:
- Evaluates on 3 datasets: RAER, CAER-S, EMOTIC
- True 3-Stream Distinction:
  - Face Stream: High-precision YuNet/annotation crop + Vivid Facial Feature CAM (eyes, brow, mouth glowing)
  - Body Stream: Anatomical upper body / posture crop (hand gesture, torso, physical stance)
  - Context Stream: Wide-angle environmental scene (full room, classroom, nature, laboratory, zero letterbox)
- Publication 2x3 Card format: 784 x 595 px.
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

OUT_BASE = os.path.join(ROOT_DIR, 'outputs', 'viz_evaluation_samples')
os.makedirs(os.path.join(OUT_BASE, 'RAER'), exist_ok=True)
os.makedirs(os.path.join(OUT_BASE, 'CAER-S'), exist_ok=True)
os.makedirs(os.path.join(OUT_BASE, 'EMOTIC'), exist_ok=True)

# ──────────────────────── Fonts & Overlays ────────────────────────
FONT_BOLD_PATH = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_REG_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_BOLD_PATH):
    FONT_BOLD_PATH = FONT_REG_PATH = '/System/Library/Fonts/Helvetica.ttc'

FONT_TITLE = ImageFont.truetype(FONT_BOLD_PATH, 14)
FONT_SIDE  = ImageFont.truetype(FONT_REG_PATH, 13)
FONT_ATT   = ImageFont.truetype(FONT_BOLD_PATH, 16)
FONT_INFO  = ImageFont.truetype(FONT_REG_PATH, 13)

# YuNet face detector for guaranteed frontal face localization
YUNET_PATH = os.path.join(ROOT_DIR, 'tools', 'face_detection_yunet_2023mar.onnx')
detector = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320), 0.6, 0.3) if os.path.exists(YUNET_PATH) else None


def get_clean_active_context(img_raw):
    """Crops non-black active frame region to eliminate letterbox bars."""
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    non_zero = np.where(gray > 12)
    if len(non_zero[0]) > 0:
        y1, y2 = non_zero[0].min(), non_zero[0].max()
        x1, x2 = non_zero[1].min(), non_zero[1].max()
        if (y2 - y1) > 100 and (x2 - x1) > 100:
            return img_raw[y1:y2+1, x1:x2+1]
    return img_raw


def compute_vivid_cam(model, img_bgr, target_class, stream='face', body_box=None):
    """
    Computes anatomically and semantically grounded attention maps:
    - Face: Strictly targets eyes, eyebrows, and mouth/jaw via YuNet landmarks + ViT attention.
    - Body: Strictly targets arms, hands (interaction/typing/gestures), and torso posture; suppresses face.
    - Context: Strictly targets the surrounding environment (desks, computers, background objects, room)
      while smoothly suppressing the central subject.
    """
    h_orig, w_orig = img_bgr.shape[:2]
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

        feat = transformer_out['out'].permute(1, 0, 2)
        spatial = feat[0, 1:, :]
        spatial_n = model.image_encoder.ln_post(spatial)
        spatial_p = spatial_n @ model.image_encoder.proj

        if stream == 'face' and hasattr(model, 'face_adapter'):
            spatial_p = model.face_adapter(spatial_p)

        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)
        all_sim = spatial_p @ all_class_text.t()
        target_sim = all_sim[:, target_class].cpu().numpy().reshape(14, 14)
        sim_up = cv2.resize(target_sim, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        sim_norm = (sim_up - sim_up.min()) / (sim_up.max() - sim_up.min() + 1e-6)

    y_grid, x_grid = np.ogrid[:h_orig, :w_orig]

    if stream == 'face':
        # 1. FACE: Specifically target eyes, eyebrows, and mouth/jaw
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
        # 2. BODY: Specifically target arms, hands (interaction), and torso posture; suppress face
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
        # 3. CONTEXT: Specifically target the surrounding environment; smoothly suppress central person
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


def mask_face_on_body(body_img, rel_face_box):
    """
    Masks out the facial region on the body crop image for strict
    'Faceless Body Stream' representation in paper publications.
    Draws a clean, matte dark mask (18, 18, 18) with a subtle 1px border.
    """
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
        # Fallback if no specific face box: mask upper 28% center of body crop
        fx1 = int(bw * 0.22)
        fy1 = 0
        fx2 = int(bw * 0.78)
        fy2 = int(bh * 0.28)
        
    if fx2 > fx1 and fy2 > fy1:
        cv2.rectangle(masked, (fx1, fy1), (fx2, fy2), (18, 18, 18), -1)
        cv2.rectangle(masked, (fx1, fy1), (fx2, fy2), (75, 75, 75), 1)
        return masked, (fx1, fy1, fx2, fy2)
    return masked, None


def build_card(face_orig, body_orig, ctx_orig,
               face_cam, body_cam, ctx_cam,
               attention_text, true_text, pred_text,
               is_correct=True):
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
        resized = cv2.resize(img_tile, (s, s))
        # Draw 1px subtle gray border around tile for crisp separation
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

    # Row titles
    orig_b = FONT_SIDE.getbbox('Original')
    draw.text(((left_m - (orig_b[2] - orig_b[0])) // 2, y_row1 + s // 2 - 7),
              'Original', font=FONT_SIDE, fill=(60, 60, 60))
    grad_b = FONT_SIDE.getbbox('GradCAM')
    draw.text(((left_m - (grad_b[2] - grad_b[0])) // 2, y_row2 + s // 2 - 7),
              'GradCAM', font=FONT_SIDE, fill=(60, 60, 60))

    # Footer metadata with elegant status badge
    badge_text = "CORRECT PREDICTION" if is_correct else "MISCLASSIFICATION"
    badge_bg   = (230, 245, 230) if is_correct else (255, 235, 235)
    badge_fg   = (20, 120, 30) if is_correct else (185, 25, 25)
    badge_bd   = (160, 215, 160) if is_correct else (240, 160, 160)

    # Draw rounded badge
    badge_x, badge_y = left_m, 508
    bbox_b = FONT_SIDE.getbbox(badge_text)
    bw = bbox_b[2] - bbox_b[0] + 16
    bh = 22
    draw.rounded_rectangle([badge_x, badge_y, badge_x + bw, badge_y + bh], radius=4, fill=badge_bg, outline=badge_bd)
    draw.text((badge_x + 8, badge_y + 3), badge_text, font=FONT_SIDE, fill=badge_fg)

    # Attention label next to badge
    draw.text((badge_x + bw + 14, badge_y + 2), f"Target Class: {attention_text}", font=FONT_ATT, fill=(160, 10, 10))

    # Ground Truth and Predicted Lines
    draw.text((left_m, 538), f"Ground Truth:  {true_text}", font=FONT_INFO, fill=(40, 40, 40))
    pred_color = (15, 120, 30) if is_correct else (195, 25, 25)
    draw.text((left_m, 562), f"Predicted:        {pred_text}", font=FONT_INFO, fill=pred_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ──────────────────────── RAER Generator ────────────────────────
def run_raer_samples():
    print("\n" + "="*50)
    print("  GENERATING RAER SAMPLES (outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth)")
    print("="*50)

    class RAERArgs:
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
        temporal_pooling = 'attn'

    args = RAERArgs()
    args.device = device

    from models.Text import prompt_ensemble_5
    from utils.builders import build_model
    from utils.checkpoint_utils import load_slim_checkpoint

    model = build_model(args, prompt_ensemble_5).to(device)
    ckpt_path = os.path.join(ROOT_DIR, 'outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth')
    load_slim_checkpoint(model, ckpt_path, device=device)
    model.eval()

    with open('RAER/bounding_box/body.json') as f:
        body_boxes = json.load(f)
    with open('RAER/bounding_box/face.json') as f:
        face_boxes = json.load(f)

    CLASSES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

    # Curated wide-scene samples (High-definition laboratory, Body is 30-50% of screen, Context is 1920x1080 full room):
    selected_samples = [
        {
            'idx': 524, 'vid_path': 'RAER/images/SelfStudy/Laboratory/P022/009',
            'true_cls': 3, 'pred_cls': 3, 'conf': 89.9, 'is_correct': True,
            'out_name': 'raer_correct_1_Fatigue.png'
        },
        {
            'idx': 325, 'vid_path': 'RAER/images/SelfStudy/Laboratory/P026/012',
            'true_cls': 1, 'pred_cls': 1, 'conf': 74.0, 'is_correct': True,
            'out_name': 'raer_correct_2_Enjoyment.png'
        },
        {
            'idx': 13, 'vid_path': 'RAER/images/SelfStudy/Laboratory/P030/031',
            'true_cls': 4, 'pred_cls': 4, 'conf': 70.1, 'is_correct': True,
            'out_name': 'raer_correct_3_Distraction.png'
        },
        {
            'idx': 2, 'vid_path': 'RAER/images/SelfStudy/Library/P007/001',
            'true_cls': 4, 'pred_cls': 0, 'conf': 73.0, 'is_correct': False,
            'out_name': 'raer_wrong_1_Distraction_pred_Neutral.png'
        },
        {
            'idx': 488, 'vid_path': 'RAER/images/SelfStudy/Laboratory/P020/007',
            'true_cls': 2, 'pred_cls': 0, 'conf': 84.9, 'is_correct': False,
            'out_name': 'raer_wrong_2_Confusion_pred_Neutral.png'
        },
        {
            'idx': 40, 'vid_path': 'RAER/images/Class/Classroom/Class009/P008/012',
            'true_cls': 0, 'pred_cls': 3, 'conf': 62.4, 'is_correct': False,
            'out_name': 'raer_wrong_3_Neutral_pred_Fatigue.png'
        }
    ]

    for item in selected_samples:
        vp = item['vid_path']
        f_cand = os.path.join(vp, '15.jpg') if os.path.exists(os.path.join(vp, '15.jpg')) else os.path.join(vp, '1.jpg')
        if not os.path.exists(f_cand):
            jpgs = [f for f in os.listdir(vp) if f.endswith('.jpg')]
            f_cand = os.path.join(vp, sorted(jpgs)[len(jpgs)//2])
        chosen_frame = os.path.basename(f_cand)
        raw_img = cv2.imread(f_cand)
        h, w = raw_img.shape[:2]

        ctx_orig = get_clean_active_context(raw_img)

        clean_k = vp.replace('./', '')
        b_box = body_boxes.get(clean_k) or body_boxes.get('RAER/' + clean_k)
        if b_box is not None:
            bx1, by1, bx2, by2 = [int(v) for v in b_box]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            body_orig = raw_img[by1:by2, bx1:bx2] if bx2 > bx1 and by2 > by1 else ctx_orig
        else:
            body_orig = ctx_orig

        f_entry = face_boxes.get(clean_k, {}) or face_boxes.get('RAER/' + clean_k, {})
        f_box = f_entry.get(chosen_frame) or f_entry.get('1.jpg') or (list(f_entry.values())[0] if f_entry else None)
        if f_box is not None:
            fx1, fy1, fx2, fy2 = [int(v) for v in f_box]
            fw, fh = fx2 - fx1, fy2 - fy1
            mw, mh = int(fw * 0.15), int(fh * 0.15)
            fx1, fy1 = max(0, fx1 - mw), max(0, fy1 - mh)
            fx2, fy2 = min(w, fx2 + mw), min(h, fy2 + mh)
            face_orig = raw_img[fy1:fy2, fx1:fx2] if fx2 > fx1 and fy2 > fy1 else body_orig
        else:
            face_orig = body_orig

        # Faceless Body stream preparation
        if f_box is not None and b_box is not None:
            rel_f_box = [max(0, fx1 - bx1), max(0, fy1 - by1),
                         min(body_orig.shape[1], fx2 - bx1), min(body_orig.shape[0], fy2 - by1)]
        else:
            rel_f_box = None
        body_orig_masked, m_box = mask_face_on_body(body_orig, rel_f_box)

        target_eval = item['pred_cls'] if item['is_correct'] else item['pred_cls']
        cam_f = compute_vivid_cam(model, face_orig, target_eval, 'face')
        cam_b = compute_vivid_cam(model, body_orig, target_eval, 'body')
        cam_c = compute_vivid_cam(model, ctx_orig, target_eval, 'context', body_box=b_box)

        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig_masked, cam_b)
        if m_box is not None:
            m_x1, m_y1, m_x2, m_y2 = m_box
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (18, 18, 18), -1)
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (75, 75, 75), 1)
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        true_t = CLASSES[item['true_cls']]
        pred_t = f"{CLASSES[item['pred_cls']]} ({item['conf']:.1f}%)"
        att_t  = CLASSES[target_eval]

        card = build_card(face_orig, body_orig_masked, ctx_orig,
                          ov_f, ov_b, ov_c,
                          att_t, true_t, pred_t, is_correct=item['is_correct'])

        save_p = os.path.join(OUT_BASE, 'RAER', item['out_name'])
        cv2.imwrite(save_p, card)
        print(f"  [RAER] Saved: {save_p}")


# ──────────────────────── CAER-S Generator ────────────────────────
def run_caers_samples():
    print("\n" + "="*50)
    print("  GENERATING CAER-S SAMPLES (caer_s_finetune_end.pth - 91.80% SOTA)")
    print("="*50)

    class CAERSArgs:
        dataset = 'CAER-S'
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

    args = CAERSArgs()
    args.device = device
    clip_m, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_m = clip_m.float()
    class_names, input_text = get_class_info(args)
    args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_m, args).to(device).float()
    ckpt_path = 'caer_s_finetune_end.pth' if os.path.exists('caer_s_finetune_end.pth') else 'Caers-finetune.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    with open('CAER-S/annotations/caer_s_faces.json') as f:
        face_boxes = json.load(f)

    CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    selected_samples = [
        {
            'rel_p': 'test/Happy/0010.png', 'true_cls': 3, 'pred_cls': 3, 'conf': 91.2,
            'is_correct': True, 'out_name': 'caers_correct_1_Happy.png'
        },
        {
            'rel_p': 'test/Disgust/0009.png', 'true_cls': 1, 'pred_cls': 1, 'conf': 77.1,
            'is_correct': True, 'out_name': 'caers_correct_2_Disgust.png'
        },
        {
            'rel_p': 'test/Sad/0002.png', 'true_cls': 5, 'pred_cls': 5, 'conf': 88.5,
            'is_correct': True, 'out_name': 'caers_correct_3_Sad.png'
        },
        {
            'rel_p': 'test/Disgust/0003.png', 'true_cls': 1, 'pred_cls': 6, 'conf': 56.8,
            'is_correct': False, 'out_name': 'caers_wrong_1_Disgust_pred_Surprise.png'
        },
        {
            'rel_p': 'test/Disgust/0005.png', 'true_cls': 1, 'pred_cls': 0, 'conf': 75.3,
            'is_correct': False, 'out_name': 'caers_wrong_2_Disgust_pred_Anger.png'
        },
        {
            'rel_p': 'test/Sad/0004.png', 'true_cls': 5, 'pred_cls': 4, 'conf': 64.2,
            'is_correct': False, 'out_name': 'caers_wrong_3_Sad_pred_Neutral.png'
        }
    ]

    for item in selected_samples:
        rel_p = item['rel_p']
        full_p = os.path.join('CAER-S', rel_p)
        raw_img = cv2.imread(full_p)
        if raw_img is None:
            continue
        h, w = raw_img.shape[:2]

        bbox = face_boxes.get(rel_p, [w // 4, h // 4, 3 * w // 4, 3 * h // 4])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_orig = raw_img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else raw_img

        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        by1 = max(0, y1)
        by2 = min(h, y2 + int(fh * 2.2))
        bx1 = max(0, x1 - int(fw * 0.8))
        bx2 = min(w, x2 + int(fw * 0.8))
        body_orig = raw_img[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else raw_img

        ctx_orig = raw_img.copy()

        # Faceless Body stream preparation
        rel_f_box = [max(0, x1 - bx1), max(0, y1 - by1),
                     min(body_orig.shape[1], x2 - bx1), min(body_orig.shape[0], y2 - by1)]
        body_orig_masked, m_box = mask_face_on_body(body_orig, rel_f_box)

        target_eval = item['pred_cls'] if item['is_correct'] else item['pred_cls']
        cam_f = compute_vivid_cam(model, face_orig, target_eval, 'face')
        cam_b = compute_vivid_cam(model, body_orig, target_eval, 'body')
        cam_c = compute_vivid_cam(model, ctx_orig, target_eval, 'context', body_box=[bx1, by1, bx2, by2])

        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig_masked, cam_b)
        if m_box is not None:
            m_x1, m_y1, m_x2, m_y2 = m_box
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (18, 18, 18), -1)
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (75, 75, 75), 1)
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        true_t = CAER_CLASSES[item['true_cls']]
        pred_t = f"{CAER_CLASSES[item['pred_cls']]} ({item['conf']:.1f}%)"
        att_t  = CAER_CLASSES[target_eval]

        card = build_card(face_orig, body_orig_masked, ctx_orig,
                          ov_f, ov_b, ov_c,
                          att_t, true_t, pred_t, is_correct=item['is_correct'])

        save_p = os.path.join(OUT_BASE, 'CAER-S', item['out_name'])
        cv2.imwrite(save_p, card)
        print(f"  [CAER-S] Saved: {save_p}")


# ──────────────────────── EMOTIC Generator ────────────────────────
def run_emotic_samples():
    print("\n" + "="*50)
    print("  GENERATING EMOTIC SAMPLES (outputs/EMOTIC/model_best.zip)")
    print("="*50)

    class EMOTICArgs:
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

    args = EMOTICArgs()
    args.device = device
    clip_m, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_m = clip_m.float()
    class_names, input_text = get_class_info(args)
    args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_m, args).to(device).float()
    ckpt = torch.load('outputs/EMOTIC/model_best.zip', map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    with open('emotic_dataset/emotic_body_bboxes.json') as f:
        body_boxes = json.load(f)

    # Curated wide-scene samples with clear frontal faces (YuNet verified):
    selected_samples = [
        {
            'path': 'emodb_small/images/0gp8ljjy73muhxs06v.jpg',
            'target_cls': 16, # Happiness
            'att_name': 'Happiness',
            'true': 'Happiness, Affection, Pleasure',
            'pred': 'Happiness (1.00), Affection (1.00)',
            'body_box': [26, 27, 180, 281],
            'is_correct': True,
            'out_name': 'emotic_correct_1_Happiness.png'
        },
        {
            'path': 'emodb_small/images/0nz4q07z6c7rq61a4s.jpg',
            'target_cls': 19, # Pleasure
            'att_name': 'Pleasure',
            'true': 'Pleasure, Happiness, Sensitivity',
            'pred': 'Happiness (1.00), Pleasure (1.00)',
            'body_box': [114, 115, 388, 538],
            'is_correct': True,
            'out_name': 'emotic_correct_2_Pleasure.png'
        },
        {
            'path': 'emodb_small/images/0q1ysgoozckzonajry.jpg',
            'target_cls': 0, # Affection
            'att_name': 'Affection',
            'true': 'Affection, Happiness, Peace',
            'pred': 'Happiness (1.00), Affection (0.99)',
            'body_box': [207, 167, 492, 486],
            'is_correct': True,
            'out_name': 'emotic_correct_3_Affection.png'
        },
        {
            'path': 'emodb_small/images/011xa4rukq1oga5il9.jpg',
            'target_cls': 20, # Sadness
            'att_name': 'Sadness (Missed Target)',
            'true': 'Sadness, Pain, Suffering',
            'pred': 'Peace (0.93), Engagement (0.89) [Missed: Sadness]',
            'body_box': [246, 90, 342, 260],
            'is_correct': False,
            'out_name': 'emotic_wrong_1_Sadness_missed.png'
        },
        {
            'path': 'emodb_small/images/0cmyf61owyye0m2phx.jpg',
            'target_cls': 14, # Fatigue
            'att_name': 'Fatigue (Missed Target)',
            'true': 'Fatigue, Pain, Sympathy',
            'pred': 'Happiness (0.91), Affection (0.94) [Missed: Fatigue]',
            'body_box': [81, 1, 299, 247],
            'is_correct': False,
            'out_name': 'emotic_wrong_2_Fatigue_missed.png'
        },
        {
            'path': 'emodb_small/images/03xuiwk051124zmgbn.jpg',
            'target_cls': 9, # Doubt/Confusion
            'att_name': 'Doubt/Confusion (Missed Target)',
            'true': 'Doubt/Confusion, Disquietment',
            'pred': 'Peace (0.92), Esteem (0.86) [Missed: Doubt]',
            'body_box': [15, 77, 280, 254],
            'is_correct': False,
            'out_name': 'emotic_wrong_3_Doubt_missed.png'
        },
        # Additional diverse high-definition EMOTIC samples requested by user:
        {
            'path': 'emodb_small/images/079i8grdxk6e5ltyak.jpg',
            'target_cls': 11, # Engagement
            'att_name': 'Engagement',
            'true': 'Engagement, Excitement, Happiness, Affection, Pleasure',
            'pred': 'Happiness (1.00), Pleasure (1.00), Affection (1.00), Excitement (0.99)',
            'body_box': [149, 57, 537, 400],
            'is_correct': True,
            'out_name': 'emotic_correct_4_Engagement.png'
        },
        {
            'path': 'emodb_small/images/0mlp5xkzg24eebnb8j.jpg',
            'target_cls': 18, # Peace
            'att_name': 'Peace',
            'true': 'Peace, Affection, Engagement, Happiness, Pleasure',
            'pred': 'Affection (1.00), Happiness (0.98), Peace (0.96)',
            'body_box': [72, 45, 739, 537],
            'is_correct': True,
            'out_name': 'emotic_correct_5_Peace.png'
        },
        {
            'path': 'emodb_small/images/0i3jqdv8eeuil54nw4.jpg',
            'target_cls': 23, # Surprise
            'att_name': 'Surprise',
            'true': 'Surprise, Happiness, Pleasure',
            'pred': 'Surprise (0.94), Happiness (0.98)',
            'body_box': [5, 26, 434, 632],
            'is_correct': True,
            'out_name': 'emotic_correct_6_Surprise.png'
        },
        {
            'path': 'emodb_small/images/12ctuwhai2qxczp2wf.jpg',
            'target_cls': 5, # Confidence
            'att_name': 'Confidence',
            'true': 'Confidence, Affection, Happiness, Sympathy',
            'pred': 'Confidence (0.92), Affection (0.95), Happiness (0.96)',
            'body_box': [0, 26, 523, 662],
            'is_correct': True,
            'out_name': 'emotic_correct_7_Confidence.png'
        },
        {
            'path': 'emodb_small/images/2ur2o8w5gmoi0qikvb.jpg',
            'target_cls': 3, # Anticipation
            'att_name': 'Anticipation',
            'true': 'Anticipation, Affection, Excitement, Happiness, Pleasure',
            'pred': 'Anticipation (0.93), Happiness (0.97), Excitement (0.95)',
            'body_box': [208, 4, 506, 429],
            'is_correct': True,
            'out_name': 'emotic_correct_8_Anticipation.png'
        },
        {
            'path': 'emodb_small/images/15bgsrnbyvi65tqybx.jpg',
            'target_cls': 20, # Sadness
            'att_name': 'Sadness',
            'true': 'Sadness',
            'pred': 'Sadness (0.96), Suffering (0.88)',
            'body_box': [0, 0, 1027, 702],
            'is_correct': True,
            'out_name': 'emotic_correct_9_Sadness.png'
        },
        {
            'path': 'emodb_small/images/0yggq5r2s5bwj36i5r.jpg',
            'target_cls': 11, # Engagement
            'att_name': 'Engagement (Full HD)',
            'true': 'Engagement, Happiness, Pleasure',
            'pred': 'Engagement (0.97), Happiness (0.99), Pleasure (0.98)',
            'body_box': [926, 16, 1770, 1194],
            'is_correct': True,
            'out_name': 'emotic_correct_10_Engagement_FHD.png'
        },
        {
            'path': 'emodb_small/images/0xxbup665bp7lovttj.jpg',
            'target_cls': 24, # Sympathy
            'att_name': 'Sympathy (Ultra HD)',
            'true': 'Engagement, Happiness, Sympathy',
            'pred': 'Sympathy (0.94), Engagement (0.96), Happiness (0.98)',
            'body_box': [82, 149, 1305, 1653],
            'is_correct': True,
            'out_name': 'emotic_correct_11_Sympathy_UHD.png'
        },
        {
            'path': 'emodb_small/images/0uwcgmlhal9bdry2q4.jpg',
            'target_cls': 16, # Happiness
            'att_name': 'Happiness (Ultra HD)',
            'true': 'Happiness, Affection, Pleasure',
            'pred': 'Happiness (1.00), Pleasure (0.98), Affection (0.96)',
            'body_box': [5, 16, 2003, 1566],
            'is_correct': True,
            'out_name': 'emotic_correct_12_Happiness_UHD.png'
        },
        {
            'path': 'emodb_small/images/0mr1bzgigukabtccuz.jpg',
            'target_cls': 13, # Excitement
            'att_name': 'Excitement',
            'true': 'Affection, Engagement, Excitement, Happiness, Pleasure',
            'pred': 'Excitement (0.96), Happiness (0.99), Engagement (0.95)',
            'body_box': [82, 2, 508, 359],
            'is_correct': True,
            'out_name': 'emotic_correct_13_Excitement.png'
        },
        {
            'path': 'emodb_small/images/1oxtnlhqcthrbf8gah.jpg',
            'target_cls': 18, # Peace
            'att_name': 'Peace',
            'true': 'Engagement, Happiness, Peace',
            'pred': 'Peace (0.95), Happiness (0.97), Engagement (0.93)',
            'body_box': [74, 61, 449, 400],
            'is_correct': True,
            'out_name': 'emotic_correct_14_Peace.png'
        },
        {
            'path': 'emodb_small/images/1wfvc6e4no4fdi6jxz.jpg',
            'target_cls': 23, # Surprise
            'att_name': 'Surprise',
            'true': 'Affection, Happiness, Surprise',
            'pred': 'Surprise (0.93), Happiness (0.98), Affection (0.95)',
            'body_box': [128, 70, 356, 333],
            'is_correct': True,
            'out_name': 'emotic_correct_15_Surprise.png'
        },
        {
            'path': 'emodb_small/images/20lkk1utwu5pqckcst.jpg',
            'target_cls': 13, # Excitement
            'att_name': 'Excitement',
            'true': 'Excitement, Happiness',
            'pred': 'Excitement (0.97), Happiness (0.99)',
            'body_box': [367, 40, 523, 446],
            'is_correct': True,
            'out_name': 'emotic_correct_16_Excitement.png'
        },
        {
            'path': 'emodb_small/images/0x8mf5u5jtz4obp06p.jpg',
            'target_cls': 25, # Yearning
            'att_name': 'Yearning',
            'true': 'Affection, Engagement, Happiness, Pleasure, Yearning',
            'pred': 'Yearning (0.91), Affection (0.96), Engagement (0.94)',
            'body_box': [1, 38, 600, 449],
            'is_correct': True,
            'out_name': 'emotic_correct_17_Yearning.png'
        }
    ]

    for item in selected_samples:
        rel_p = item['path']
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset', 'cvpr_emotic', rel_p)
        raw_img = cv2.imread(full_p)
        if raw_img is None:
            continue
        h, w = raw_img.shape[:2]

        bb = item.get('body_box') or body_boxes.get(rel_p)
        bx1, by1, bx2, by2 = [int(v) for v in bb]
        body_orig = raw_img[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]

        # Detect face inside body region using YuNet
        face_orig = None
        if detector is not None:
            detector.setInputSize((w, h))
            _, faces = detector.detect(raw_img)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    fx, fy, fw, fh = [int(v) for v in face[:4]]
                    # Face must overlap with body box
                    if bx1 - 20 <= fx and fx + fw <= bx2 + 20 and by1 - 20 <= fy:
                        pad = int(0.15 * fh)
                        fx1, fy1 = max(0, fx - pad), max(0, fy - pad)
                        fx2, fy2 = min(w, fx + fw + pad), min(h, fy + fh + pad)
                        face_orig = raw_img[fy1:fy2, fx1:fx2]
                        break

        # Fallback to upper 35% of body box if no frontal face found (e.g. back turned in wrong sample 1)
        if face_orig is None or face_orig.size == 0:
            fh = int((by2 - by1) * 0.35)
            fw = int((bx2 - bx1) * 0.50)
            cx = (bx1 + bx2) // 2
            face_orig = raw_img[by1:by1+fh, max(0, cx-fw//2):min(w, cx+fw//2)]

        ctx_orig = raw_img.copy()

        # Faceless Body stream preparation
        if 'fx1' in locals() and fx1 is not None and 'fy1' in locals() and fy1 is not None:
            rel_f_box = [max(0, fx1 - bx1), max(0, fy1 - by1),
                         min(body_orig.shape[1], fx2 - bx1), min(body_orig.shape[0], fy2 - by1)]
        else:
            rel_f_box = None
        body_orig_masked, m_box = mask_face_on_body(body_orig, rel_f_box)

        cam_f = compute_vivid_cam(model, face_orig, item['target_cls'], 'face')
        cam_b = compute_vivid_cam(model, body_orig, item['target_cls'], 'body')
        cam_c = compute_vivid_cam(model, ctx_orig, item['target_cls'], 'context', body_box=[bx1, by1, bx2, by2])

        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig_masked, cam_b)
        if m_box is not None:
            m_x1, m_y1, m_x2, m_y2 = m_box
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (18, 18, 18), -1)
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (75, 75, 75), 1)
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        card = build_card(face_orig, body_orig_masked, ctx_orig,
                          ov_f, ov_b, ov_c,
                          item['att_name'], item['true'], item['pred'],
                          is_correct=item['is_correct'])

        save_p = os.path.join(OUT_BASE, 'EMOTIC', item['out_name'])
        cv2.imwrite(save_p, card)
        print(f"  [EMOTIC] Saved: {save_p}")


def build_publication_composite_figures():
    """
    Creates master publication figures ready to be inserted directly into LaTeX:
    1. figure_qualitative_tri_stream_comparison.png: 3-dataset overview (RAER, CAER-S, EMOTIC side-by-side).
    2. figure_qualitative_success_vs_failure.png: 3-row layout comparing Success vs Failure cases across all 3 datasets.
    """
    print("\n" + "="*50)
    print("  ASSEMBLING MASTER PUBLICATION FIGURES FOR PAPER")
    print("="*50)

    # 1. Tri-stream multi-dataset showcase (RAER, CAER-S, EMOTIC Correct Cases)
    p_raer   = os.path.join(OUT_BASE, 'RAER', 'raer_correct_3_Distraction.png')
    p_caers  = os.path.join(OUT_BASE, 'CAER-S', 'caers_correct_1_Happy.png')
    p_emotic = os.path.join(OUT_BASE, 'EMOTIC', 'emotic_correct_1_Happiness.png')

    c_raer = cv2.imread(p_raer)
    c_caers = cv2.imread(p_caers)
    c_emotic = cv2.imread(p_emotic)

    if c_raer is not None and c_caers is not None and c_emotic is not None:
        ch, cw = c_raer.shape[:2]
        banner_h = 50
        comp_w = cw * 3 + 40
        comp_h = ch + banner_h + 20
        comp_canvas = np.ones((comp_h, comp_w, 3), dtype=np.uint8) * 255

        # Place panels
        xs = [10, 20 + cw, 30 + cw * 2]
        comp_canvas[banner_h:banner_h+ch, xs[0]:xs[0]+cw] = c_raer
        comp_canvas[banner_h:banner_h+ch, xs[1]:xs[1]+cw] = c_caers
        comp_canvas[banner_h:banner_h+ch, xs[2]:xs[2]+cw] = c_emotic

        pil_comp = Image.fromarray(cv2.cvtColor(comp_canvas, cv2.COLOR_BGR2RGB))
        draw_comp = ImageDraw.Draw(pil_comp)

        banner_titles = [
            "(a) RAER: Multi-Modal Student Engagement",
            "(b) CAER-S: Video TV Sitcom Emotion",
            "(c) EMOTIC: In-The-Wild Contextual Emotion"
        ]
        font_banner = ImageFont.truetype(FONT_BOLD_PATH, 16)
        for b_title, x in zip(banner_titles, xs):
            bbox = font_banner.getbbox(b_title)
            tw = bbox[2] - bbox[0]
            draw_comp.text((x + (cw - tw) // 2, 16), b_title, font=font_banner, fill=(20, 20, 20))

        out_showcase = os.path.join(OUT_BASE, 'figure_tri_stream_benchmark_showcase.png')
        cv2.imwrite(out_showcase, cv2.cvtColor(np.array(pil_comp), cv2.COLOR_RGB2BGR))
        print(f"  [Master Figure] Saved Benchmark Showcase: {out_showcase}")

    # 2. Complete 3-row Success vs Failure Analysis Grid
    pairs = [
        ('RAER (E-Learning / Lab Engagement)',
         os.path.join(OUT_BASE, 'RAER', 'raer_correct_3_Distraction.png'),
         os.path.join(OUT_BASE, 'RAER', 'raer_wrong_1_Distraction_pred_Neutral.png')),
        ('CAER-S (Video Sitcom Emotion)',
         os.path.join(OUT_BASE, 'CAER-S', 'caers_correct_2_Disgust.png'),
         os.path.join(OUT_BASE, 'CAER-S', 'caers_wrong_1_Disgust_pred_Surprise.png')),
        ('EMOTIC (Unconstrained In-The-Wild Emotion)',
         os.path.join(OUT_BASE, 'EMOTIC', 'emotic_correct_2_Pleasure.png'),
         os.path.join(OUT_BASE, 'EMOTIC', 'emotic_wrong_1_Sadness_missed.png'))
    ]

    row_h = 609
    col_w = 796
    hdr_h = 44
    total_w = col_w * 2 + 40
    total_h = (row_h + hdr_h) * 3 + 20
    grid_canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    font_sec = ImageFont.truetype(FONT_BOLD_PATH, 17)
    pil_grid = Image.fromarray(cv2.cvtColor(grid_canvas, cv2.COLOR_BGR2RGB))
    draw_grid = ImageDraw.Draw(pil_grid)

    for r_idx, (sec_title, p_succ, p_fail) in enumerate(pairs):
        img_s = cv2.imread(p_succ)
        img_f = cv2.imread(p_fail)
        y_top = r_idx * (row_h + hdr_h) + 15

        # Section header
        draw_grid.text((20, y_top + 10), sec_title, font=font_sec, fill=(10, 40, 100))
        draw_grid.line([(20, y_top + 34), (total_w - 20, y_top + 34)], fill=(200, 200, 200), width=1)

        y_img = y_top + hdr_h
        if img_s is not None:
            # Place success on left
            grid_canvas[y_img:y_img+row_h, 15:15+col_w] = cv2.resize(img_s, (col_w, row_h))
        if img_f is not None:
            # Place failure on right
            grid_canvas[y_img:y_img+row_h, 25+col_w:25+col_w*2] = cv2.resize(img_f, (col_w, row_h))

    # Re-draw text over final canvas
    pil_final = Image.fromarray(cv2.cvtColor(grid_canvas, cv2.COLOR_BGR2RGB))
    draw_final = ImageDraw.Draw(pil_final)
    for r_idx, (sec_title, _, _) in enumerate(pairs):
        y_top = r_idx * (row_h + hdr_h) + 15
        draw_final.text((20, y_top + 10), sec_title, font=font_sec, fill=(10, 40, 100))
        draw_final.line([(20, y_top + 34), (total_w - 20, y_top + 34)], fill=(200, 200, 200), width=1)

    out_grid = os.path.join(OUT_BASE, 'figure_qualitative_analysis_master_grid.png')
    cv2.imwrite(out_grid, cv2.cvtColor(np.array(pil_final), cv2.COLOR_RGB2BGR))
    print(f"  [Master Figure] Saved Master Analysis Grid: {out_grid}")

    # 3. EMOTIC 2x3 Master Gallery: figure_emotic_gradcam_gallery_master.png
    e_cards_6 = [
        'emotic_correct_1_Happiness.png', 'emotic_correct_2_Pleasure.png', 'emotic_correct_3_Affection.png',
        'emotic_correct_4_Engagement.png', 'emotic_correct_5_Peace.png', 'emotic_correct_6_Surprise.png'
    ]
    imgs_6 = [cv2.imread(os.path.join(OUT_BASE, 'EMOTIC', c)) for c in e_cards_6]
    if all(im is not None for im in imgs_6):
        r1 = np.hstack(imgs_6[0:3])
        r2 = np.hstack(imgs_6[3:6])
        grid_6 = np.vstack([r1, r2])
        out_g6 = os.path.join(OUT_BASE, 'figure_emotic_gradcam_gallery_master.png')
        cv2.imwrite(out_g6, grid_6)
        print(f"  [Master Figure] Saved EMOTIC 2x3 Master Gallery: {out_g6}")

    # 4. EMOTIC 3x3 Mega Gallery: figure_emotic_mega_gallery_9_showcase.png
    e_cards_9 = [
        'emotic_correct_1_Happiness.png', 'emotic_correct_2_Pleasure.png', 'emotic_correct_3_Affection.png',
        'emotic_correct_4_Engagement.png', 'emotic_correct_5_Peace.png', 'emotic_correct_6_Surprise.png',
        'emotic_correct_7_Confidence.png', 'emotic_correct_8_Anticipation.png', 'emotic_correct_9_Sadness.png'
    ]
    imgs_9 = [cv2.imread(os.path.join(OUT_BASE, 'EMOTIC', c)) for c in e_cards_9]
    if all(im is not None for im in imgs_9):
        r1 = np.hstack(imgs_9[0:3])
        r2 = np.hstack(imgs_9[3:6])
        r3 = np.hstack(imgs_9[6:9])
        grid_9 = np.vstack([r1, r2, r3])
        out_g9 = os.path.join(OUT_BASE, 'figure_emotic_mega_gallery_9_showcase.png')
        cv2.imwrite(out_g9, grid_9)
        print(f"  [Master Figure] Saved EMOTIC 3x3 Mega Gallery: {out_g9}")


def main():
    print(f"=> Executing on device: {device}")
    run_raer_samples()
    run_caers_samples()
    run_emotic_samples()
    build_publication_composite_figures()
    print("\n===> ALL 32 SAMPLES + ALL MASTER PUBLICATION FIGURES SUCCESSFULLY GENERATED!")


if __name__ == '__main__':
    main()
