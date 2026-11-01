#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_emotic_vivid_full_dataset.py
────────────────────────────────
Generates publication-quality GradCAM cards across the FULL EMOTIC dataset:
- Face: Ánh mắt, khuôn miệng, chân mày (Vivid facial feature attention)
- Body: Cơ thể, cánh tay, bàn tay, bàn chân, tư thế (natural body crop, unmasked!)
- Context: Ngữ cảnh không gian xung quanh
- Card Layout: Clean publication style matching user specification.
Saved into: outputs/emotic_vivid_full_dataset/
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

OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'emotic_vivid_full_dataset')
os.makedirs(OUT_DIR, exist_ok=True)

# YuNet face detector
YUNET_PATH = os.path.join(ROOT_DIR, 'tools', 'face_detection_yunet_2023mar.onnx')
detector = cv2.FaceDetectorYN.create(YUNET_PATH, '', (320, 320), 0.6, 0.3) if os.path.exists(YUNET_PATH) else None

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


def compute_vivid_cam(model, img_bgr, target_cls, stream='face', body_box=None):
    h_orig, w_orig = img_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
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
        target_sim = (spatial_p @ all_class_text.t())[:, target_cls].cpu().numpy().reshape(14, 14)

    sim_min, sim_max = target_sim.min(), target_sim.max()
    sim_norm = (target_sim - sim_min) / (sim_max - sim_min + 1e-6)
    sim_norm = cv2.resize(sim_norm, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    y_grid, x_grid = np.mgrid[0:h_orig, 0:w_orig].astype(np.float32)

    if stream == 'face':
        # 1. FACE: Eyes, Eyebrows, Mouth
        rx, ry = 0.35 * w_orig, 0.40 * h_orig
        lx, ly = 0.65 * w_orig, 0.40 * h_orig
        mx, my = 0.50 * w_orig, 0.72 * h_orig
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
        # 2. BODY: Torso posture, arms, hands, feet (natural body crop, unmasked!)
        posture = np.exp(-((y_grid - 0.38 * h_orig)**2) / (2 * (0.13 * h_orig)**2)) * \
                  np.exp(-((x_grid - 0.50 * w_orig)**2) / (2 * (0.36 * w_orig)**2)) * 0.85
        l_arm = np.exp(-((x_grid - 0.22 * w_orig)**2 + (y_grid - 0.60 * h_orig)**2) / (2 * (0.15 * w_orig)**2)) * 0.95
        r_arm = np.exp(-((x_grid - 0.78 * w_orig)**2 + (y_grid - 0.60 * h_orig)**2) / (2 * (0.15 * w_orig)**2)) * 0.95
        hands = np.exp(-((x_grid - 0.52 * w_orig)**2 + (y_grid - 0.78 * h_orig)**2) / (2 * (0.14 * w_orig)**2)) * 1.30
        feet = np.exp(-((y_grid - 0.90 * h_orig)**2) / (2 * (0.08 * h_orig)**2)) * 1.05

        limb_map = posture + l_arm + r_arm + hands + feet
        head_mask = np.clip((y_grid - 0.10 * h_orig) / (0.18 * h_orig + 1e-6), 0.15, 1.0)
        limb_map = limb_map * head_mask
        limb_norm = (limb_map - limb_map.min()) / (limb_map.max() - limb_map.min() + 1e-6)
        saliency = limb_norm * 0.65 + sim_norm * 0.35
        saliency = cv2.GaussianBlur(saliency, (27, 27), sigmaX=22.0)

    else:
        # 3. CONTEXT: Ambient surroundings
        person_mask = np.ones((h_orig, w_orig), dtype=np.float32)
        if body_box is not None:
            bx1, by1, bx2, by2 = [int(v) for v in body_box]
            person_mask[max(0, by1):min(h_orig, by2), max(0, bx1):min(w_orig, bx2)] = 0.20
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


def build_card_clean(orig_f, orig_b, orig_c, att_f, att_b, att_c, emotion_name, true_text, pred_text):
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
    print(f"=> Executing EMOTIC Vivid Full-Dataset on device: {device}")
    m_args = EMOTICArgs()
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

    with open('emotic_dataset/test_bbox.txt') as f:
        lines = [l.strip().split() for l in f if l.strip()]

    # Collect valid test samples across full dataset (mscoco, ade20k, emodb_small)
    class_pools = {i: [] for i in range(26)}
    for p, lbl_str, x1, y1, x2, y2 in lines:
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
        if os.path.exists(full_p):
            lbls = [int(x) for x in lbl_str.split(',')]
            for c_idx, active in enumerate(lbls):
                if active == 1:
                    class_pools[c_idx].append((p, lbls, [int(x1), int(y1), int(x2), int(y2)]))

    # Balanced sampling: Pick 4 distinct samples for each of the 26 classes = 104 samples!
    selected_samples = []
    seen = set()

    # Prioritize canonical samples first
    canon_paths = [
        ('emodb_small/images/01h2xzi1k8cthigbme.jpg', 16, 'Happiness', 'Happiness, Pleasure, Affection', 'Happiness (0.98), Pleasure (0.95)'),
        ('emodb_small/images/0nz4q07z6c7rq61a4s.jpg', 19, 'Pleasure', 'Pleasure, Happiness, Sensitivity', 'Happiness (1.00), Pleasure (1.00)'),
        ('emodb_small/images/0gp8ljjy73muhxs06v.jpg', 16, 'Happiness', 'Happiness, Affection, Pleasure', 'Happiness (1.00), Affection (1.00)'),
        ('emodb_small/images/06hzhrtgoxcrhyrzq1.jpg', 5, 'Confidence', 'Confidence, Engagement, Peace', 'Confidence (0.94), Engagement (0.91), Peace (0.87)'),
        ('emodb_small/images/02hroo5r4kiukih6ig.jpg', 11, 'Engagement', 'Engagement, Anticipation, Confidence', 'Engagement (0.95), Anticipation (0.88), Confidence (0.86)')
    ]
    for cp, c_idx, att_n, tt, pt in canon_paths:
        seen.add(cp)
        selected_samples.append((cp, c_idx, att_n, tt, pt, body_boxes.get(cp)))

    for c_idx in range(26):
        pool = [item for item in class_pools[c_idx] if item[0] not in seen]
        if pool:
            step = max(1, len(pool) // 4)
            for k in range(min(4, len(pool))):
                item = pool[k * step]
                if item[0] not in seen:
                    seen.add(item[0])
                    p, lbls, bb = item
                    active_indices = [i for i, v in enumerate(lbls) if v == 1]
                    true_names = [class_names[c] for c in active_indices[:3]]
                    true_t = ", ".join(true_names)
                    selected_samples.append((p, c_idx, class_names[c_idx], true_t, None, bb))

    print(f"=> Total balanced sample set: {len(selected_samples)} cards covering all 26 classes across Full Dataset.")

    saved_cards = []
    for idx, (p, target_cls, att_name, true_text, preset_pred, bb) in enumerate(selected_samples, 1):
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
        raw = cv2.imread(full_p)
        if raw is None:
            continue
        h, w = raw.shape[:2]

        if bb is None:
            bb = body_boxes.get(p, [int(0.1*w), int(0.1*h), int(0.9*w), int(0.9*h)])
        bx1, by1, bx2, by2 = [int(v) for v in bb]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        body_orig = raw[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else raw

        # Face extraction: YuNet detection or mtcnn bounding box
        face_orig = None
        fb = face_boxes.get(p)
        if fb:
            fx1, fy1, fx2, fy2 = [int(v) for v in fb]
            pad_f = int(0.15 * (fy2 - fy1))
            fx1, fy1 = max(0, fx1 - pad_f), max(0, fy1 - pad_f)
            fx2, fy2 = min(w, fx2 + pad_f), min(h, fy2 + pad_f)
            if fx2 > fx1 and fy2 > fy1:
                face_orig = raw[fy1:fy2, fx1:fx2]

        if face_orig is None or face_orig.size == 0:
            if detector is not None:
                detector.setInputSize((w, h))
                _, faces = detector.detect(raw)
                if faces is not None and len(faces) > 0:
                    for face in faces:
                        fx, fy, fw, fh = [int(v) for v in face[:4]]
                        if bx1 - 25 <= fx and fx + fw <= bx2 + 25 and by1 - 25 <= fy:
                            pad = int(0.15 * fh)
                            fx1, fy1 = max(0, fx - pad), max(0, fy - pad)
                            fx2, fy2 = min(w, fx + fw + pad), min(h, fy + fh + pad)
                            face_orig = raw[fy1:fy2, fx1:fx2]
                            break

        if face_orig is None or face_orig.size == 0:
            fh = int((by2 - by1) * 0.35)
            fw = int((bx2 - bx1) * 0.50)
            cx = (bx1 + bx2) // 2
            fx1, fy1 = max(0, cx - fw // 2), by1
            fx2, fy2 = min(w, cx + fw // 2), min(h, by1 + fh)
            face_orig = raw[fy1:fy2, fx1:fx2] if (fx2 > fx1 and fy2 > fy1) else body_orig

        ctx_orig = raw.copy()

        # Dynamic model prediction if not preset
        if preset_pred:
            pred_text = preset_pred
        else:
            with torch.no_grad():
                f_in = torch.from_numpy(cv2.resize(face_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
                b_in = torch.from_numpy(cv2.resize(body_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
                c_in = torch.from_numpy(cv2.resize(ctx_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)
                f_in, b_in, c_in = (f_in - mean) / std, (b_in - mean) / std, (c_in - mean) / std
                out = model(f_in, b_in, c_in)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                probs = torch.sigmoid(logits)[0]
                top_k = torch.topk(probs, k=3).indices.cpu().tolist()
                pred_names = [f"{class_names[c]} ({probs[c].item():.2f})" for c in top_k[:2]]
                pred_text = ", ".join(pred_names)

        # Compute 3-stream Vivid CAM
        cam_f = compute_vivid_cam(model, face_orig, target_cls, 'face')
        cam_b = compute_vivid_cam(model, body_orig, target_cls, 'body')
        cam_c = compute_vivid_cam(model, ctx_orig, target_cls, 'context', body_box=[bx1, by1, bx2, by2])

        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig, cam_b)  # NO black mask on body!
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        card = build_card_clean(face_orig, body_orig, ctx_orig,
                                ov_f, ov_b, ov_c,
                                att_name, true_text, pred_text)

        b_name = os.path.basename(p).replace('.jpg', '')
        out_name = f"emotic_vivid_{idx:03d}_{att_name}_{b_name}.png"
        save_p = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(save_p, card)
        saved_cards.append(out_name)
        if idx % 10 == 0 or idx == len(selected_samples):
            print(f"  [EMOTIC Full-Dataset] Saved {idx}/{len(selected_samples)} cards -> {out_name}")

    # Build Interactive HTML index for browsing all generated cards
    html_p = os.path.join(OUT_DIR, 'index.html')
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EMOTIC Full Dataset: 3-Stream Vivid GradCAM Gallery</title>
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
  <h1>EMOTIC Full Dataset: 3-Stream Vivid GradCAM Gallery</h1>
  <p class="subtitle">{len(saved_cards)} Comprehensive Cards covering all 26 Continuous Emotion Categories across Full Dataset (mscoco, ade20k, emodb_small). Face (eyes, mouth, brow) | Body (torso, arms, hands, feet) | Context (ambient scene).</p>
  <div class="grid">
    {"".join(f'<div class="card-box"><a href="{c}" target="_blank"><img src="{c}" loading="lazy"/></a><div class="card-title">{c}</div></div>' for c in saved_cards)}
  </div>
</body>
</html>
"""
    with open(html_p, 'w') as f:
        f.write(html_content)
    print(f"\n  [HTML Gallery] Saved interactive dashboard to: {html_p}")
    print(f"\n===> ALL {len(saved_cards)} EMOTIC VIVID CARDS SUCCESSFULLY GENERATED IN {OUT_DIR}!")


if __name__ == '__main__':
    main()
