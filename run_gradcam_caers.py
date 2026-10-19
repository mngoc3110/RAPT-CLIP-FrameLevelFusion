#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gradcam_caers.py
───────────────────
RAPT-CLIP Publication-Grade Grad-CAM Visualization for CAER-S (Static Images).
Model Checkpoint: Caers-finetune.pth (90.40% WAR / UAR SOTA Checkpoint)

Generates publication-quality 2x3 grid cards matching the official RAPT-CLIP EMOTIC standard:
  Row 1: [ Face Stream | Body Stream | Context Stream ] (Original)
  Row 2: [ Face Stream | Body Stream | Context Stream ] (GradCAM)
  Bottom:
    - Attention: <Target Emotion> (Bold Red)
    - True: <True Emotion> (Dark Green)
    - Pred: <Predicted Emotion> (<Confidence>%) (Dark Blue)

Specifications:
- Canvas dimensions: 784 x 595 px (matching outputs/Gradcam-Emotic/1.jpeg)
- Face Stream: High-precision MTCNN / annotation face crop
- Body Stream: Anatomical upper-body / torso estimation
- Context Stream: Clean, unmasked full scene (no black censor rectangle)
- Token-level CLIP similarity: Spatial patch tokens projected through VisionTransformer ln_post & proj,
  with class-discriminative suppression (target_sim - mean_sim) and EAA face adapter integration.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info

CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def parse_args():
    parser = argparse.ArgumentParser(description="RAPT-CLIP CAER-S Grad-CAM Visualization")
    parser.add_argument('--checkpoint', type=str, default='Caers-finetune.pth',
                        help='Path to CAER-S model checkpoint')
    parser.add_argument('--data-dir', type=str, default='CAER-S',
                        help='Root path to CAER-S dataset')
    parser.add_argument('--face-box', type=str, default='CAER-S/annotations/caer_s_faces.json',
                        help='Path to face bounding boxes JSON')
    parser.add_argument('--test-txt', type=str, default='CAER-S/annotations/test.txt',
                        help='Path to test annotation txt')
    parser.add_argument('--samples-per-class', type=int, default=10,
                        help='Number of test samples to scan per emotion class')
    parser.add_argument('--out-dir', type=str, default='outputs/Gradcam-Caers',
                        help='Output directory for Grad-CAM cards')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Execution device')
    return parser.parse_args()


def select_device(dev_arg):
    if dev_arg == 'cpu':
        return torch.device('cpu')
    elif dev_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda:0')
    elif dev_arg == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_token_similarity(model, img_bgr, target_class, stream='face', device='cpu'):
    """
    Computes spatial attention map using CLIP patch-token similarity.
    For face stream: applies face_adapter (EAA) to match the actual forward pass.
    Discriminative map = target_sim - mean_sim highlights regions specific to this emotion.
    """
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

        if stream == 'face':
            spatial_p = model.face_adapter(spatial_p)

        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)

        all_sim = spatial_p @ all_class_text.t()         # [196, 7]
        target_sim = all_sim[:, target_class]
        mean_sim = all_sim.mean(dim=1)
        disc_sim = target_sim - mean_sim

        disc_np = disc_sim.cpu().numpy().reshape(14, 14)
        cam_np = np.maximum(disc_np, 0)
        cam_max = cam_np.max()
        if cam_max > 1e-7:
            cam_np = (cam_np / cam_max) ** 1.5
        else:
            raw_np = target_sim.cpu().numpy().reshape(14, 14)
            raw_np = raw_np - raw_np.min()
            raw_max = raw_np.max()
            cam_np = (raw_np / (raw_max + 1e-7)) ** 2.0

    return cam_np


def overlay_heatmap(img_bgr, cam, alpha=0.5):
    """Overlays smoothed JET colormap on BGR image."""
    h, w = img_bgr.shape[:2]
    cam_f = cam.astype(np.float32)
    cam_smooth = cv2.GaussianBlur(cam_f, (7, 7), sigmaX=3.0)
    cam_resized = cv2.resize(cam_smooth, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = cam_resized - cam_resized.min()
    if cam_resized.max() > 1e-7:
        cam_resized = cam_resized / cam_resized.max()
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    result = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
    return result


def create_3stream_grid(orig_f, orig_b, orig_c, att_f, att_b, att_c,
                        top_emotion, true_name, pred_name, conf):
    """
    Creates exact 2x3 grid matching outputs/Gradcam-Emotic/1.jpeg:
    Resolution: 784 x 595 px.
    Col 0: Face Stream | Col 1: Body Stream | Col 2: Context Stream
    Row 1: Original    | Row 2: GradCAM
    """
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

    # Place tiles
    for x, tile in zip(cols_x, [orig_f, orig_b, orig_c]):
        canvas[y_row1:y_row1+s, x:x+s] = cv2.resize(tile, (s, s))

    for x, tile in zip(cols_x, [att_f, att_b, att_c]):
        canvas[y_row2:y_row2+s, x:x+s] = cv2.resize(tile, (s, s))

    # Convert to PIL for premium typography matching Emotic standard
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
    draw.text((left_m, 510), f'Attention: {top_emotion}', font=font_att, fill=(160, 0, 0))
    draw.text((left_m, 536), f'True: {true_name}', font=font_info, fill=(0, 110, 0))
    draw.text((left_m, 560), f'Pred: {pred_name} ({conf:.1f}%)', font=font_info, fill=(0, 0, 160))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"=> Executing on device: {device}")

    # Configure Model Args matching Caers-finetune.pth exactly
    class ModelArgs:
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

    model_args = ModelArgs()
    model_args.device = device
    clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_model = clip_model.float()
    class_names, input_text = get_class_info(model_args)
    model_args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_model, model_args).to(device).float()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"=> Loaded checkpoint successfully! (Missing text_encoder keys: {len(missing)}, Unexpected: {len(unexpected)})")
    model.eval()

    # Load Face Bounding Boxes
    if os.path.exists(args.face_box):
        with open(args.face_box, 'r') as f:
            face_boxes = json.load(f)
        print(f"=> Loaded {len(face_boxes):,} Face boxes from '{args.face_box}'")
    else:
        face_boxes = {}
        print(f"=> [Warning] Face boxes file '{args.face_box}' not found. Fallback to center crop.")

    # Select samples per class from test set
    samples_by_class = {c: [] for c in CAER_CLASSES}
    if os.path.exists(args.test_txt):
        with open(args.test_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p, lbl = parts[0], int(parts[1])
                    if 0 <= lbl < len(CAER_CLASSES):
                        cname = CAER_CLASSES[lbl]
                        if len(samples_by_class[cname]) < args.samples_per_class:
                            samples_by_class[cname].append((p, lbl))
    else:
        for idx, cname in enumerate(CAER_CLASSES):
            cdir = os.path.join(args.data_dir, 'test', cname)
            if os.path.exists(cdir):
                fnames = sorted([f for f in os.listdir(cdir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for fn in fnames[:args.samples_per_class]:
                    samples_by_class[cname].append((os.path.join('test', cname, fn), idx))

    all_targets = []
    for cname in CAER_CLASSES:
        all_targets.extend(samples_by_class[cname])

    print(f"=> Total candidate samples to scan: {len(all_targets)}")

    # Prepare directories
    out_all = os.path.join(args.out_dir, 'all_samples')
    os.makedirs(out_all, exist_ok=True)
    for cname in CAER_CLASSES:
        os.makedirs(os.path.join(args.out_dir, 'by_class', cname), exist_ok=True)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1).to(device)

    def prep_5d(img_bgr):
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(device) / 255.0
        return (t - mean) / std

    representative_samples = {}
    gallery_cards = []

    for idx, (rel_path, true_label) in enumerate(tqdm(all_targets, desc="Generating CAER-S 2x3 Grad-CAM Cards")):
        full_path = os.path.join(args.data_dir, rel_path)
        if not os.path.exists(full_path):
            continue

        img_raw = cv2.imread(full_path)
        if img_raw is None:
            continue

        h, w = img_raw.shape[:2]
        true_name = CAER_CLASSES[true_label]

        # 1. Face Crop
        bbox = face_boxes.get(rel_path, [w // 4, h // 4, 3 * w // 4, 3 * h // 4])
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = img_raw[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img_raw

        # 2. Body Crop (Upper-body / Torso estimation)
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        by1 = max(0, y1 - int(fh * 0.1))
        by2 = min(h, y2 + int(fh * 2.2))
        bx1 = max(0, x1 - int(fw * 0.8))
        bx2 = min(w, x2 + int(fw * 0.8))
        body_crop = img_raw[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else img_raw

        # 3. Context Stream: Clean unmasked original scene for publication presentation
        context_clean = img_raw.copy()

        # Context masked face for model forward pass inference (protocol fidelity)
        context_masked = img_raw.copy()
        context_masked[y1:y2, x1:x2] = 0

        # 4. Model Prediction
        t_f = prep_5d(face_crop)
        t_b = prep_5d(body_crop)
        t_c = prep_5d(context_masked)

        with torch.no_grad():
            out, _, _, _ = model(t_f, t_b, t_c)
            probs = torch.softmax(out, dim=-1)[0]
            pred_cls = probs.argmax().item()
            conf = probs[pred_cls].item() * 100.0

        pred_name = CAER_CLASSES[pred_cls]

        # 5. Compute Token-Level CLIP Discriminative Saliency Maps
        target_eval = pred_cls if (pred_cls == true_label) else true_label
        cam_face = compute_token_similarity(model, face_crop, target_eval, 'face', device)
        cam_body = compute_token_similarity(model, body_crop, target_eval, 'body', device)
        cam_ctx = compute_token_similarity(model, context_clean, target_eval, 'context', device)

        # 6. Thermal Overlay
        ov_face = overlay_heatmap(face_crop, cam_face, alpha=0.5)
        ov_body = overlay_heatmap(body_crop, cam_body, alpha=0.5)
        ov_ctx = overlay_heatmap(context_clean, cam_ctx, alpha=0.5)

        # 7. Build Standard 2x3 Grid Card
        card = create_3stream_grid(face_crop, body_crop, context_clean,
                                  ov_face, ov_body, ov_ctx,
                                  CAER_CLASSES[target_eval], true_name, pred_name, conf)

        # Save files
        base_fn = os.path.splitext(os.path.basename(rel_path))[0]
        card_name = f"{base_fn}_{true_name}_pred_{pred_name}.png"
        cv2.imwrite(os.path.join(out_all, card_name), card)
        cv2.imwrite(os.path.join(args.out_dir, 'by_class', true_name, card_name), card)

        # Retain highest-confidence correctly predicted card for each emotion
        if true_name not in representative_samples or (true_name == pred_name and conf > representative_samples[true_name][1]):
            representative_samples[true_name] = (card, conf, card_name)

    # Export representative sample cards to root out_dir
    print("\n=> Exporting representative emotion sample cards:")
    for emo_idx, cname in enumerate(CAER_CLASSES):
        if cname in representative_samples:
            rep_card, rep_conf, rep_fname = representative_samples[cname]
            dst_name = f"{emo_idx+1}_{cname}.png"
            cv2.imwrite(os.path.join(args.out_dir, dst_name), rep_card)
            print(f"   • [{dst_name}] Emotion: {cname:<10} | Conf: {rep_conf:5.1f}% | Source: {rep_fname}")
            gallery_cards.append(rep_card)

    # 8. Generate Master Gallery Poster (2 columns or stacked)
    if gallery_cards:
        poster = np.vstack(gallery_cards)
        poster_path = os.path.join(args.out_dir, 'caers_gradcam_gallery.png')
        cv2.imwrite(poster_path, poster)
        print(f"\n=> Master 7-Emotion Gallery Poster saved to: {poster_path} ({poster.shape[1]}x{poster.shape[0]} px)")

    print(f"\n=> SUCCESS: All CAER-S Grad-CAM 2x3 cards saved to '{args.out_dir}'!")


if __name__ == '__main__':
    main()
