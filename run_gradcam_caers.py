#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_gradcam_caers.py
───────────────────
Full Qualitative Grad-CAM & Multimodal Attention Visualization for RAPT-CLIP on CAER-S.
Model: Caers-finetune.pth (90.40% WAR / UAR SOTA Checkpoint)

Generates publication-quality 4-stream cards:
[ (a) Original Scene (with BBoxes) | (b) Face Saliency | (c) Body Posture Saliency | (d) Context Saliency ]

Features:
- Precise Face cropping via CAER-S face bounding boxes
- Anatomical Upper-Body / Torso region estimation
- Scene Context with facial suppression (matching training preprocessing)
- Full model inference (Face + Body + Context via CMAF Frame-Level Fusion)
- Multi-layer visual attention thermal overlay (hot red / jet colormap)
- Class-organized exports & a master 7-emotion comparison poster
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
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
    parser.add_argument('--samples-per-class', type=int, default=5,
                        help='Number of test samples to visualize per emotion class')
    parser.add_argument('--out-dir', type=str, default='outputs/Gradcam-Caers',
                        help='Output directory for Grad-CAM cards')
    parser.add_argument('--device', type=str, default='auto',
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
    # Default to CPU for maximum stability and float precision across vision hooks
    return torch.device('cpu')


def compute_thermal_map(model, img_bgr, device, sigma=20.0):
    """Extracts multi-layer visual attention saliency and computes smooth thermal map."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
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
        x = inputs[0]  # (197, 1, 768)
        feats.append(x[1:, 0])  # (196, 768)
        w_q, w_k, _ = block.attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, _ = block.attn.in_proj_bias.chunk(3, dim=0)
        q = (x @ w_q.t() + b_q).view(197, 1, 12, 64).permute(1, 2, 0, 3)
        k = (x @ w_k.t() + b_k).view(197, 1, 12, 64).permute(1, 2, 0, 3)
        a = torch.softmax((q @ k.transpose(-2, -1)) / 8.0, dim=-1)[0, :, 0, 1:].mean(dim=0)
        attns.append(a)

    attn_stack = torch.stack(attns).mean(dim=0)
    feat_stack = torch.stack(feats).mean(dim=0).norm(dim=-1)
    saliency = (attn_stack * feat_stack).detach().cpu().numpy().reshape(14, 14)

    up = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (51, 51), sigmaX=sigma)
    b_min, b_max = blurred.min(), blurred.max()
    cam_norm = np.clip((blurred - b_min) / (b_max - b_min + 1e-6), 0, 1)
    return cam_norm


def overlay_thermal(img_bgr, cam_norm):
    """Blends vibrant thermal JET colormap with adaptive alpha."""
    heat = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET).astype(np.float32) / 255.0
    img_f = img_bgr.astype(np.float32) / 255.0
    alpha = (cam_norm ** 1.1) * 0.58 + 0.12
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    blend = (1.0 - alpha_3d) * img_f + alpha_3d * heat
    return np.uint8(255 * np.clip(blend, 0, 1))


def build_single_card(tile_orig, tile_face_ov, tile_body_ov, tile_ctx_ov,
                      true_name, pred_name, conf, sample_idx):
    """
    Builds a professional 4-column card:
    [ (a) Original Scene | (b) Face Saliency | (c) Body Posture | (d) Context Saliency ]
    """
    tile_size = 224
    gap = 12
    pad_top = 44
    pad_bot = 42
    card_w = tile_size * 4 + gap * 5
    card_h = pad_top + tile_size + pad_bot

    card = np.ones((card_h, card_w, 3), dtype=np.uint8) * 255

    cols = ['(a) Original Scene', '(b) Face Saliency', '(c) Body Posture', '(d) Context Saliency']
    font = cv2.FONT_HERSHEY_DUPLEX
    for i, title in enumerate(cols):
        tw = cv2.getTextSize(title, font, 0.44, 1)[0][0]
        x = gap + i * (tile_size + gap) + (tile_size - tw) // 2
        cv2.putText(card, title, (x, 28), font, 0.44, (30, 30, 30), 1, cv2.LINE_AA)

    y = pad_top
    tiles = [tile_orig, tile_face_ov, tile_body_ov, tile_ctx_ov]
    for i, tile in enumerate(tiles):
        x = gap + i * (tile_size + gap)
        card[y:y+tile_size, x:x+tile_size] = cv2.resize(tile, (tile_size, tile_size))
        cv2.rectangle(card, (x, y), (x+tile_size, y+tile_size), (210, 210, 210), 1)

    match = (true_name == pred_name)
    status_color = (25, 135, 25) if match else (25, 25, 205)
    symbol = "[CORRECT]" if match else "[MISMATCH]"
    info = f"Sample #{sample_idx:03d} | Ground Truth: {true_name} | Predicted: {pred_name} ({conf:.1f}%) {symbol}"
    cv2.putText(card, info, (gap + 4, card_h - 15), font, 0.45, status_color, 1, cv2.LINE_AA)

    return card


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"=> Execution Device: {device}")
    print(f"=> Checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'")

    model_args = argparse.Namespace(
        dataset='CAER-S',
        clip_path='ViT-B/16',
        drop_path_rate=0.0,
        lr_image_encoder=1e-6,
        freeze_image_encoder=False,
        text_type='prompt_ensemble',
        contexts_number=8,
        class_token_position='end',
        class_specific_contexts=True,
        use_context=True,
        crop_body=True,
        mask_context_face=True,
        fusion_type='cmaf',
        fusion_stage='frame',
        temporal_layers=1,
        temporal_pooling='attn',
        load_and_tune_prompt_learner=True,
        image_size=224,
        temperature=0.07
    )

    print("=> Loading CLIP ViT-B/16 and RAPT-CLIP CAER-S Architecture...")
    clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_model = clip_model.float()
    class_names, class_descriptors = get_class_info(model_args)

    model = GenerateModel(class_descriptors, clip_model, model_args).to(device).float()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"=> Loaded checkpoint successfully (Unexpected keys: {len(unexpected)}, Missing keys: {len(missing)})")
    model.eval()

    # Load Face Bounding Boxes
    if os.path.exists(args.face_box):
        with open(args.face_box, 'r') as f:
            face_boxes = json.load(f)
        print(f"=> Loaded {len(face_boxes):,} Face boxes from '{args.face_box}'")
    else:
        face_boxes = {}
        print(f"=> [Warning] Face boxes file '{args.face_box}' not found. Fallback to center-crop.")

    # Select representative samples per class from test set
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
        # Fallback to direct directory scan
        for idx, cname in enumerate(CAER_CLASSES):
            cdir = os.path.join(args.data_dir, 'test', cname)
            if os.path.exists(cdir):
                fnames = sorted([f for f in os.listdir(cdir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for fn in fnames[:args.samples_per_class]:
                    samples_by_class[cname].append((os.path.join('test', cname, fn), idx))

    all_targets = []
    for cname in CAER_CLASSES:
        all_targets.extend(samples_by_class[cname])

    print(f"=> Total samples to visualize: {len(all_targets)} ({args.samples_per_class} per class across 7 emotions)")

    # Prepare directories
    out_all = os.path.join(args.out_dir, 'all_samples')
    os.makedirs(out_all, exist_ok=True)
    for cname in CAER_CLASSES:
        os.makedirs(os.path.join(args.out_dir, 'by_class', cname), exist_ok=True)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1).to(device)

    def prep_5d_tensor(img_bgr):
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(device) / 255.0
        return (t - mean) / std

    gallery_cards = []
    representative_samples = {}

    for idx, (rel_path, true_label) in enumerate(tqdm(all_targets, desc="Generating CAER-S Grad-CAM Cards")):
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

        # 3. Context Stream (with masked face)
        context_img = img_raw.copy()
        context_img[y1:y2, x1:x2] = 0

        # Draw bounding boxes on Original Scene tile
        orig_with_boxes = img_raw.copy()
        cv2.rectangle(orig_with_boxes, (bx1, by1), (bx2, by2), (255, 160, 0), 2)  # Blue/Cyan for Body
        cv2.rectangle(orig_with_boxes, (x1, y1), (x2, y2), (0, 220, 0), 2)       # Green for Face

        # 4. Model Prediction
        t_f = prep_5d_tensor(face_crop)
        t_b = prep_5d_tensor(body_crop)
        t_c = prep_5d_tensor(context_img)

        with torch.no_grad():
            out, _, _, _ = model(t_f, t_b, t_c)
            probs = torch.softmax(out, dim=-1)[0]
            pred_cls = probs.argmax().item()
            conf = probs[pred_cls].item() * 100.0

        pred_name = CAER_CLASSES[pred_cls]

        # 5. Compute Visual Attention Grad-CAM Maps
        cam_face = compute_thermal_map(model, face_crop, device, sigma=18.0)
        cam_body = compute_thermal_map(model, body_crop, device, sigma=20.0)
        cam_ctx = compute_thermal_map(model, context_img, device, sigma=24.0)

        ov_face = overlay_thermal(face_crop, cam_face)
        ov_body = overlay_thermal(body_crop, cam_body)
        ov_ctx = overlay_thermal(context_img, cam_ctx)

        # 6. Build High-Resolution Card
        card = build_single_card(orig_with_boxes, ov_face, ov_body, ov_ctx,
                                 true_name, pred_name, conf, idx + 1)

        # Save files
        card_name = f"sample_{idx+1:03d}_{true_name}_pred_{pred_name}.png"
        cv2.imwrite(os.path.join(out_all, card_name), card)
        cv2.imwrite(os.path.join(args.out_dir, 'by_class', true_name, card_name), card)

        # Keep one top-confidence correct card per emotion for the gallery poster
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

    # 7. Generate Master Gallery Poster
    if gallery_cards:
        poster = np.vstack(gallery_cards)
        poster_path = os.path.join(args.out_dir, 'caers_gradcam_gallery.png')
        cv2.imwrite(poster_path, poster)
        print(f"\n=> Master 7-Emotion Gallery Poster saved to: {poster_path} ({poster.shape[1]}x{poster.shape[0]} px)")

    print(f"\n=> SUCCESS: Generated all CAER-S Grad-CAM cards in '{args.out_dir}'!")


if __name__ == '__main__':
    main()
