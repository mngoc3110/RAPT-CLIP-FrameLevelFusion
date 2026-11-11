#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_50_gradcam_cards_per_benchmark.py
──────────────────────────────────────────
Generates ~50 high-definition publication-ready GradCAM cards for each of the 3 benchmarks:
1. CAER-S (50 cards, ~7-8 per class across 7 emotion classes)
2. RAER (50 cards, exactly 10 per class across 5 engagement classes)
3. EMOTIC (50 cards, diverse selection across 26 continuous emotions)

All cards feature:
- Row 1: Original (Face, Faceless Body with matte dark mask, Context)
- Row 2: GradCAM (Face with glowing eyes/mouth/brow, Body with glowing arms/torso/hands + matte mask on face, Context with glowing ambient scene)
- Status badge (CORRECT PREDICTION / MISCLASSIFICATION), Ground truth, Prediction + Confidence.
Saved into: outputs/gradcam_faceless_50_samples/{CAER-S, RAER, EMOTIC}/
"""

import os
import sys
import json
import glob
import cv2
import numpy as np
import torch
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.generate_publication_samples import (
    compute_vivid_cam, overlay_heatmap, build_card,
    mask_face_on_body, get_clean_active_context, detector, device
)
from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info

DEST_BASE = os.path.join(ROOT_DIR, 'outputs', 'gradcam_faceless_50_samples')
DIR_CAERS = os.path.join(DEST_BASE, 'CAER-S')
DIR_RAER  = os.path.join(DEST_BASE, 'RAER')
DIR_EMOTIC = os.path.join(DEST_BASE, 'EMOTIC')

for d in [DIR_CAERS, DIR_RAER, DIR_EMOTIC]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1. CAER-S GENERATOR (50 SAMPLES)
# ══════════════════════════════════════════════════════════════════
def run_caers_50():
    print("\n" + "="*60)
    print("  RUNNING CAER-S (50 SAMPLES) ->", DIR_CAERS)
    print("="*60)

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
    ckpt_path = os.path.join(ROOT_DIR, 'caer_s_finetune_end.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT_DIR, 'Caers-finetune.pth')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()

    with open(os.path.join(ROOT_DIR, 'CAER-S/annotations/caer_s_faces.json')) as f:
        face_boxes = json.load(f)

    with open(os.path.join(ROOT_DIR, 'CAER-S/annotations/test.txt')) as f:
        test_lines = [l.strip().split() for l in f if l.strip()]

    CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    by_class = {i: [] for i in range(7)}
    for rel_p, cls_s in test_lines:
        c_idx = int(cls_s)
        full_p = os.path.join(ROOT_DIR, 'CAER-S', rel_p)
        if rel_p in face_boxes and os.path.exists(full_p):
            by_class[c_idx].append(rel_p)

    # Distribute 50 samples: 7 per class (classes 0..6 = 49) + 1 extra for Happy = 50
    selected_items = []
    quota = [7, 7, 7, 8, 7, 7, 7]
    for c_idx in range(7):
        pool = by_class[c_idx]
        step = max(1, len(pool) // quota[c_idx])
        picked = [pool[i * step] for i in range(quota[c_idx])]
        for p in picked:
            selected_items.append((p, c_idx))

    print(f"=> Sampled {len(selected_items)} CAER-S test images across 7 classes.")

    count = 0
    for idx, (rel_p, true_cls) in enumerate(selected_items, 1):
        full_p = os.path.join(ROOT_DIR, 'CAER-S', rel_p)
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

        # Forward pass for genuine prediction
        with torch.no_grad():
            f_in = torch.from_numpy(cv2.resize(face_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
            b_in = torch.from_numpy(cv2.resize(body_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
            c_in = torch.from_numpy(cv2.resize(ctx_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(1).float().to(device) / 255.0
            # Normalize with CLIP mean/std
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)
            f_in, b_in, c_in = (f_in - mean) / std, (b_in - mean) / std, (c_in - mean) / std
            out = model(f_in, b_in, c_in)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=-1)
            pred_cls = probs.argmax().item()
            conf = probs[0, pred_cls].item() * 100.0

        is_correct = (pred_cls == true_cls)
        target_eval = pred_cls if is_correct else true_cls

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

        true_t = CAER_CLASSES[true_cls]
        pred_t = f"{CAER_CLASSES[pred_cls]} ({conf:.1f}%)"
        att_t  = CAER_CLASSES[target_eval]

        card = build_card(face_orig, body_orig_masked, ctx_orig,
                          ov_f, ov_b, ov_c,
                          att_t, true_t, pred_t, is_correct=is_correct)

        b_name = os.path.basename(rel_p).replace('.png', '')
        out_name = f"caers_{idx:02d}_{true_t}_{b_name}.png"
        save_p = os.path.join(DIR_CAERS, out_name)
        cv2.imwrite(save_p, card)
        count += 1
        if count % 10 == 0 or count == len(selected_items):
            print(f"  [CAER-S] Saved {count}/{len(selected_items)} cards (latest: {out_name})")


# ══════════════════════════════════════════════════════════════════
# 2. RAER GENERATOR (50 SAMPLES)
# ══════════════════════════════════════════════════════════════════
def run_raer_50():
    print("\n" + "="*60)
    print("  RUNNING RAER (50 SAMPLES) ->", DIR_RAER)
    print("="*60)

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

    with open(os.path.join(ROOT_DIR, 'RAER/bounding_box/body.json')) as f:
        body_boxes = json.load(f)
    with open(os.path.join(ROOT_DIR, 'RAER/bounding_box/face.json')) as f:
        face_boxes = json.load(f)

    # Gather test lines + val lines for rare Confusion class
    lines_all = []
    for fn in ['RAER/annotation/test.txt', 'RAER/annotation/val_20.txt']:
        p = os.path.join(ROOT_DIR, fn)
        if os.path.exists(p):
            with open(p) as f:
                lines_all.extend([l.strip().split() for l in f if l.strip()])

    CLASSES = ['Neutral', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
    by_class = {i: [] for i in range(5)}
    for vp, nframes, lbl in lines_all:
        c_idx = int(lbl) - 1
        clean_k = vp.replace('./', '')
        has_b = clean_k in body_boxes or ('RAER/' + clean_k) in body_boxes
        has_f = clean_k in face_boxes or ('RAER/' + clean_k) in face_boxes
        full_dir = os.path.join(ROOT_DIR, vp)
        if has_b and has_f and os.path.exists(full_dir):
            if vp not in by_class[c_idx]:
                by_class[c_idx].append(vp)

    # Pick exactly 10 samples per class = 50 samples
    selected_items = []
    for c_idx in range(5):
        pool = by_class[c_idx]
        step = max(1, len(pool) // 10)
        picked = pool[:10] if len(pool) <= 10 else [pool[i * step] for i in range(10)]
        for vp in picked[:10]:
            selected_items.append((vp, c_idx))

    print(f"=> Sampled {len(selected_items)} RAER videos (10 per class x 5 classes).")

    count = 0
    for idx, (vp, true_cls) in enumerate(selected_items, 1):
        full_dir = os.path.join(ROOT_DIR, vp)
        f_cand = os.path.join(full_dir, '15.jpg') if os.path.exists(os.path.join(full_dir, '15.jpg')) else os.path.join(full_dir, '1.jpg')
        if not os.path.exists(f_cand):
            jpgs = [f for f in os.listdir(full_dir) if f.endswith('.jpg')]
            if not jpgs:
                continue
            f_cand = os.path.join(full_dir, sorted(jpgs)[len(jpgs)//2])
        chosen_frame = os.path.basename(f_cand)
        raw_img = cv2.imread(f_cand)
        if raw_img is None:
            continue
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

        # Forward pass for model prediction
        with torch.no_grad():
            f_in = torch.from_numpy(cv2.resize(face_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1).float().to(device) / 255.0
            b_in = torch.from_numpy(cv2.resize(body_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1).float().to(device) / 255.0
            c_in = torch.from_numpy(cv2.resize(ctx_orig, (224, 224))).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1).float().to(device) / 255.0
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)
            f_in, b_in, c_in = (f_in - mean) / std, (b_in - mean) / std, (c_in - mean) / std
            out = model(f_in, b_in, c_in)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=-1)
            pred_cls = probs.argmax().item()
            conf = probs[0, pred_cls].item() * 100.0

        is_correct = (pred_cls == true_cls)
        target_eval = pred_cls if is_correct else true_cls

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

        true_t = CLASSES[true_cls]
        pred_t = f"{CLASSES[pred_cls]} ({conf:.1f}%)"
        att_t  = CLASSES[target_eval]

        card = build_card(face_orig, body_orig_masked, ctx_orig,
                          ov_f, ov_b, ov_c,
                          att_t, true_t, pred_t, is_correct=is_correct)

        safe_vp = vp.replace('/', '_').replace('RAER_images_', '')
        out_name = f"raer_{idx:02d}_{true_t}_{safe_vp}.png"
        save_p = os.path.join(DIR_RAER, out_name)
        cv2.imwrite(save_p, card)
        count += 1
        if count % 10 == 0 or count == len(selected_items):
            print(f"  [RAER] Saved {count}/{len(selected_items)} cards (latest: {out_name})")


# ══════════════════════════════════════════════════════════════════
# 3. EMOTIC GENERATOR (50 SAMPLES)
# ══════════════════════════════════════════════════════════════════
def run_emotic_50():
    print("\n" + "="*60)
    print("  RUNNING EMOTIC (50 SAMPLES) ->", DIR_EMOTIC)
    print("="*60)

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
    ckpt_path = os.path.join(ROOT_DIR, 'outputs/EMOTIC/model_best.zip')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    with open(os.path.join(ROOT_DIR, 'emotic_dataset/test_bbox.txt')) as f:
        e_lines = [l.strip().split() for l in f if l.strip()]

    # Collect valid samples that physically exist
    valid_emotic = []
    for rel_p, lbl_str, x1, y1, x2, y2 in e_lines:
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', rel_p)
        if os.path.exists(full_p):
            lbls = [int(x) for x in lbl_str.split(',')]
            box = [int(x1), int(y1), int(x2), int(y2)]
            active_cls = [i for i, v in enumerate(lbls) if v == 1]
            if active_cls:
                valid_emotic.append((rel_p, active_cls, box))

    # Pick 50 diverse samples across the dataset
    step = max(1, len(valid_emotic) // 50)
    selected_items = [valid_emotic[i * step] for i in range(50)]
    print(f"=> Sampled {len(selected_items)} EMOTIC images across diverse continuous categories.")

    count = 0
    for idx, (rel_p, active_classes, body_box) in enumerate(selected_items, 1):
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', rel_p)
        raw_img = cv2.imread(full_p)
        if raw_img is None:
            continue
        h, w = raw_img.shape[:2]

        bx1, by1, bx2, by2 = body_box
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        body_orig = raw_img[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else raw_img

        # Detect face inside body region using YuNet
        fx1, fy1, fx2, fy2 = None, None, None, None
        face_orig = None
        if detector is not None:
            detector.setInputSize((w, h))
            _, faces = detector.detect(raw_img)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    fx, fy, fw, fh = [int(v) for v in face[:4]]
                    if bx1 - 20 <= fx and fx + fw <= bx2 + 20 and by1 - 20 <= fy:
                        pad = int(0.15 * fh)
                        fx1, fy1 = max(0, fx - pad), max(0, fy - pad)
                        fx2, fy2 = min(w, fx + fw + pad), min(h, fy + fh + pad)
                        face_orig = raw_img[fy1:fy2, fx1:fx2]
                        break

        # Fallback to upper 35% of body box
        if face_orig is None or face_orig.size == 0:
            fh = int((by2 - by1) * 0.35)
            fw = int((bx2 - bx1) * 0.50)
            cx = (bx1 + bx2) // 2
            fx1, fy1 = max(0, cx - fw // 2), by1
            fx2, fy2 = min(w, cx + fw // 2), min(h, by1 + fh)
            face_orig = raw_img[fy1:fy2, fx1:fx2] if (fx2 > fx1 and fy2 > fy1) else body_orig

        ctx_orig = raw_img.copy()

        # Faceless Body stream preparation
        if fx1 is not None and fy1 is not None:
            rel_f_box = [max(0, fx1 - bx1), max(0, fy1 - by1),
                         min(body_orig.shape[1], fx2 - bx1), min(body_orig.shape[0], fy2 - by1)]
        else:
            rel_f_box = None
        body_orig_masked, m_box = mask_face_on_body(body_orig, rel_f_box)

        # Forward pass for model prediction (Multi-label Sigmoid)
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
            # True multi-label dynamic thresholding (variable label counts: 1 to 5+)
            threshold_val = 0.55
            pred_indices = [c for c in range(len(class_names)) if probs[c].item() >= threshold_val]
            if not pred_indices:
                pred_indices = [probs.argmax().item()]
            pred_indices = sorted(pred_indices, key=lambda c: probs[c].item(), reverse=True)
            top_cls = pred_indices[0]

        target_cls = active_classes[0] if active_classes[0] in pred_indices else top_cls
        is_correct = any(c in active_classes for c in pred_indices)

        cam_f = compute_vivid_cam(model, face_orig, target_cls, 'face')
        cam_b = compute_vivid_cam(model, body_orig, target_cls, 'body')
        cam_c = compute_vivid_cam(model, ctx_orig, target_cls, 'context', body_box=[bx1, by1, bx2, by2])

        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig_masked, cam_b)
        if m_box is not None:
            m_x1, m_y1, m_x2, m_y2 = m_box
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (18, 18, 18), -1)
            cv2.rectangle(ov_b, (m_x1, m_y1), (m_x2, m_y2), (75, 75, 75), 1)
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        # Full ground-truth and multi-label predictions (No artificial [:3] clipping!)
        true_names = [class_names[c] for c in active_classes]
        pred_names = [f"{class_names[c]} ({probs[c].item():.2f})" for c in pred_indices]
        true_t = ", ".join(true_names)
        pred_t = ", ".join(pred_names)
        att_t  = class_names[target_cls]

        card = build_card(face_orig, body_orig_masked, ctx_orig,
                          ov_f, ov_b, ov_c,
                          att_t, true_t, pred_t, is_correct=is_correct)

        b_name = os.path.basename(rel_p).replace('.jpg', '')
        out_name = f"emotic_{idx:02d}_{att_t}_{b_name}.png"
        save_p = os.path.join(DIR_EMOTIC, out_name)
        cv2.imwrite(save_p, card)
        count += 1
        if count % 10 == 0 or count == len(selected_items):
            print(f"  [EMOTIC] Saved {count}/{len(selected_items)} cards (latest: {out_name})")


# ══════════════════════════════════════════════════════════════════
# 4. MASTER HTML OVERVIEW INDEX BUILDER
# ══════════════════════════════════════════════════════════════════
def build_html_gallery():
    print("\n" + "="*60)
    print("  BUILDING INTERACTIVE HTML VISUALIZATION GALLERY")
    print("="*60)
    html_p = os.path.join(DEST_BASE, 'index.html')

    c_cards = sorted([f for f in os.listdir(DIR_CAERS) if f.endswith('.png')])
    r_cards = sorted([f for f in os.listdir(DIR_RAER) if f.endswith('.png')])
    e_cards = sorted([f for f in os.listdir(DIR_EMOTIC) if f.endswith('.png')])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RAPT-CLIP: 150 Tri-Stream GradCAM Evaluation Cards</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f1117; color: #e6edf3; margin: 0; padding: 24px; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; color: #58a6ff; }}
  p.subtitle {{ color: #8b949e; font-size: 15px; margin-bottom: 24px; }}
  .tabs {{ display: flex; gap: 12px; margin-bottom: 24px; border-bottom: 1px solid #30363d; padding-bottom: 12px; }}
  .tab-btn {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: 600; transition: all 0.2s; }}
  .tab-btn.active {{ background: #1f6feb; color: #fff; border-color: #58a6ff; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 18px; }}
  .card-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; transition: transform 0.2s, border-color 0.2s; }}
  .card-box:hover {{ transform: translateY(-3px); border-color: #58a6ff; }}
  .card-box img {{ width: 100%; display: block; }}
  .card-title {{ padding: 10px 14px; font-size: 13px; color: #8b949e; font-weight: 500; border-top: 1px solid #21262d; }}
</style>
</head>
<body>
  <h1>RAPT-CLIP Publication GradCAM Gallery</h1>
  <p class="subtitle">150 Comprehensive Tri-Stream Cards across 3 Benchmarks with strict Faceless Body Masking and ambient Context isolation.</p>
  
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('caers')">CAER-S Benchmark ({len(c_cards)} Cards)</button>
    <button class="tab-btn" onclick="showTab('raer')">RAER Benchmark ({len(r_cards)} Cards)</button>
    <button class="tab-btn" onclick="showTab('emotic')">EMOTIC Benchmark ({len(e_cards)} Cards)</button>
  </div>

  <div id="tab-caers" class="tab-content active">
    {"".join(f'<div class="card-box"><a href="CAER-S/{c}" target="_blank"><img src="CAER-S/{c}" loading="lazy"/></a><div class="card-title">{c}</div></div>' for c in c_cards)}
  </div>

  <div id="tab-raer" class="tab-content">
    {"".join(f'<div class="card-box"><a href="RAER/{c}" target="_blank"><img src="RAER/{c}" loading="lazy"/></a><div class="card-title">{c}</div></div>' for c in r_cards)}
  </div>

  <div id="tab-emotic" class="tab-content">
    {"".join(f'<div class="card-box"><a href="EMOTIC/{c}" target="_blank"><img src="EMOTIC/{c}" loading="lazy"/></a><div class="card-title">{c}</div></div>' for c in e_cards)}
  </div>

<script>
function showTab(tabName) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + tabName).classList.add('active');
}}
</script>
</body>
</html>
"""
    with open(html_p, 'w') as f:
        f.write(html_content)
    print(f"  [HTML Gallery] Saved interactive gallery to: {html_p}")


def main():
    print(f"=> Executing on device: {device}")
    run_caers_50()
    run_raer_50()
    run_emotic_50()
    build_html_gallery()
    print("\n===> ALL 150 CARDS (50 CAER-S + 50 RAER + 50 EMOTIC) SUCCESSFULLY CREATED!")


if __name__ == '__main__':
    main()
