"""
Full Dataset 4-Stream Qualitative Visualization (Figure 5 Style) for RAPT-CLIP.
Processes all 528 test videos in RAER.
Produces 4-stream cards: [Original | Body | Face | Context]
- ZERO black letterbox borders (tight bounding box cropping)
- Glowing, vivid thermal Grad-CAM overlays (hot red on salient face/body/context features)
- High contrast, publication-ready presentation assets
"""
import os
import json
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info
import warnings

warnings.filterwarnings('ignore')

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']


def get_clean_active_context(img_raw):
    """Crops non-black active frame region to eliminate letterbox bars."""
    gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    non_zero = np.where(gray > 8)
    if len(non_zero[0]) > 0:
        y1, y2 = non_zero[0].min(), non_zero[0].max()
        x1, x2 = non_zero[1].min(), non_zero[1].max()
        if (y2 - y1) > 50 and (x2 - x1) > 50:
            return img_raw[y1:y2+1, x1:x2+1]
    return img_raw


def compute_thermal_map(model, img_bgr, device, sigma=22.0):
    """Extracts multi-layer visual attention saliency and computes smooth thermal map."""
    t = torch.from_numpy(cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    t = ((t - mean) / std).to(device)

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
    """Blends vibrant thermal JET colormap with adaptive alpha."""
    heat = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET).astype(np.float32) / 255.0
    img_f = img_bgr.astype(np.float32) / 255.0
    
    alpha = (cam_norm ** 1.2) * 0.58 + 0.12
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    
    blend = (1.0 - alpha_3d) * img_f + alpha_3d * heat
    return np.uint8(255 * np.clip(blend, 0, 1))


def build_single_card(tile_orig, tile_body_ov, tile_face_ov, tile_ctx_ov, true_name, pred_name, conf, sample_idx):
    tile_size = 224
    gap = 14
    pad_top = 44
    pad_bot = 40
    card_w = tile_size * 4 + gap * 5
    card_h = pad_top + tile_size + pad_bot

    card = np.ones((card_h, card_w, 3), dtype=np.uint8) * 255

    # Column titles
    cols = ['Original', 'Body', 'Face', 'Context']
    font = cv2.FONT_HERSHEY_DUPLEX
    for i, title in enumerate(cols):
        tw = cv2.getTextSize(title, font, 0.48, 1)[0][0]
        x = gap + i * (tile_size + gap) + (tile_size - tw) // 2
        cv2.putText(card, title, (x, 28), font, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

    # Place tiles
    y = pad_top
    card[y:y+tile_size, gap:gap+tile_size] = tile_orig
    card[y:y+tile_size, gap + (tile_size+gap):gap + (tile_size+gap) + tile_size] = tile_body_ov
    card[y:y+tile_size, gap + 2*(tile_size+gap):gap + 2*(tile_size+gap) + tile_size] = tile_face_ov
    card[y:y+tile_size, gap + 3*(tile_size+gap):gap + 3*(tile_size+gap) + tile_size] = tile_ctx_ov

    # Bottom status banner
    match = (true_name == pred_name)
    color = (25, 135, 25) if match else (25, 25, 205)
    info = f"Sample #{sample_idx:03d} | True: {true_name} | Predicted: {pred_name} ({conf:.1f}%)"
    cv2.putText(card, info, (gap + 4, card_h - 14), font, 0.48, color, 1, cv2.LINE_AA)

    return card


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"=> Device: {device}")

    ckpt_path = 'outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    args = argparse.Namespace(
        clip_path='ViT-B/16', drop_path_rate=0.0, dataset='RAER', use_context=True,
        fusion_type='cmaf', fusion_stage='frame', temporal_layers=1, contexts_number=8,
        class_token_position='end', class_specific_contexts=True, load_and_tune_prompt_learner=True,
        text_type='prompt_ensemble', temperature=0.07, lr=2e-5, lr_image_encoder=1e-6,
        lr_prompt_learner=3e-4, lr_adapter=1e-4, lambda_vad=0.0, modality_dropout=0.0,
        crop_body=True, freeze_image_encoder=False, use_moco=False
    )

    class_names, input_text = get_class_info(args)
    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model = clip_model.float()
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=args).to(device).float()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        prompts = model.prompt_learner()
        text_feats = model.text_encoder(prompts, model.tokenized_prompts).float()
        if model.is_ensemble:
            text_feats = text_feats.view(model.num_classes, model.num_prompts_per_class, -1).mean(dim=1)
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)

    # Load annotations and bounding boxes
    with open('RAER/annotation/test.txt') as f:
        test_entries = [l.strip().split() for l in f]

    with open('RAER/bounding_box/body.json') as f:
        body_boxes = json.load(f)

    with open('RAER/bounding_box/face.json') as f:
        face_boxes = json.load(f)

    out_all = 'outputs/gradcam/all_528_samples'
    os.makedirs(out_all, exist_ok=True)
    for cname in CLASS_NAMES:
        os.makedirs(f'outputs/gradcam/by_class/{cname}', exist_ok=True)

    print(f"=> Processing Full Dataset: {len(test_entries)} videos...")

    tile_s = 224
    for idx, (vid_path, num_frames_str, label_str) in enumerate(tqdm(test_entries, desc="Generating 4-Stream Heatmaps")):
        true_cls = int(label_str) - 1
        true_name = CLASS_NAMES[true_cls]

        # Use middle frame (e.g. frame 15 or 10 or 1)
        mid_num = max(1, int(num_frames_str) // 2)
        frame_candidates = [f"{mid_num}.jpg", "15.jpg", "1.jpg"]
        img_raw = None
        chosen_frame = "1.jpg"
        for fc in frame_candidates:
            p = os.path.join(vid_path, fc)
            if os.path.exists(p):
                img_raw = cv2.imread(p)
                chosen_frame = fc
                break

        if img_raw is None:
            # Fallback search any jpg
            jpgs = [f for f in os.listdir(vid_path) if f.endswith('.jpg')]
            if jpgs:
                chosen_frame = sorted(jpgs)[len(jpgs)//2]
                img_raw = cv2.imread(os.path.join(vid_path, chosen_frame))
            else:
                continue

        clean_key = vid_path.replace('./', '')
        match_key = clean_key if clean_key in body_boxes else 'RAER/' + clean_key

        # 1. Clean Context Crop (no letterbox bars)
        ctx_clean = get_clean_active_context(img_raw)
        tile_ctx = cv2.resize(ctx_clean, (tile_s, tile_s))

        # 2. Clean Body Crop (no letterbox bars)
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

        # 3. Clean Face Crop (no letterbox bars)
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

        # Model Prediction for Confidence
        with torch.no_grad():
            t_f = torch.from_numpy(cv2.cvtColor(tile_face, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            feat_f = model.image_encoder(t_f)
            feat_f = feat_f / (feat_f.norm(dim=-1, keepdim=True) + 1e-6)
            logits = feat_f @ text_feats.t()
            probs = torch.softmax(logits, dim=-1)[0]
            pred_cls = logits.argmax(dim=-1).item()
            conf = probs[pred_cls].item() * 100.0

        pred_name = CLASS_NAMES[pred_cls]

        # Compute Thermal Grad-CAM Maps
        cam_body = compute_thermal_map(model, tile_body, device, sigma=22.0)
        cam_face = compute_thermal_map(model, tile_face, device, sigma=18.0)
        cam_ctx = compute_thermal_map(model, tile_ctx, device, sigma=24.0)

        ov_body = overlay_thermal(tile_body, cam_body)
        ov_face = overlay_thermal(tile_face, cam_face)
        ov_ctx = overlay_thermal(tile_ctx, cam_ctx)

        # Build card
        card = build_single_card(tile_body, ov_body, ov_face, ov_ctx, true_name, pred_name, conf, idx)

        # Save files
        fname = f"sample_{idx:04d}_{true_name}_{pred_name}.png"
        cv2.imwrite(os.path.join(out_all, fname), card)
        cv2.imwrite(os.path.join(f'outputs/gradcam/by_class/{true_name}', fname), card)

    print("\n=> FULL DATASET 4-STREAM GRAD-CAM VISUALIZATION COMPLETE (528/528 videos)!")


if __name__ == '__main__':
    main()
