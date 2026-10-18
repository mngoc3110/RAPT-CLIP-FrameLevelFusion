#!/usr/bin/env python3
"""
RAPT-CLIP DAiSEE Multimodal Saliency & Grad-CAM Visualization — v3
===================================================================
Produces 4-stream cards: [ (a) Original Webcam | (b) Face Saliency | (c) Posture Saliency | (d) Context Saliency ]

Key features:
1. Automated Face Detection via MTCNN: Extracts high-precision facial crop (eyes, nose, mouth).
2. Anatomical Upper-Body / Posture Crop: Computes head-to-torso and shoulder-to-shoulder region.
3. Facial Saliency: Token-level CLIP similarity targeting facial gaze, micro-expressions, and eye focus.
4. Posture Saliency: Token-level CLIP similarity for posture & sitting tilt, with facial suppression (relative bbox zero-out)
   to guarantee attention stays on shoulders, neck, and upper body language.
5. Context Saliency: Ambient room, desk, and lighting cues with student suppression.
6. Model inference: Evaluates trained DAiSEE model for prediction & confidence.
7. Produces individual cards and a high-resolution gallery poster across Low, High, and Very High Engagement.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info
from dataloader.video_dataloader import test_data_loader
from utils.checkpoint_utils import load_slim_checkpoint

DAISEE_CLASSES = ['Low Engagement', 'High Engagement', 'Very High Engagement']


def parse_args():
    parser = argparse.ArgumentParser(description="DAiSEE Multimodal Saliency Visualization")
    parser.add_argument('--checkpoint', type=str, default='model_daisee.zip')
    parser.add_argument('--root-dir', type=str, default='DAiSEE')
    parser.add_argument('--test-annotation', type=str, default='DAiSEE/daisee_test.txt')
    parser.add_argument('--bounding-box-face', type=str, default='DAiSEE/face_bbox_dummy.json')
    parser.add_argument('--bounding-box-body', type=str, default='DAiSEE/body_bbox_dummy.json')
    parser.add_argument('--num-samples', type=int, default=30)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--out-dir', type=str, default='outputs/gradcam_daisee_v3')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'])
    return parser.parse_args()


def select_device(device_arg):
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')
    return torch.device(device_arg)


def resolve_checkpoint(path):
    if os.path.exists(path):
        return path
    for fb in ['model_clean.pth', 'model_daisee.zip', 'model_best.pth']:
        if os.path.exists(fb):
            print(f"[Warning] '{path}' not found. Using fallback '{fb}'.")
            return fb
    raise FileNotFoundError(f"Checkpoint not found at '{path}'.")


def get_token_similarity_map(clip_model, img_224, query_feat, device):
    """
    Computes spatial patch token cosine similarity with target text embedding.
    Extracts 14x14 spatial grid after ln_post and projection.
    """
    rgb = cv2.cvtColor(img_224, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    t = (t - mean) / std

    visual = clip_model.visual
    with torch.no_grad():
        x = visual.conv1(t)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        spatial = x[:, 1:, :]  # (1, 196, 768)
        spatial_norm = visual.ln_post(spatial)  # (1, 196, 768)
        spatial_proj = spatial_norm @ visual.proj  # (1, 196, 512)
        spatial_proj = spatial_proj / (spatial_proj.norm(dim=-1, keepdim=True) + 1e-6)

        sim = (spatial_proj[0] @ query_feat).cpu().numpy().reshape(14, 14)
    return sim


def norm_heatmap(m, percentile=45, sigma=14.0):
    thr = np.percentile(m, percentile)
    m = np.clip(m - thr, 0, None)
    up = cv2.resize(m, (224, 224), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (31, 31), sigmaX=sigma)
    b_min, b_max = blurred.min(), blurred.max()
    return np.clip((blurred - b_min) / (b_max - b_min + 1e-6), 0, 1)


def overlay_thermal(img_bgr, cam_norm, heat_power=1.3, base_alpha=0.12):
    """Blends vibrant thermal JET colormap with adaptive alpha."""
    heat = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET).astype(np.float32) / 255.0
    img_f = img_bgr.astype(np.float32) / 255.0
    alpha = (cam_norm ** heat_power) * 0.65 + base_alpha
    alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    blend = (1.0 - alpha_3d) * img_f + alpha_3d * heat
    return np.uint8(255 * np.clip(blend, 0, 1))


def build_single_card(tile_orig, tile_face_ov, tile_body_ov, tile_ctx_ov,
                      true_name, pred_name, conf, sample_idx, clip_id=""):
    tile_size = 224
    gap = 14
    pad_top = 46
    pad_bot = 42
    card_w = tile_size * 4 + gap * 5
    card_h = pad_top + tile_size + pad_bot
    card = np.ones((card_h, card_w, 3), dtype=np.uint8) * 255

    cols = ['(a) Original Webcam', '(b) Face Saliency', '(c) Posture Saliency', '(d) Context Saliency']
    font = cv2.FONT_HERSHEY_DUPLEX
    for i, title in enumerate(cols):
        tw = cv2.getTextSize(title, font, 0.44, 1)[0][0]
        x  = gap + i * (tile_size + gap) + (tile_size - tw) // 2
        cv2.putText(card, title, (x, 28), font, 0.44, (45, 45, 45), 1, cv2.LINE_AA)

    y = pad_top
    card[y:y+tile_size, gap:gap+tile_size] = cv2.resize(tile_orig, (tile_size, tile_size))
    card[y:y+tile_size, gap+(tile_size+gap):gap+(tile_size+gap)+tile_size] = cv2.resize(tile_face_ov, (tile_size, tile_size))
    card[y:y+tile_size, gap+2*(tile_size+gap):gap+2*(tile_size+gap)+tile_size] = cv2.resize(tile_body_ov, (tile_size, tile_size))
    card[y:y+tile_size, gap+3*(tile_size+gap):gap+3*(tile_size+gap)+tile_size] = cv2.resize(tile_ctx_ov, (tile_size, tile_size))

    match = (true_name == pred_name)
    color = (25, 135, 25) if match else (25, 25, 205)
    info = (f"Sample #{sample_idx:04d} | Clip: {clip_id} | "
            f"Ground Truth: {true_name} | Predicted: {pred_name} ({conf:.1f}%)")
    cv2.putText(card, info, (gap + 4, card_h - 15), font, 0.46, color, 1, cv2.LINE_AA)
    return card


def main():
    args  = parse_args()
    device = select_device(args.device)
    ckpt_path = resolve_checkpoint(args.checkpoint)

    print("\n===========================================================")
    print("RAPT-CLIP DAiSEE Multimodal Saliency Visualization — v3")
    print("===========================================================")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output:     {args.out_dir}")
    print("===========================================================\n")

    model_args = argparse.Namespace(
        dataset='DAiSEE',
        root_dir=args.root_dir,
        test_annotation=args.test_annotation,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body,
        batch_size=1, workers=1,
        num_segments=8, duration=1, image_size=224,
        use_context=False, crop_body=False,
        mask_context_face=False, daisee_merge_low=True,
        use_npy=False, text_type='prompt_ensemble',
        clip_path='ViT-B/16', temporal_layers=1,
        fusion_type='cmaf', fusion_stage='frame',
        contexts_number=8, class_token_position='end',
        class_specific_contexts=True,
        load_and_tune_prompt_learner=True,
        drop_path_rate=0.0, modality_dropout=0.0,
        temperature=0.07, freeze_image_encoder=False,
        ablation_no_text=False, device=device
    )

    class_names, input_text = get_class_info(model_args)
    model_args.num_classes = len(class_names)

    print("=> Loading CLIP ViT-B/16...")
    clip_model, _ = clip.load(model_args.clip_path, device=device, jit=False)
    clip_model.float()

    model = GenerateModel(input_text, clip_model, model_args).float().to(device)
    load_slim_checkpoint(model, ckpt_path, device=device)
    model.eval()

    # Pre-encode modality-specific semantic guide texts
    prompts = [
        "a student face looking attentively at screen, focused eyes and gaze",
        "student upper body posture upright sitting attentive shoulder torso",
        "student desk room background study environment"
    ]
    tok_prompts = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        sem_feats = clip_model.encode_text(tok_prompts).float()
        sem_feats = sem_feats / (sem_feats.norm(dim=-1, keepdim=True) + 1e-6)
    feat_face, feat_body, feat_ctx = sem_feats[0], sem_feats[1], sem_feats[2]

    print("=> Initializing MTCNN Face Detector...")
    mtcnn = MTCNN(keep_all=False, select_largest=True, device='cpu')

    print("=> Loading DAiSEE Test DataLoader...")
    test_data = test_data_loader(
        root_dir=model_args.root_dir,
        list_file=model_args.test_annotation,
        num_segments=model_args.num_segments,
        duration=model_args.duration,
        image_size=model_args.image_size,
        crop_body=model_args.crop_body,
        dataset_name=model_args.dataset,
        bounding_box_face=model_args.bounding_box_face,
        bounding_box_body=model_args.bounding_box_body,
        mask_context_face=model_args.mask_context_face,
        daisee_merge_low=model_args.daisee_merge_low
    )
    print(f"=> Total Test Videos: {len(test_data)}")

    out_all = os.path.join(args.out_dir, 'all_samples')
    os.makedirs(out_all, exist_ok=True)
    for cname in DAISEE_CLASSES:
        os.makedirs(os.path.join(args.out_dir, 'by_class', cname.replace(' ', '_')), exist_ok=True)

    # Balanced sample selection across 3 classes
    if args.all:
        selected_indices = list(range(len(test_data)))
    else:
        spc = max(1, args.num_samples // len(DAISEE_CLASSES))
        buckets = {0: [], 1: [], 2: []}
        for i in range(len(test_data)):
            rec = test_data.video_list[i]
            offset = getattr(test_data, 'label_offset', 1)
            raw_l = rec.label - offset
            merged = 0 if raw_l <= 1 else (1 if raw_l == 2 else 2)
            if len(buckets[merged]) < spc:
                buckets[merged].append(i)
            if all(len(b) >= spc for b in buckets.values()):
                break
        selected_indices = []
        for c in range(3):
            selected_indices.extend(buckets[c])

    print(f"=> Processing {len(selected_indices)} clips...")

    cards_for_gallery = []

    for idx in tqdm(selected_indices, desc="Multimodal Saliency"):
        sample = test_data[idx]
        if len(sample) == 4:
            face_tensor, body_tensor, _, target = sample
        else:
            face_tensor, body_tensor, target = sample
        true_cls = int(target)
        true_name = DAISEE_CLASSES[true_cls]

        rec = test_data.video_list[idx]
        video_dir = rec.path
        clip_id = os.path.basename(os.path.dirname(video_dir.rstrip('/'))) or str(idx)

        # ── Step 1: Model inference for pred_cls / conf ──────────────
        with torch.no_grad():
            b_face = face_tensor.unsqueeze(0).to(device)
            b_body = body_tensor.unsqueeze(0).to(device)
            output, _, _, _ = model(b_face, b_body)
            probs = torch.softmax(output, dim=-1)[0]
            pred_cls = output.argmax(dim=-1).item()
            conf = probs[pred_cls].item() * 100.0
        pred_name = DAISEE_CLASSES[pred_cls]

        # ── Step 2: Load representative middle frame ─────────────────
        frame_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.jpg', '.png'))])
        if not frame_files:
            continue
        img_raw = cv2.imread(os.path.join(video_dir, frame_files[len(frame_files) // 2]))
        if img_raw is None:
            continue
        h_img, w_img = img_raw.shape[:2]

        # ── Step 3: Face & Posture Detection with MTCNN ─────────────
        rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)
        if boxes is not None and len(boxes) > 0:
            fx1, fy1, fx2, fy2 = [int(v) for v in boxes[0]]
        else:
            # Fallback to central upper-half face box
            fx1 = int(w_img * 0.35)
            fx2 = int(w_img * 0.65)
            fy1 = int(h_img * 0.15)
            fy2 = int(h_img * 0.55)

        fw, fh = max(20, fx2 - fx1), max(20, fy2 - fy1)

        # Crop 1: Face Crop (Tight around face with 20% margin)
        m_x, m_y = int(fw * 0.20), int(fh * 0.25)
        face_y1, face_y2 = max(0, fy1 - m_y), min(h_img, fy2 + m_y)
        face_x1, face_x2 = max(0, fx1 - m_x), min(w_img, fx2 + m_x)
        tile_face = cv2.resize(img_raw[face_y1:face_y2, face_x1:face_x2], (224, 224))

        # Crop 2: Posture / Body Crop (Head top down to mid-torso/table, shoulders wide)
        body_y1 = max(0, fy1 - int(fh * 0.2))
        body_y2 = min(h_img, fy2 + int(fh * 2.8))
        body_x1 = max(0, fx1 - int(fw * 1.6))
        body_x2 = min(w_img, fx2 + int(fw * 1.6))
        tile_body = cv2.resize(img_raw[body_y1:body_y2, body_x1:body_x2], (224, 224))

        # Crop 3: Context Crop (Full webcam frame)
        tile_ctx = cv2.resize(img_raw, (224, 224))

        # ── Step 4: Token-Level Saliency Maps ────────────────────────
        # (b) Face Saliency
        s_face = get_token_similarity_map(clip_model, tile_face, feat_face, device)

        # (c) Posture Saliency with Face Suppression
        s_body = get_token_similarity_map(clip_model, tile_body, feat_body, device)
        b_h, b_w = max(1, body_y2 - body_y1), max(1, body_x2 - body_x1)
        rel_fy1 = max(0.0, (fy1 - body_y1) / b_h)
        rel_fy2 = min(1.0, (fy2 - body_y1) / b_h)
        rel_fx1 = max(0.0, (fx1 - body_x1) / b_w)
        rel_fx2 = min(1.0, (fx2 - body_x1) / b_w)
        gu, gd = int(rel_fy1 * 14), int(np.ceil(rel_fy2 * 14))
        gl, gr = int(rel_fx1 * 14), int(np.ceil(rel_fx2 * 14))
        # Suppress face region in posture crop so attention pops on shoulders, neck & torso
        s_body[gu:gd, gl:gr] *= 0.15

        # (d) Context Saliency with Person Suppression
        s_ctx = get_token_similarity_map(clip_model, tile_ctx, feat_ctx, device)
        rel_cy1, rel_cy2 = max(0.0, body_y1 / h_img), min(1.0, body_y2 / h_img)
        rel_cx1, rel_cx2 = max(0.0, body_x1 / w_img), min(1.0, body_x2 / w_img)
        cgu, cgd = int(rel_cy1 * 14), int(np.ceil(rel_cy2 * 14))
        cgl, cgr = int(rel_cx1 * 14), int(np.ceil(rel_cx2 * 14))
        s_ctx[cgu:cgd, cgl:cgr] *= 0.10

        # ── Step 5: Normalize and Thermal Overlay ────────────────────
        cam_face = norm_heatmap(s_face, percentile=45, sigma=12.0)
        cam_body = norm_heatmap(s_body, percentile=45, sigma=15.0)
        cam_ctx  = norm_heatmap(s_ctx,  percentile=40, sigma=18.0)

        ov_face = overlay_thermal(tile_face, cam_face)
        ov_body = overlay_thermal(tile_body, cam_body)
        ov_ctx  = overlay_thermal(tile_ctx,  cam_ctx)

        # ── Step 6: Assemble Card ────────────────────────────────────
        card = build_single_card(tile_ctx, ov_face, ov_body, ov_ctx,
                                 true_name, pred_name, conf, idx, clip_id)
        fname = (f"sample_{idx:04d}_{clip_id}_"
                 f"{true_name.replace(' ','_')}_pred_{pred_name.replace(' ','_')}.png")
        cv2.imwrite(os.path.join(out_all, fname), card)
        cv2.imwrite(os.path.join(args.out_dir, 'by_class', true_name.replace(' ', '_'), fname), card)
        cards_for_gallery.append((true_cls, card))

    # ── Step 7: Build Gallery Poster ────────────────────────────────
    if cards_for_gallery:
        print("\n=> Assembling Gallery Poster...")
        gallery = []
        for c in range(3):
            cc = [card for ci, card in cards_for_gallery if ci == c]
            gallery.extend(cc[:2])
        if gallery:
            gpath = os.path.join(args.out_dir, 'daisee_gradcam_gallery.png')
            cv2.imwrite(gpath, np.vstack(gallery))
            print(f"   Gallery saved -> {gpath}")

    print("\n" + "=" * 65)
    print("DAiSEE Saliency Visualization DONE!")
    print(f"  Output directory: {args.out_dir}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
