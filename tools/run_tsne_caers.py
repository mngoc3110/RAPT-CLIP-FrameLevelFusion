#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_tsne_caers.py
───────────────────────
t-SNE visualization of CAER-S test set features using caer_s_finetune_end.pth (91.80% Acc).
Extracts fused visual embeddings (Face + Body + Context) -> CMAF -> Linear Projection,
and plots t-SNE 2D scatter matching Figure 4 in the publication.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info

CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# Distinct harmonious colors matching publication standard
CLASS_COLORS = ['#e11d48', '#ea580c', '#0284c7', '#16a34a', '#8b5cf6', '#0d9488', '#d97706']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='caer_s_finetune_end.pth')
    parser.add_argument('--data-dir', type=str, default='CAER-S')
    parser.add_argument('--face-box', type=str, default='CAER-S/annotations/caer_s_faces.json')
    parser.add_argument('--test-txt', type=str, default='CAER-S/annotations/test.txt')
    parser.add_argument('--samples-per-class', type=int, default=150)
    parser.add_argument('--out-dir', type=str, default='outputs/TSNE')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"=> Execution Device: {device} | Checkpoint: {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Model configuration for static CAER-S
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

    print("=> Loading CLIP backbone...")
    clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_model = clip_model.float()
    class_names, input_text = get_class_info(model_args)
    model_args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_model, model_args).to(device).float()

    print(f"=> Loading weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load Face boxes
    with open(args.face_box, 'r') as f:
        face_boxes = json.load(f)

    # Sample balanced set from test.txt
    samples_by_class = {c: [] for c in CAER_CLASSES}
    with open(args.test_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                p, lbl = parts[0], int(parts[1])
                if 0 <= lbl < len(CAER_CLASSES):
                    cname = CAER_CLASSES[lbl]
                    if len(samples_by_class[cname]) < args.samples_per_class:
                        samples_by_class[cname].append((p, lbl))

    all_samples = []
    for cname in CAER_CLASSES:
        all_samples.extend(samples_by_class[cname])
    print(f"=> Sampled {len(all_samples)} total test images across 7 classes ({args.samples_per_class}/class).")

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1).to(device)

    def prep_tensor(img_bgr):
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(1).to(device) / 255.0
        return (t - mean) / std

    finetuned_feats = []
    pretrained_feats = []
    labels = []

    print("=> Extracting multi-modal features...")
    with torch.no_grad():
        for rel_path, lbl in tqdm(all_samples):
            full_path = os.path.join(args.data_dir, rel_path)
            if not os.path.exists(full_path):
                continue
            img_raw = cv2.imread(full_path)
            if img_raw is None:
                continue
            h, w = img_raw.shape[:2]

            # Face crop
            bbox = face_boxes.get(rel_path, None)
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                    face_crop = img_raw[y1:y2, x1:x2]
                else:
                    face_crop = img_raw
            else:
                face_crop = img_raw

            # Body crop (anatomical expansion)
            if bbox and len(bbox) == 4:
                bw = x2 - x1
                bh = y2 - y1
                bx1 = max(0, int(x1 - 0.5 * bw))
                bx2 = min(w, int(x2 + 0.5 * bw))
                by1 = max(0, y1)
                by2 = min(h, int(y2 + 2.5 * bh))
                body_crop = img_raw[by1:by2, bx1:bx2]
            else:
                body_crop = img_raw

            context_img = img_raw.copy()

            t_face = prep_tensor(face_crop)
            t_body = prep_tensor(body_crop)
            t_ctx = prep_tensor(context_img)

            # RAPT-CLIP fine-tuned feature
            out = model(t_face, t_body, t_ctx)
            feat_ft = model.last_video_features.squeeze(0).cpu().numpy()

            # Pretrained CLIP baseline feature (direct CLIP ViT-B/16 on context image)
            raw_t = t_ctx.squeeze(1).type(model.dtype)
            feat_pre = clip_model.encode_image(raw_t)
            feat_pre = (feat_pre / (feat_pre.norm(dim=-1, keepdim=True) + 1e-6)).squeeze(0).cpu().numpy()

            finetuned_feats.append(feat_ft)
            pretrained_feats.append(feat_pre)
            labels.append(lbl)

    finetuned_feats = np.array(finetuned_feats)
    pretrained_feats = np.array(pretrained_feats)
    labels = np.array(labels)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'caers_finetuned_features.npy'), finetuned_feats)
    np.save(os.path.join(args.out_dir, 'caers_pretrained_features.npy'), pretrained_feats)
    np.save(os.path.join(args.out_dir, 'caers_labels.npy'), labels)

    print("=> Computing t-SNE projections...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_ft = tsne.fit_transform(finetuned_feats)
    emb_pre = tsne.fit_transform(pretrained_feats)

    # Plot 1: Pretrained Baseline
    plt.figure(figsize=(7, 6), dpi=300)
    for idx, (cname, color) in enumerate(zip(CAER_CLASSES, CLASS_COLORS)):
        mask = (labels == idx)
        plt.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=color, label=cname, alpha=0.75, s=25, edgecolors='none')
    plt.title("Baseline CLIP (Pre-trained)", fontsize=13, fontweight='bold', pad=10)
    plt.legend(loc='lower left', frameon=True, fontsize=8, markerscale=1.4)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_pre = os.path.join(args.out_dir, 'caers_tsne_pretrained.png')
    plt.savefig(out_pre, dpi=300)
    plt.close()

    # Plot 2: RAPT-CLIP Fine-tuned
    plt.figure(figsize=(7, 6), dpi=300)
    for idx, (cname, color) in enumerate(zip(CAER_CLASSES, CLASS_COLORS)):
        mask = (labels == idx)
        plt.scatter(emb_ft[mask, 0], emb_ft[mask, 1], c=color, label=cname, alpha=0.75, s=25, edgecolors='none')
    plt.title("RAPT-CLIP (Ours Fine-tuned)", fontsize=13, fontweight='bold', pad=10)
    plt.legend(loc='lower left', frameon=True, fontsize=8, markerscale=1.4)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_ft = os.path.join(args.out_dir, 'caers_tsne_finetuned.png')
    plt.savefig(out_ft, dpi=300)
    plt.close()

    # Plot 3: Side-by-side combined matching Figure 4 (c, d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    for idx, (cname, color) in enumerate(zip(CAER_CLASSES, CLASS_COLORS)):
        mask = (labels == idx)
        ax1.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=color, label=cname, alpha=0.75, s=25, edgecolors='none')
        ax2.scatter(emb_ft[mask, 0], emb_ft[mask, 1], c=color, label=cname, alpha=0.75, s=25, edgecolors='none')

    ax1.set_title("Baseline CLIP", fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='lower left', frameon=True, fontsize=8, markerscale=1.4)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_title("RAPT-CLIP", fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='lower left', frameon=True, fontsize=8, markerscale=1.4)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    out_comb = os.path.join(args.out_dir, 'caers_tsne_figure4_cd.png')
    plt.savefig(out_comb, dpi=300)
    plt.close()

    print(f"✅ SUCCESS: All CAER-S t-SNE figures saved to '{args.out_dir}':")
    print(f"   • Pre-trained: {out_pre}")
    print(f"   • Fine-tuned : {out_ft}")
    print(f"   • Combined Fig 4(c,d): {out_comb}")


if __name__ == '__main__':
    main()
