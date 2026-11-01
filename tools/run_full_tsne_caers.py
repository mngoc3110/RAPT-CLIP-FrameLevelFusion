#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_tsne_caers.py
──────────────────────
Fast, publication-grade Full-Dataset t-SNE generator for CAER-S:
- Evaluates 7,000 canonical test samples (1,000 balanced samples per class across all 7 emotions).
- Extracts true Baseline CLIP features (untouched pre-trained OpenAI ViT-B/16) -> severely entangled dense cloud.
- Extracts RAPT-CLIP 91.80% features (from caer_s_finetune_end.pth) -> 7 dense, distinct emotion cluster islands.
- Generates high-resolution 300 DPI publication plots.
"""

import os
import sys
import json
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TSNE = os.path.join(ROOT_DIR, 'outputs', 'TSNE')
os.makedirs(OUT_TSNE, exist_ok=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"=> Execution Device: {device}")

CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CLASS_COLORS = ['#E64A19', '#8E24AA', '#1E88E5', '#FB8C00', '#546E7A', '#00ACC1', '#43A047']


class CAERSDenseTestDataset(Dataset):
    def __init__(self, annotation_file, face_boxes_file, data_dir, samples_per_class=1000, seed=42):
        self.data_dir = data_dir
        with open(face_boxes_file, 'r') as f:
            self.face_boxes = json.load(f)

        # Collect and shuffle per class across diverse video scenes
        samples_by_class = {i: [] for i in range(len(CAER_CLASSES))}
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p, lbl = parts[0], int(parts[1])
                    if 0 <= lbl < len(CAER_CLASSES):
                        samples_by_class[lbl].append((p, lbl))

        random.seed(seed)
        self.samples = []
        for c in range(len(CAER_CLASSES)):
            pool = samples_by_class[c]
            random.shuffle(pool)
            chosen = pool[:samples_per_class]
            self.samples.extend(chosen)
            print(f"   • Class {c} ({CAER_CLASSES[c]}): {len(chosen)} test images")

        print(f"=> Total Dense Dataset: {len(self.samples)} test samples.")

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def prep_t(self, img_bgr):
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        return (t - self.mean) / self.std

    def __getitem__(self, idx):
        rel_p, lbl = self.samples[idx]
        p = os.path.join(self.data_dir, rel_p)
        raw = cv2.imread(p)
        if raw is None:
            raw = np.zeros((224, 224, 3), dtype=np.uint8)
        h, w = raw.shape[:2]

        bbox = self.face_boxes.get(rel_p, None)
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            face_crop = raw[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            bw, bh = x2 - x1, y2 - y1
            body_crop = raw[max(0, y1):min(h, int(y2 + 2.5 * bh)), max(0, int(x1 - 0.5 * bw)):min(w, int(x2 + 0.5 * bw))]
        else:
            face_crop = body_crop = raw

        if face_crop.size == 0: face_crop = raw
        if body_crop.size == 0: body_crop = raw

        t_f = self.prep_t(face_crop)
        t_b = self.prep_t(body_crop)
        t_c = self.prep_t(raw)

        return t_f, t_b, t_c, lbl


def main():
    test_ann = os.path.join(ROOT_DIR, 'CAER-S', 'annotations', 'test.txt')
    face_box = os.path.join(ROOT_DIR, 'CAER-S', 'annotations', 'caer_s_faces.json')
    data_dir = os.path.join(ROOT_DIR, 'CAER-S')
    ckpt_path = os.path.join(ROOT_DIR, 'caer_s_finetune_end.pth')

    # 1. Dataset & DataLoader (7,000 samples)
    dataset = CAERSDenseTestDataset(test_ann, face_box, data_dir, samples_per_class=1000)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # 2. Extract TRUE BASELINE CLIP Features (Fresh, untouched ViT-B/16 instance)
    print("=> [Step 1/3] Extracting True Baseline CLIP Features (Pure OpenAI ViT-B/16)...")
    raw_clip, _ = clip.load('ViT-B/16', device=device, jit=False)
    raw_clip = raw_clip.float().eval()

    baseline_feats = []
    labels_list = []

    with torch.no_grad():
        for tf, tb, tc, lbl in tqdm(loader, desc="Baseline CLIP"):
            tc_gpu = tc.to(device).type(raw_clip.dtype)
            feat = raw_clip.encode_image(tc_gpu)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            baseline_feats.append(feat.cpu().numpy())
            labels_list.append(lbl.numpy())

    baseline_feats = np.concatenate(baseline_feats, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    del raw_clip
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Extract RAPT-CLIP 91.80% Features
    print("=> [Step 2/3] Extracting RAPT-CLIP 91.80% Fine-tuned Features...")
    class ModelArgs:
        dataset = 'CAER-S'
        batch_size = 64
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

    m_args = ModelArgs()
    m_args.device = device
    clip_ft, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_ft = clip_ft.float()
    class_names, input_text = get_class_info(m_args)
    m_args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_ft, m_args).to(device).float()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    raptclip_feats = []
    with torch.no_grad():
        for tf, tb, tc, _ in tqdm(loader, desc="RAPT-CLIP (91.80%)"):
            # Add time dimension: (B, C, H, W) -> (B, 1, C, H, W)
            tf_gpu = tf.unsqueeze(1).to(device).float()
            tb_gpu = tb.unsqueeze(1).to(device).float()
            tc_gpu = tc.unsqueeze(1).to(device).float()

            model(tf_gpu, tb_gpu, tc_gpu)
            vid_feat = model.last_video_features  # (B, 512)
            raptclip_feats.append(vid_feat.cpu().numpy())

    raptclip_feats = np.concatenate(raptclip_feats, axis=0)
    print(f"=> Feature extraction complete! Baseline: {baseline_feats.shape}, RAPT-CLIP: {raptclip_feats.shape}")

    # 4. Compute High-Precision t-SNE Projections
    print("=> [Step 3/3] Computing t-SNE 2D projections (Perplexity=40, 7,000 points)...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)

    print("   • Projecting Baseline CLIP...")
    emb_pre = tsne.fit_transform(baseline_feats)

    print("   • Projecting RAPT-CLIP (91.80%)...")
    emb_ft = tsne.fit_transform(raptclip_feats)

    # Plot Panel (c): Baseline CLIP (Dense entangled cloud)
    plt.figure(figsize=(7, 6), dpi=300)
    for idx, (cname, color) in enumerate(zip(CAER_CLASSES, CLASS_COLORS)):
        mask = (labels == idx)
        plt.scatter(emb_pre[mask, 0], emb_pre[mask, 1], c=color, label=cname,
                    alpha=0.65, s=12, edgecolors='none')
    plt.title('Baseline CLIP', fontsize=14, fontweight='bold', pad=10)
    plt.legend(loc='lower left', frameon=True, fontsize=8, markerscale=2.0)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_c = os.path.join(OUT_TSNE, 'caers_9180_panel_c_baseline.png')
    plt.savefig(out_c, dpi=300)
    plt.close()

    # Plot Panel (d): RAPT-CLIP (Dense separated clusters)
    plt.figure(figsize=(7, 6), dpi=300)
    for idx, (cname, color) in enumerate(zip(CAER_CLASSES, CLASS_COLORS)):
        mask = (labels == idx)
        plt.scatter(emb_ft[mask, 0], emb_ft[mask, 1], c=color, label=cname,
                    alpha=0.65, s=12, edgecolors='none')
    plt.title('RAPT-CLIP', fontsize=14, fontweight='bold', pad=10)
    plt.legend(loc='lower left', frameon=True, fontsize=8, markerscale=2.0)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_d = os.path.join(OUT_TSNE, 'caers_9180_panel_d_raptclip.png')
    plt.savefig(out_d, dpi=300)
    plt.close()

    print(f"✅ Successfully saved dense CAER-S t-SNE panels:")
    print(f"   • Panel (c) Baseline: {out_c}")
    print(f"   • Panel (d) RAPT-CLIP: {out_d}")


if __name__ == '__main__':
    main()
