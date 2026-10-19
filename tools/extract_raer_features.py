#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/extract_raer_features.py
───────────────────────────────
Offline Visual Feature Extractor for RAER Dataset (Fast Screening Pre-computation).
Extracts frozen CLIP ViT-B/16 visual tokens for all video segments (T=16):
  - Face stream: (N, 16, 512)
  - Body stream: (N, 16, 512)
  - Context stream: (N, 16, 512)
  - Ground-truth labels: (N,)

Saves cached .pt files (~170MB total) to enable ultra-fast ablation sweeps (seconds per experiment).
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repo root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from dataloader.video_dataloader import test_data_loader

RAER_CLASSES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and Cache RAER Visual Features")
    parser.add_argument('--root-dir', type=str, default='.',
                        help='Root directory for RAER dataset')
    parser.add_argument('--train-annotation', type=str, default='RAER/annotation/train.txt',
                        help='Path to train.txt')
    parser.add_argument('--test-annotation', type=str, default='RAER/annotation/test.txt',
                        help='Path to test.txt')
    parser.add_argument('--bounding-box-face', type=str, default='RAER/bounding_box/face.json',
                        help='Path to face bounding boxes JSON')
    parser.add_argument('--bounding-box-body', type=str, default='RAER/bounding_box/body.json',
                        help='Path to body bounding boxes JSON')
    parser.add_argument('--clip-path', type=str, default='ViT-B/16',
                        help='CLIP backbone model architecture')
    parser.add_argument('--num-segments', type=int, default=16,
                        help='Number of segments sampled per video')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size of videos to process per forward pass')
    parser.add_argument('--workers', type=int, default=4,
                        help='Dataloader num_workers')
    parser.add_argument('--out-dir', type=str, default='cached_features/raer',
                        help='Directory to save cached feature tensors')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Computation device')
    return parser.parse_args()


def select_device(device_arg):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_arg)


def resolve_path(p, candidate_dirs=None):
    if os.path.exists(p):
        return os.path.abspath(p)
    if candidate_dirs:
        for c in candidate_dirs:
            # Direct join
            joined = os.path.join(c, p)
            if os.path.exists(joined):
                return os.path.abspath(joined)
            # If p has 'RAER/' prefix, try without it
            if p.startswith('RAER/') or p.startswith('RAER\\'):
                sub = p[5:]
                joined_sub = os.path.join(c, sub)
                if os.path.exists(joined_sub):
                    return os.path.abspath(joined_sub)
            # If c ends with 'RAER', also try joining directly
            if os.path.basename(c) == 'RAER':
                joined_parent = os.path.join(os.path.dirname(c), p)
                if os.path.exists(joined_parent):
                    return os.path.abspath(joined_parent)
    return p


def extract_split(loader, visual_encoder, device, split_name):
    print(f"\n=> [Extracting {split_name}] Total samples: {len(loader.dataset):,} videos")
    all_face = []
    all_body = []
    all_context = []
    all_labels = []

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    start_t = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split_name} features"):
            # batch: face (B, T, 3, 224, 224), body (B, T, 3, 224, 224), context (B, T, 3, 224, 224), label (B,)
            img_face, img_body, img_ctx, labels = batch
            b, t, c, h, w = img_face.shape

            # Reshape to (B*T, 3, 224, 224)
            f_in = img_face.view(-1, c, h, w).to(device)
            b_in = img_body.view(-1, c, h, w).to(device)
            c_in = img_ctx.view(-1, c, h, w).to(device)

            # Normalize for CLIP
            f_in = (f_in - mean) / std
            b_in = (b_in - mean) / std
            c_in = (c_in - mean) / std

            # Forward pass through frozen CLIP visual backbone
            f_feat = visual_encoder(f_in).view(b, t, -1).cpu()
            b_feat = visual_encoder(b_in).view(b, t, -1).cpu()
            c_feat = visual_encoder(c_in).view(b, t, -1).cpu()

            all_face.append(f_feat)
            all_body.append(b_feat)
            all_context.append(c_feat)
            all_labels.append(labels.cpu())

    all_face = torch.cat(all_face, dim=0).float()
    all_body = torch.cat(all_body, dim=0).float()
    all_context = torch.cat(all_context, dim=0).float()
    all_labels = torch.cat(all_labels, dim=0).long()

    elapsed = time.time() - start_t
    print(f"=> [{split_name} Complete] Face: {all_face.shape} | Body: {all_body.shape} | "
          f"Context: {all_context.shape} | Time: {elapsed:.1f}s")
    
    return {
        'face_feats': all_face,
        'body_feats': all_body,
        'context_feats': all_context,
        'labels': all_labels
    }


def main():
    args = parse_args()
    device = select_device(args.device)
    print("=" * 70)
    print("  RAPT-CLIP: Fast Screening Feature Extractor for RAER")
    print(f"  Device: {device} | CLIP Backbone: {args.clip_path}")
    print("=" * 70)

    # Candidate directories on Kaggle & Local
    candidate_roots = [
        '.',
        'RAER',
        '/kaggle/input/datasets/bearmn/raer-video-emotion-dataset',
        '/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER',
        '/kaggle/input/datasets/bearmn/raer-annot',
        '/kaggle/input/raer-video-emotion-dataset',
        '/kaggle/input/raer-video-emotion-dataset/RAER',
        '/kaggle/input/raer-annot'
    ]

    root_dir = resolve_path(args.root_dir, candidate_roots)
    train_annot = resolve_path(args.train_annotation, candidate_roots)
    test_annot = resolve_path(args.test_annotation, candidate_roots)
    face_box = resolve_path(args.bounding_box_face, candidate_roots)
    body_box = resolve_path(args.bounding_box_body, candidate_roots)

    print(f"=> Root Dir:         {root_dir}")
    print(f"=> Train Annotation: {train_annot}")
    print(f"=> Test Annotation:  {test_annot}")
    print(f"=> Face BBoxes:      {face_box}")
    print(f"=> Body BBoxes:      {body_box}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load CLIP Model
    print(f"\n=> Loading CLIP {args.clip_path}...")
    clip_model, _ = clip.load(args.clip_path, device=device, jit=False)
    visual_encoder = clip_model.visual.float().eval()
    for param in visual_encoder.parameters():
        param.requires_grad = False

    # 2. Build Dataloaders for Feature Extraction (Clean deterministic sampling)
    print("\n=> Initializing Video Dataloaders (Clean deterministic evaluation mode)...")
    train_dataset = test_data_loader(
        root_dir=root_dir,
        list_file=train_annot,
        num_segments=args.num_segments,
        duration=1,
        image_size=224,
        bounding_box_face=face_box,
        bounding_box_body=body_box,
        crop_body=True,
        num_classes=len(RAER_CLASSES),
        dataset_name="RAER",
        mask_context_face=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )

    test_dataset = test_data_loader(
        root_dir=root_dir,
        list_file=test_annot,
        num_segments=args.num_segments,
        duration=1,
        image_size=224,
        bounding_box_face=face_box,
        bounding_box_body=body_box,
        crop_body=True,
        num_classes=len(RAER_CLASSES),
        dataset_name="RAER",
        mask_context_face=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )

    # 3. Extract and save Train features
    train_dict = extract_split(train_loader, visual_encoder, device, "TRAIN")
    train_out_path = os.path.join(args.out_dir, 'raer_train_features.pt')
    torch.save(train_dict, train_out_path)
    print(f"=> Saved Train features to: {train_out_path} ({os.path.getsize(train_out_path):,} bytes)")

    # 4. Extract and save Test features
    test_dict = extract_split(test_loader, visual_encoder, device, "TEST")
    test_out_path = os.path.join(args.out_dir, 'raer_test_features.pt')
    torch.save(test_dict, test_out_path)
    print(f"=> Saved Test features to: {test_out_path} ({os.path.getsize(test_out_path):,} bytes)")

    # 5. Save Metadata
    meta = {
        'dataset': 'RAER',
        'classes': RAER_CLASSES,
        'num_classes': len(RAER_CLASSES),
        'num_train': len(train_dict['labels']),
        'num_test': len(test_dict['labels']),
        'num_segments': args.num_segments,
        'feature_dim': train_dict['face_feats'].shape[-1],
        'class_distribution_train': torch.bincount(train_dict['labels']).tolist(),
        'class_distribution_test': torch.bincount(test_dict['labels']).tolist(),
        'extracted_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    meta_path = os.path.join(args.out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n=> [SUCCESS] All RAER features cached successfully in '{args.out_dir}'!")
    print(f"   Train samples: {meta['num_train']} | Class dist: {meta['class_distribution_train']}")
    print(f"   Test samples:  {meta['num_test']} | Class dist: {meta['class_distribution_test']}")


if __name__ == '__main__':
    main()
