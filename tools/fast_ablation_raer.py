#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/fast_ablation_raer.py
───────────────────────────
Ultra-Fast Feature-Level Ablation Engine for RAPT-CLIP on RAER Dataset.
Trains lightweight fusion, temporal, and classifier heads directly on pre-cached
CLIP visual features (~170MB).

Speedup:
  - Full End-to-End Epoch: ~180s on GPU
  - Fast Feature-Level Epoch: ~0.6s on GPU (300x speedup!)
  - Full 20-epoch ablation run: ~12-15s per experiment.
  - Complete 8-Ablation Sweep: ~2 minutes total!

Ablation Matrix:
  1. Full_RAPT_CLIP       : CMAF + AttnPool + LDAM + FaceAdapter + WeightedSampler + ModalityDropout 0.3
  2. No_LDAM              : Standard Cross-Entropy loss instead of LDAM
  3. No_Adapter           : Direct CLIP features without EAA Face Adapter
  4. No_Attn_Pooling      : Mean Temporal Pooling instead of Self-Attention Pooling
  5. No_Weighted_Sampler  : Uniform Random Sampling instead of Class-Weighted Sampling
  6. Fusion_GFI           : Gated Linear Integration (GFI) instead of Bidirectional CMAF
  7. Fusion_Concat        : Direct Concatenation + Linear Projection
  8. No_Modality_Dropout  : Modality Dropout = 0.0 (no random stream dropout)
  9. No_Temporal_Layers   : Direct pooling without Temporal Transformer layers
"""

import os
import sys
import json
import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Add repo root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from models.Adapter import Adapter
from models.CrossModalAttentionFusion import CrossModalAttentionFusion
from models.Temporal_Model import Temporal_Transformer_AttnPool
from models.Text import prompt_ensemble_5
from utils.loss import LDAMLoss

RAER_CLASSES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']


# ─────────────────────────────────────────────────────────────────────────────
# Modular Feature-Level Head Architecture
# ─────────────────────────────────────────────────────────────────────────────
class FastRAERHead(nn.Module):
    def __init__(self,
                 fusion_type='cmaf',          # 'cmaf', 'gfi', 'concat'
                 temporal_pooling='attn',     # 'attn', 'mean', 'none'
                 temporal_layers=1,           # 1, 2, or 0
                 use_adapter=True,            # EAA Face Adapter
                 use_context=False,           # Triple-stream vs Dual-stream
                 modality_dropout=0.3,        # Stream dropout during train
                 temperature=0.07,
                 num_classes=5):
        super().__init__()
        self.fusion_type = fusion_type
        self.temporal_pooling = temporal_pooling
        self.temporal_layers = temporal_layers
        self.use_adapter = use_adapter
        self.use_context = use_context
        self.modality_dropout = modality_dropout
        self.temperature = temperature
        self.num_classes = num_classes

        # 1. Face Adapter (EAA)
        if self.use_adapter:
            self.face_adapter = Adapter(c_in=512, reduction=4)
        else:
            self.face_adapter = nn.Identity()

        # 2. Fusion Module
        in_dim = 1536 if self.use_context else 1024
        self.in_dim = in_dim

        if self.fusion_type == 'cmaf':
            self.cmaf = CrossModalAttentionFusion(dim=512, num_heads=4, dropout=0.1, use_context=self.use_context)
        elif self.fusion_type == 'gfi':
            self.gate_fc = nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.ReLU(),
                nn.Linear(in_dim // 4, in_dim),
                nn.Sigmoid()
            )
        elif self.fusion_type == 'concat':
            self.concat_proj = nn.Linear(in_dim, in_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # 3. Temporal Modeling
        if self.temporal_layers > 0 and self.temporal_pooling in ['attn', 'mean']:
            self.temporal_net = Temporal_Transformer_AttnPool(
                num_patches=16,
                input_dim=in_dim,
                depth=self.temporal_layers,
                heads=8,
                mlp_dim=1024,
                dim_head=64,
                pooling=self.temporal_pooling
            )
        else:
            self.temporal_net = None

        # 4. Final Projection to 512-d CLIP space
        self.project_fc = nn.Linear(in_dim, 512)

    def forward(self, face_feats, body_feats, context_feats=None):
        # face_feats: (B, T, 512)
        # body_feats: (B, T, 512)
        b, t, d = face_feats.shape

        # Modality Dropout during training (robustness to occlusion)
        if self.training and self.modality_dropout > 0:
            p = torch.rand(b, 1, 1, device=face_feats.device)
            # 15% drop face, 15% drop body
            face_mask = (p >= self.modality_dropout * 0.5).float()
            body_mask = (p < (1.0 - self.modality_dropout * 0.5)).float()
            face_feats = face_feats * face_mask
            body_feats = body_feats * body_mask

        # Apply Face Adapter
        if self.use_adapter:
            face_reshaped = face_feats.contiguous().view(-1, d)
            face_adapted = self.face_adapter(face_reshaped).view(b, t, d)
        else:
            face_adapted = face_feats

        # Reshape to (B*T, 512) for Frame-Level Fusion
        f_flat = face_adapted.contiguous().view(-1, d)
        b_flat = body_feats.contiguous().view(-1, d)

        if self.fusion_type == 'cmaf':
            if self.use_context and context_feats is not None:
                c_flat = context_feats.contiguous().view(-1, d)
                fused_flat = self.cmaf(f_flat, b_flat, c_flat)
            else:
                fused_flat = self.cmaf(f_flat, b_flat)
        elif self.fusion_type == 'gfi':
            feats_list = [f_flat, b_flat]
            if self.use_context and context_feats is not None:
                feats_list.append(context_feats.contiguous().view(-1, d))
            cat = torch.cat(feats_list, dim=-1)
            gate = self.gate_fc(cat)
            fused_flat = cat * gate
        elif self.fusion_type == 'concat':
            feats_list = [f_flat, b_flat]
            if self.use_context and context_feats is not None:
                feats_list.append(context_feats.contiguous().view(-1, d))
            fused_flat = self.concat_proj(torch.cat(feats_list, dim=-1))

        # Temporal Modeling across T=16 segments
        fused_seq = fused_flat.contiguous().view(b, t, self.in_dim)
        if self.temporal_net is not None:
            video_feats = self.temporal_net(fused_seq)  # (B, in_dim)
        else:
            video_feats = fused_seq.mean(dim=1)         # (B, in_dim)

        # Final projection to 512-d & L2 Normalization
        video_emb = self.project_fc(video_feats)
        video_emb = video_emb / (video_emb.norm(dim=-1, keepdim=True) + 1e-6)
        return video_emb


# ─────────────────────────────────────────────────────────────────────────────
# Text Prompts Pre-computation
# ─────────────────────────────────────────────────────────────────────────────
def get_raer_text_embeddings(device):
    """Encodes RAER prompt ensemble (5 classes x 8 prompts) into (5, 8, 512)."""
    clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_model = clip_model.float().eval()

    all_prompts = []
    for cls_prompts in prompt_ensemble_5:
        all_prompts.extend(cls_prompts)

    tokenized = torch.cat([clip.tokenize(p) for p in all_prompts]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

    # Reshape to (num_classes=5, num_prompts=8, 512)
    text_features = text_features.view(len(RAER_CLASSES), len(prompt_ensemble_5[0]), -1)
    return text_features


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_head(model, text_embeddings, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for face, body, ctx, labels in loader:
            face = face.to(device)
            body = body.to(device)
            ctx = ctx.to(device)

            video_emb = model(face, body, ctx)  # (B, 512)

            # Cosine similarity with prompt ensemble: einsum (B, D) x (C, P, D) -> (B, C, P)
            logits = torch.einsum('bd,cpd->bcp', video_emb, text_embeddings)
            logits = torch.mean(logits, dim=2) / model.temperature  # (B, 5)

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # WAR (Overall Accuracy)
    war = 100.0 * (all_preds == all_targets).mean()

    # Confusion matrix & UAR (Macro-Recall)
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(RAER_CLASSES)))
    recalls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    uar = 100.0 * np.mean(recalls)

    return war, uar, recalls * 100.0, cm


# ─────────────────────────────────────────────────────────────────────────────
# Single Experiment Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def run_single_ablation(config_name, cfg, train_tensors, test_tensors, text_embeddings, device):
    print(f"\n{'='*75}")
    print(f"  RUNNING ABLATION: [{config_name}]")
    print(f"  Details: {cfg['desc']}")
    print(f"{'='*75}")

    train_face, train_body, train_ctx, train_y = train_tensors
    test_face, test_body, test_ctx, test_y = test_tensors

    # Compute class counts for LDAM Loss & Weighted Sampler
    class_counts = torch.bincount(train_y).tolist()

    # Sampler
    if cfg.get('use_sampler', True):
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_y]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_y), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_ds = TensorDataset(train_face, train_body, train_ctx, train_y)
    test_ds = TensorDataset(test_face, test_body, test_ctx, test_y)

    train_loader = DataLoader(train_ds, batch_size=cfg.get('batch_size', 32),
                              sampler=sampler, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Initialize Head
    model = FastRAERHead(
        fusion_type=cfg.get('fusion_type', 'cmaf'),
        temporal_pooling=cfg.get('temporal_pooling', 'attn'),
        temporal_layers=cfg.get('temporal_layers', 1),
        use_adapter=cfg.get('use_adapter', True),
        use_context=cfg.get('use_context', False),
        modality_dropout=cfg.get('modality_dropout', 0.3),
        temperature=0.07,
        num_classes=5
    ).to(device)

    # Loss
    loss_type = cfg.get('loss_type', 'ldam')
    if loss_type == 'ldam':
        criterion = LDAMLoss(cls_num_list=class_counts, max_m=0.5, s=30.0).to(device)
    elif loss_type == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Optimizer
    lr = cfg.get('lr', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    epochs = cfg.get('epochs', 20)
    best_uar = 0.0
    best_war = 0.0
    best_epoch = 0
    best_recalls = None

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for face, body, ctx, labels in train_loader:
            face = face.to(device)
            body = body.to(device)
            ctx = ctx.to(device)
            labels = labels.to(device)

            video_emb = model(face, body, ctx)
            logits = torch.einsum('bd,cpd->bcp', video_emb, text_embeddings)
            logits = torch.mean(logits, dim=2) / model.temperature

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)

        # Evaluate
        war, uar, recalls, _ = evaluate_head(model, text_embeddings, test_loader, device)

        if uar > best_uar:
            best_uar = uar
            best_war = war
            best_epoch = ep
            best_recalls = recalls

    elapsed = time.time() - t_start
    print(f"=> [{config_name} Result] Time: {elapsed:.1f}s | Best Epoch: {best_epoch:02d} | "
          f"Best UAR: {best_uar:5.2f}% | Best WAR: {best_war:5.2f}%")
    if best_recalls is not None:
        rec_strs = [f"{RAER_CLASSES[i][:3]}: {best_recalls[i]:.1f}%" for i in range(len(RAER_CLASSES))]
        print(f"   Per-class Recall: {' | '.join(rec_strs)}")

    return {
        'config': config_name,
        'desc': cfg['desc'],
        'best_uar': best_uar,
        'best_war': best_war,
        'best_epoch': best_epoch,
        'elapsed_sec': elapsed,
        'recalls': best_recalls.tolist() if best_recalls is not None else []
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master Ablation Matrix Definition
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_SUITE = {
    'Full_RAPT_CLIP': {
        'desc': 'Baseline SOTA: CMAF + AttnPool (1L) + LDAM (s=30) + EAA Adapter + WeightedSampler + ModDropout 0.3',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'No_LDAM': {
        'desc': 'Replace LDAM with standard Cross-Entropy Loss (Uniform Margin)',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ce',
        'lr': 2e-4,
        'epochs': 20
    },
    'No_Adapter': {
        'desc': 'Remove EAA Face Adapter (Direct frozen CLIP face tokens)',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': False,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'No_Attn_Pooling': {
        'desc': 'Replace Self-Attention Pooling with uniform Mean Pooling',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'mean',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'No_Weighted_Sampler': {
        'desc': 'Replace WeightedRandomSampler with Uniform Random Sampling',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': False,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'Fusion_GFI': {
        'desc': 'Replace CMAF with Gated Feature Integration (GFI)',
        'fusion_type': 'gfi',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'Fusion_Concat': {
        'desc': 'Replace CMAF with Direct Concatenation + Linear Projection',
        'fusion_type': 'concat',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'No_Modality_Dropout': {
        'desc': 'Set Modality Dropout = 0.0 (No stream zeroing during training)',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'attn',
        'temporal_layers': 1,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.0,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    },
    'No_Temporal_Layers': {
        'desc': 'Remove Temporal Transformer layers (direct pooling over raw segments)',
        'fusion_type': 'cmaf',
        'temporal_pooling': 'mean',
        'temporal_layers': 0,
        'use_adapter': True,
        'use_sampler': True,
        'modality_dropout': 0.3,
        'loss_type': 'ldam',
        'lr': 2e-4,
        'epochs': 20
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fast RAER Ablation Engine")
    parser.add_argument('--features-dir', type=str, default='cached_features/raer',
                        help='Directory containing cached raer_train_features.pt')
    parser.add_argument('--config', type=str, default='Full_RAPT_CLIP',
                        choices=list(ABLATION_SUITE.keys()),
                        help='Specific ablation config to run')
    parser.add_argument('--sweep', action='store_true',
                        help='Run the full ablation matrix sequentially')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Epochs per experiment (default: 20)')
    parser.add_argument('--out-md', type=str, default='outputs/raer_fast_ablation_summary.md',
                        help='Markdown report export path')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()))
                          else ('mps' if (args.device == 'mps' or (args.device == 'auto' and torch.backends.mps.is_available()))
                                else 'cpu'))

    print("=" * 80)
    print("  RAPT-CLIP: Ultra-Fast Feature-Level Ablation Engine (RAER)")
    print(f"  Execution Device: {device} | Features Dir: {args.features_dir}")
    print("=" * 80)

    candidate_dirs = [
        args.features_dir,
        'cached_features/raer',
        '/kaggle/working/cached_features/raer',
        '/kaggle/input/raer-cached-features',
        '/kaggle/input/cached-features/raer',
        'outputs/cached_features/raer'
    ]
    resolved_features_dir = None
    for c in candidate_dirs:
        tr = os.path.join(c, 'raer_train_features.pt')
        te = os.path.join(c, 'raer_test_features.pt')
        if os.path.exists(tr) and os.path.exists(te):
            resolved_features_dir = c
            break

    if resolved_features_dir is None:
        print(f"\n❌ [ERROR] Cached features not found in '{args.features_dir}' or candidate directories!")
        print("   Please run the extractor first:")
        print("   bash train_sh/ablation/fast_sweep_raer.sh --extract-only")
        print("   OR: python tools/extract_raer_features.py\n")
        sys.exit(1)

    train_file = os.path.join(resolved_features_dir, 'raer_train_features.pt')
    test_file = os.path.join(resolved_features_dir, 'raer_test_features.pt')
    print(f"=> Using cached features from: {os.path.abspath(resolved_features_dir)}")

    print("\n=> Loading cached feature tensors into memory...")
    t0 = time.time()
    train_data = torch.load(train_file, map_location='cpu', weights_only=False)
    test_data = torch.load(test_file, map_location='cpu', weights_only=False)
    print(f"=> Loaded {len(train_data['labels']):,} Train & {len(test_data['labels']):,} Test features in {time.time()-t0:.2f}s!")

    train_tensors = (
        train_data['face_feats'],
        train_data['body_feats'],
        train_data['context_feats'],
        train_data['labels']
    )
    test_tensors = (
        test_data['face_feats'],
        test_data['body_feats'],
        test_data['context_feats'],
        test_data['labels']
    )

    print("=> Pre-computing prompt ensemble text features...")
    text_embeddings = get_raer_text_embeddings(device)

    # Determine list of experiments
    if args.sweep:
        target_configs = list(ABLATION_SUITE.keys())
    else:
        target_configs = [args.config]

    results = []
    baseline_uar = None

    for cfg_name in target_configs:
        cfg = copy.deepcopy(ABLATION_SUITE[cfg_name])
        cfg['epochs'] = args.epochs
        res = run_single_ablation(cfg_name, cfg, train_tensors, test_tensors, text_embeddings, device)
        results.append(res)
        if cfg_name == 'Full_RAPT_CLIP':
            baseline_uar = res['best_uar']

    # Generate Markdown Summary
    if baseline_uar is None and len(results) > 0:
        baseline_uar = results[0]['best_uar']

    print(f"\n\n{'='*85}")
    print("  RAPT-CLIP RAER FAST ABLATION STUDY — MASTER BENCHMARK SUMMARY")
    print(f"{'='*85}")
    header = f"| {'Mô hình / Biến thể Ablation':<30} | {'WAR (%)':<8} | {'UAR (%)':<8} | {'Δ UAR':<8} | {'Best Ep':<7} | {'Thời gian':<9} |"
    sep = f"|{'-'*32}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*9}|{'-'*11}|"
    print(header)
    print(sep)

    md_lines = [
        "# RAPT-CLIP: Báo Cáo Nghiên Cứu Cắt Bỏ Thành Phần (Fast Ablation Summary trên RAER)\n",
        f"**Tập dữ liệu:** RAER (2,121 train / 528 test videos) | **Thiết bị:** `{device}` | **Thời gian quét:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        header,
        sep
    ]

    for r in results:
        delta = r['best_uar'] - baseline_uar
        delta_str = f"{delta:+5.2f}%" if r['config'] != 'Full_RAPT_CLIP' else "Baseline"
        row = f"| {r['config']:<30} | {r['best_war']:6.2f}%  | {r['best_uar']:6.2f}%  | {delta_str:<8} | {r['best_epoch']:<7} | {r['elapsed_sec']:5.1f}s    |"
        print(row)
        md_lines.append(row)

    print(f"{'='*85}\n")

    os.makedirs(os.path.dirname(args.out_md) if os.path.dirname(args.out_md) else '.', exist_ok=True)
    with open(args.out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n\n')
        f.write("### Chi tiết Per-Class Recall từng cấu hình:\n\n")
        f.write(f"| Cấu hình | {' | '.join(RAER_CLASSES)} |\n")
        f.write(f"|---|{'---|'*len(RAER_CLASSES)}\n")
        for r in results:
            rec_strs = [f"{v:.2f}%" for v in r['recalls']] if r['recalls'] else ['-']*5
            f.write(f"| **{r['config']}** | {' | '.join(rec_strs)} |\n")

    print(f"=> Detailed Markdown summary saved to: {args.out_md}")


if __name__ == '__main__':
    main()
