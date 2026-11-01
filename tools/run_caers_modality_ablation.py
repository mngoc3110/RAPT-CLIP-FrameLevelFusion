#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_caers_modality_ablation.py
───────────────────────────────────
Thực nghiệm Modality Ablation Study trên bộ dữ liệu CAER-S (Static Images)
Đánh giá chính xác vai trò và đóng góp của từng nhánh:
1. Full Tri-Stream (Face + Body + Context)
2. W/o Body (Face + Context)
3. W/o Context (Face + Body)
4. W/o Face (Body + Context)
5. Face-Only
6. Body-Only
7. Context-Only
"""

import os
import sys
import json
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.Text import *
from models.Generate_Model import GenerateModel
from utils.checkpoint_utils import load_slim_checkpoint
from utils.builders import get_class_info
import models.clip.clip as clip

CAER_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class CAERSAblationDataset(Dataset):
    def __init__(self, test_ann_path, face_box_path, data_dir, samples_per_class=200):
        self.data_dir = data_dir
        with open(face_box_path, 'r') as f:
            self.face_boxes = json.load(f)

        all_lines = []
        with open(test_ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    p = parts[0]
                    lbl = int(parts[1])
                    all_lines.append((p, lbl))

        self.samples = []
        for c in range(len(CAER_CLASSES)):
            pool = [s for s in all_lines if s[1] == c]
            random.seed(42)
            random.shuffle(pool)
            chosen = pool[:samples_per_class]
            self.samples.extend(chosen)
            print(f"   • Class {c} ({CAER_CLASSES[c]}): {len(chosen)} samples")

        print(f"=> Total Dataset: {len(self.samples)} test samples.")
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
            # Body crop with face region masked to avoid face leakage
            bx1, by1 = max(0, int(x1 - 0.5 * bw)), max(0, y1)
            bx2, by2 = min(w, int(x2 + 0.5 * bw)), min(h, int(y2 + 2.5 * bh))
            body_crop = raw[by1:by2, bx1:bx2].copy()
            # Mask face inside body crop
            rel_fx1 = max(0, x1 - bx1)
            rel_fy1 = max(0, y1 - by1)
            rel_fx2 = min(body_crop.shape[1], x2 - bx1)
            rel_fy2 = min(body_crop.shape[0], y2 - by1)
            body_crop[rel_fy1:rel_fy2, rel_fx1:rel_fx2] = 0
        else:
            face_crop = body_crop = raw

        if face_crop.size == 0: face_crop = raw
        if body_crop.size == 0: body_crop = raw

        t_f = self.prep_t(face_crop)
        t_b = self.prep_t(body_crop)
        t_c = self.prep_t(raw)

        return t_f, t_b, t_c, lbl


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"=> Device: {device}")

    test_ann = os.path.join(ROOT_DIR, 'CAER-S', 'annotations', 'test.txt')
    face_box = os.path.join(ROOT_DIR, 'CAER-S', 'annotations', 'caer_s_faces.json')
    data_dir = os.path.join(ROOT_DIR, 'CAER-S')
    ckpt_path = os.path.join(ROOT_DIR, 'caer_s_finetune_end.pth')

    # Samples per class (200 * 7 = 1,400 balanced test samples)
    dataset = CAERSAblationDataset(test_ann, face_box, data_dir, samples_per_class=200)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Build Model
    class ModelArgs:
        dataset = 'CAER-S'
        batch_size = 64
        workers = 0
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
        lr_image_encoder = 0.0
        freeze_image_encoder = True
        lr_prompt_learner = 1e-4

    m_args = ModelArgs()
    m_args.device = device
    clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_model = clip_model.float()
    class_names, input_text = get_class_info(m_args)
    m_args.num_classes = len(class_names)

    model = GenerateModel(input_text, clip_model, m_args).to(device).float()
    load_slim_checkpoint(model, ckpt_path, device=device)
    model.eval()

    # 1. Feature Extraction
    cache_path = os.path.join(ROOT_DIR, 'outputs', 'caers_ablation_features.pt')
    if os.path.exists(cache_path):
        print(f"=> Loading pre-extracted CAER-S features from {cache_path}...")
        cache = torch.load(cache_path, map_location='cpu')
        all_f_feat = cache['f_feat']
        all_b_feat = cache['b_feat']
        all_c_feat = cache['c_feat']
        targets_all = cache['targets']
    else:
        print("=> Extracting multi-stream features for CAER-S...")
        all_f_feat, all_b_feat, all_c_feat, targets_list = [], [], [], []
        with torch.no_grad():
            for tf, tb, tc, lbl in tqdm(loader, desc="Extracting"):
                tf = tf.to(device).float()
                tb = tb.to(device).float()
                tc = tc.to(device).float()

                f_f = model.image_encoder(tf)
                if hasattr(model, 'face_adapter'):
                    f_f = model.face_adapter(f_f)

                b_f = model.image_encoder(tb)
                c_f = model.image_encoder(tc)

                all_f_feat.append(f_f.cpu())
                all_b_feat.append(b_f.cpu())
                all_c_feat.append(c_f.cpu())
                targets_list.append(lbl)

        all_f_feat = torch.cat(all_f_feat, dim=0)
        all_b_feat = torch.cat(all_b_feat, dim=0)
        all_c_feat = torch.cat(all_c_feat, dim=0)
        targets_all = torch.cat(targets_list, dim=0).numpy()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'f_feat': all_f_feat,
            'b_feat': all_b_feat,
            'c_feat': all_c_feat,
            'targets': targets_all
        }, cache_path)
        print(f"=> Cached features to {cache_path}")

    # 2. Compute Text Embeddings
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized_prompts = model.tokenized_prompts
        text_features = model.text_encoder(prompts, tokenized_prompts).float()
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_features = text_features.view(len(CAER_CLASSES), model.num_prompts_per_class, -1)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

    # 3. Universal Modular Evaluation
    def eval_variant(use_f=True, use_b=True, use_c=True):
        preds = []
        batch_size = 64
        total_samples = len(targets_all)
        num_batches = int(np.ceil(total_samples / batch_size))

        with torch.no_grad():
            for b_idx in range(num_batches):
                start = b_idx * batch_size
                end = min((b_idx + 1) * batch_size, total_samples)

                f_b = all_f_feat[start:end].to(device) if use_f else torch.zeros_like(all_f_feat[start:end]).to(device)
                b_b = all_b_feat[start:end].to(device) if use_b else torch.zeros_like(all_b_feat[start:end]).to(device)
                c_b = all_c_feat[start:end].to(device) if use_c else torch.zeros_like(all_c_feat[start:end]).to(device)

                fused = model.cmaf(f_b, b_b, c_b)
                vid = model.project_fc(fused)
                vid = vid / (vid.norm(dim=-1, keepdim=True) + 1e-6)

                # Similarity with Text Prompts
                # vid: (B, 512), text_features: (7, 8, 512)
                sim = torch.einsum('bd,ckd->bck', vid, text_features).mean(dim=-1)
                pred = sim.argmax(dim=-1).cpu()
                preds.append(pred)

        preds = torch.cat(preds, dim=0).numpy()
        cm = confusion_matrix(targets_all, preds, labels=range(len(CAER_CLASSES)))
        war = np.mean(preds == targets_all) * 100.0
        recalls = cm.diagonal() / cm.sum(axis=1) * 100.0
        uar = np.mean(recalls)
        return war, uar, recalls, cm

    configs = [
        ("Full Tri-Stream (Face+Body+Context)", True,  True,  True),
        ("W/o Body (Face + Context)",          True,  False, True),
        ("W/o Context (Face + Body)",          True,  True,  False),
        ("W/o Face (Body + Context)",          False, True,  True),
        ("Face-Only",                          True,  False, False),
        ("Body-Only",                          False, True,  False),
        ("Context-Only",                       False, False, True),
    ]

    print("\n" + "="*110)
    print("🏆 BẢNG KẾT QUẢ MODALITY ABLATION TRÊN BỘ DỮ LIỆU CAER-S (1,400 TEST SAMPLES - 200/LỚP)")
    print("="*110)
    header = f"{'Cấu hình nhánh (Modality)':<35} | {'WAR (%)':<8} | {'UAR (%)':<8} | " + " | ".join([f"{c[:4]:<5}" for c in CAER_CLASSES])
    print(header)
    print("-" * 110)

    results = {}
    for name, uf, ub, uc in configs:
        war, uar, recs, cm = eval_variant(uf, ub, uc)
        rec_str = " | ".join([f"{r:>5.1f}%" for r in recs])
        print(f"{name:<35} | {war:>7.2f}% | {uar:>7.2f}% | {rec_str}")
        results[name] = {
            'war': war, 'uar': uar,
            'recalls': {CAER_CLASSES[i]: float(recs[i]) for i in range(len(CAER_CLASSES))},
            'cm': cm.tolist()
        }
    print("="*110)

    out_json = os.path.join(ROOT_DIR, 'outputs', 'caers_modality_ablation_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"=> Saved results to {out_json}")

if __name__ == '__main__':
    main()

