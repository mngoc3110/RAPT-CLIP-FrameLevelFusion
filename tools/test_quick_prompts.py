#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/test_quick_prompts.py
───────────────────────────
Công cụ kiểm chứng siêu tốc (chạy trong 2-3 giây trên Mac/MPS):
So sánh độ chính xác của các cách thiết kế Prompt khác nhau trên tập Test RAER (528 videos).

Bao gồm 3 thí nghiệm trực diện:
1. Standard Prompt (Chỉ có tên nhãn: "A photo of a [CLASS] student.")
2. Pure Visual Cues (HOÀN TOÀN KHÔNG CÓ TÊN NHÃN: chỉ miêu tả hành vi ngáp, sụp mí, nhìn đi chỗ khác...)
3. Hybrid Enriched Prompt (Hiện tại trong Text.py: Kết hợp cả nhãn và đặc tả hành vi)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip

CLASSES = ["Neutral", "Enjoyment", "Confusion", "Fatigue", "Distraction"]

# 1. Cấu hình Prompt chỉ có tên nhãn kinh điển (Classic Label-only)
PROMPTS_LABEL_ONLY = [
    ["A photo of a neutral student."],
    ["A photo of an enjoyment student."],
    ["A photo of a confused student."],
    ["A photo of a fatigued student."],
    ["A photo of a distracted student."]
]

# 2. Cấu hình Prompt HOÀN TOÀN KHÔNG CÓ TÊN NHÃN (Pure Observable Physical Cues)
PROMPTS_NO_LABEL = [
    [   # Lớp 0 (Neutral) - KHÔNG CÓ TỪ NEUTRAL
        "A photo of a student being alert and looking straight ahead.",
        "A photo of a student with a calm and steady gaze at the board.",
        "A photo of a student sitting still and listening attentively."
    ],
    [   # Lớp 1 (Enjoyment) - KHÔNG CÓ TỪ ENJOYMENT
        "A photo of a student smiling brightly and laughing.",
        "A photo of a student showing enthusiasm with crinkled eyes.",
        "A photo of a student appearing delighted and engaged."
    ],
    [   # Lớp 2 (Confusion) - KHÔNG CÓ TỪ CONFUSION
        "A photo of a student frowning with furrowed eyebrows and a puzzled look.",
        "A photo of a student scratching their head in uncertainty.",
        "A photo of a student squinting with a perplexed open mouth."
    ],
    [   # Lớp 3 (Fatigue) - KHÔNG CÓ TỪ FATIGUE
        "A photo of a student yawning or falling asleep at the desk.",
        "A photo of a student with heavy drooping eyelids.",
        "A photo of a student resting their head on hands, looking exhausted."
    ],
    [   # Lớp 4 (Distraction) - KHÔNG CÓ TỪ DISTRACTION
        "A photo of a student looking away from the lecture.",
        "A photo of a student turning their head completely to the side.",
        "A photo of a student playing with objects, not paying attention."
    ]
]

# 3. Cấu hình trong Text.py (prompt_ensemble_5)
from models.Text import prompt_ensemble_5

def evaluate_prompt_set(clip_model, prompt_set, visual_features, targets, device, name=""):
    num_classes = len(prompt_set)
    text_embeddings = []
    
    with torch.no_grad():
        for class_prompts in prompt_set:
            tokens = clip.tokenize(class_prompts).to(device)
            feats = clip_model.encode_text(tokens).float()
            feats = F.normalize(feats, dim=-1)
            # Trung bình ensemble cho lớp đó
            class_feat = feats.mean(dim=0)
            class_feat = F.normalize(class_feat, dim=-1)
            text_embeddings.append(class_feat)
            
        text_embeddings = torch.stack(text_embeddings, dim=0) # [5, 512]
        
        # visual_features: [528, 512]
        sims = torch.matmul(visual_features, text_embeddings.T) # [528, 5]
        preds = torch.argmax(sims, dim=-1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        
    correct = (preds == targets_np).sum()
    war = (correct / len(targets_np)) * 100.0
    
    recalls = []
    for c in range(num_classes):
        mask = (targets_np == c)
        if mask.sum() > 0:
            rec = ((preds == c) & mask).sum() / mask.sum() * 100.0
        else:
            rec = 0.0
        recalls.append(rec)
    uar = np.mean(recalls)
    
    return war, uar, recalls

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"⚡ Thiết bị chạy test nhanh: {device}")
    
    cache_path = os.path.join(ROOT_DIR, "outputs", "raer_extracted_features.pt")
    if not os.path.exists(cache_path):
        print(f"❌ Không tìm thấy cache: {cache_path}")
        return
        
    print(f"📦 Đang nạp clip model ViT-B/16 và 528 test features...")
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    num_videos = len(data['targets_all'])
    face_all = data['face_all'].view(num_videos, 16, 512).to(device)
    face_video_feats = face_all.mean(dim=1) # [528, 512]
    face_video_feats = F.normalize(face_video_feats, dim=-1)
    targets_raw = data['targets_all']
    if isinstance(targets_raw, np.ndarray):
        targets = torch.from_numpy(targets_raw).to(device)
    else:
        targets = targets_raw.to(device)
    
    from models.Text import class_descriptor_5_only_face
    test_configs = [
        ("1. Standard (Chỉ có tên nhãn: 'A photo of a [CLASS] student')", PROMPTS_LABEL_ONLY),
        ("2. Pure Visual Cues (KHÔNG CÓ TÊN NHÃN: ngáp, sụp mí, nhìn lệch...)", PROMPTS_NO_LABEL),
        ("3. Prompt Ensemble 5 (Hiện tại trong Text.py)", prompt_ensemble_5),
        ("4. Class Descriptors Face (Câu neo MI Loss)", [[p] for p in class_descriptor_5_only_face]),
    ]
    
    print("\n" + "="*78)
    print(" ⚡ KẾT QUẢ TEST NHANH ZERO-SHOT TRÊN TẬP TEST RAER (528 VIDEOS - 3 GIÂY)")
    print("="*78)
    print(f"{'Cấu hình Prompt':<48} | {'WAR (%)':<8} | {'UAR (%)':<8}")
    print("-" * 78)
    
    for name, p_set in test_configs:
        war, uar, recs = evaluate_prompt_set(clip_model, p_set, face_video_feats, targets, device, name)
        print(f"{name:<48} | {war:6.2f}%  | {uar:6.2f}%")
        rec_str = " | ".join([f"{CLASSES[i][:4]}: {recs[i]:.1f}%" for i in range(5)])
        print(f"   ↳ Recall: {rec_str}")
        print("-" * 78)
    print("="*78)

if __name__ == "__main__":
    main()
