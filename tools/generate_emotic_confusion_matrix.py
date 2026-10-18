import os
import sys
import json
import argparse
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath('.'))

from main import get_class_info, build_model
from utils.builders import build_dataloaders
from utils.checkpoint_utils import load_slim_checkpoint

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = 'outputs/EMOTIC'
    checkpoint_path = os.path.join(output_dir, 'model_best.zip')
    preds_cache_path = os.path.join(output_dir, 'test_preds.npy')
    targets_cache_path = os.path.join(output_dir, 'test_targets.npy')

    # Load thresholds
    thresh_path = os.path.join(output_dir, 'emotic_threshold.json')
    if not os.path.exists(thresh_path):
        thresh_path = os.path.join(output_dir, 'emotic_thresholds.json')
    
    with open(thresh_path, 'r') as f:
        thresholds_dict = {int(k): float(v) for k, v in json.load(f).items()}
    print(f"Loaded {len(thresholds_dict)} thresholds from {thresh_path}")

    args = argparse.Namespace(
        dataset='EMOTIC',
        mode='eval',
        clip_path='ViT-B/16',
        use_npy=True,
        npy_dir='emotic_dataset/emotic_pre',
        batch_size=32,
        workers=2,
        seed=42,
        gpu='mps',
        device=device,
        image_size=224,
        text_type='prompt_ensemble',
        contexts_number=8,
        ctx_init='',
        class_token_position='end',
        class_specific_contexts=True,
        load_and_tune_prompt_learner=True,
        num_segments=1,
        duration=1,
        temperature=1.0,
        crop_body=True,
        fusion_type='cmaf',
        fusion_stage='frame',
        use_context=True,
        modality_dropout=0.3,
        drop_path_rate=0.0,
        freeze_image_encoder=False,
        lr_image_encoder=1e-5,
        no_temporal=False,
        use_temporal=True,
        temporal_layers=1,
        ablation_no_text=False,
        use_v2=False,
        mask_context_face=False,
        daisee_merge_low=False,
        eval_checkpoint=checkpoint_path,
        root_dir='./',
        train_annotation='emotic_dataset/train_bbox.txt',
        val_annotation='emotic_dataset/val_bbox.txt',
        test_annotation='emotic_dataset/test_bbox.txt',
        bounding_box_face='emotic_dataset/emotic_face_bboxes.json',
        bounding_box_body='emotic_dataset/emotic_body_bboxes.json',
        output_path=output_dir
    )

    class_names, input_text = get_class_info(args)
    num_classes = len(class_names)
    thresh_array = np.array([thresholds_dict.get(c, 0.5) for c in range(num_classes)])

    if os.path.exists(preds_cache_path) and os.path.exists(targets_cache_path):
        print(f"Loading cached predictions from {preds_cache_path}...")
        all_preds = np.load(preds_cache_path)
        all_targets = np.load(targets_cache_path)
    else:
        model = build_model(args, input_text).to(device)
        print(f"=> Loading checkpoint from {checkpoint_path}...")
        load_slim_checkpoint(model, checkpoint_path, device=device)
        model.eval()

        _, _, test_loader = build_dataloaders(args)
        all_preds_list = []
        all_targets_list = []

        print(f"=> Running inference on Test set ({len(test_loader.dataset)} samples)...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="[EMOTIC Test Inference]"):
                face, body, context, target = [x.to(device) if x is not None else None for x in batch]
                out = model(face, body, context)[0]
                probs = torch.sigmoid(out).cpu().numpy()
                all_preds_list.append(probs)
                all_targets_list.append(target.cpu().numpy())

        all_preds = np.concatenate(all_preds_list, axis=0)
        all_targets = np.concatenate(all_targets_list, axis=0)
        np.save(preds_cache_path, all_preds)
        np.save(targets_cache_path, all_targets)
        print(f"Saved test predictions cache to {preds_cache_path}")

    # Binary predictions using optimal thresholds
    bin_preds = (all_preds >= thresh_array).astype(int)

    # 1. Multi-label Co-occurrence / Confusion Matrix (26x26)
    # C[i, j] = P(predict j | true i) = sum(y_i * y_pred_j) / sum(y_i)
    cm_multilabel = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        true_mask = (all_targets[:, i] == 1)
        total_true = np.sum(true_mask)
        if total_true > 0:
            cm_multilabel[i, :] = np.sum(bin_preds[true_mask, :], axis=0) / total_true * 100.0
        else:
            cm_multilabel[i, :] = 0.0

    # 2. Dominant-Class Confusion Matrix (Argmax mapping)
    cm_dominant = np.zeros((num_classes, num_classes), dtype=float)
    # Samples with at least 1 positive target
    has_target = np.sum(all_targets, axis=1) > 0
    sub_targets = all_targets[has_target]
    sub_preds = all_preds[has_target]
    true_dominant = np.argmax(sub_targets, axis=1)
    pred_dominant = np.argmax(sub_preds, axis=1)
    for t_idx, p_idx in zip(true_dominant, pred_dominant):
        cm_dominant[t_idx, p_idx] += 1
    # Row normalize
    row_sums = cm_dominant.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_dominant_norm = (cm_dominant / row_sums) * 100.0

    # Plot 1: Publication Quality Multi-label Normalized Confusion Heatmap
    plt.figure(figsize=(18, 15))
    sns.set_theme(style="white", font="sans-serif")
    
    # Custom color palette with subtle grid
    ax = sns.heatmap(
        cm_multilabel,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Tỷ lệ kích hoạt dự đoán (% | True Label Active)'},
        linewidths=0.5,
        linecolor='#e2e8f0',
        annot_kws={"size": 7.5, "weight": "normal"}
    )
    
    plt.title(
        "Ma trận nhầm lẫn đa nhãn (Multi-Label Confusion Matrix) trên tập TEST của EMOTIC\n"
        "[RAPT-CLIP model_best.zip | Macro mAP: 34.76% | Đường chéo: Recall (%)]",
        fontsize=14, weight="bold", pad=15
    )
    plt.xlabel("Nhãn cảm xúc dự đoán (Predicted Class)", fontsize=12, weight="bold", labelpad=10)
    plt.ylabel("Nhãn cảm xúc thực tế (Ground-Truth Class)", fontsize=12, weight="bold", labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    out_fig_multilabel = os.path.join(output_dir, 'confusion_matrix_test.png')
    plt.savefig(out_fig_multilabel, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Saved Multi-label Confusion Matrix to {out_fig_multilabel}")

    # Plot 2: Dominant Emotion Confusion Matrix (26x26)
    plt.figure(figsize=(18, 15))
    ax2 = sns.heatmap(
        cm_dominant_norm,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Tỷ lệ nhầm lẫn chuẩn hóa theo hàng (%)'},
        linewidths=0.5,
        linecolor='#e2e8f0',
        annot_kws={"size": 7.5, "weight": "normal"}
    )
    plt.title(
        "Ma trận nhầm lẫn cảm xúc chủ đạo (Dominant Emotion Confusion Matrix) - EMOTIC Test\n"
        "[Argmax Prediction vs Dominant Ground Truth | RAPT-CLIP]",
        fontsize=14, weight="bold", pad=15
    )
    plt.xlabel("Nhãn cảm xúc dự đoán (Predicted Dominant Class)", fontsize=12, weight="bold", labelpad=10)
    plt.ylabel("Nhãn cảm xúc thực tế (Ground-Truth Dominant Class)", fontsize=12, weight="bold", labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()

    out_fig_dominant = os.path.join(output_dir, 'confusion_matrix_dominant_test.png')
    plt.savefig(out_fig_dominant, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"=> Saved Dominant Confusion Matrix to {out_fig_dominant}")

    # Copy to standard confusion_matrix.png in outputs/EMOTIC
    import shutil
    shutil.copyfile(out_fig_multilabel, os.path.join(output_dir, 'confusion_matrix.png'))
    print(f"=> Synced to {os.path.join(output_dir, 'confusion_matrix.png')}")

if __name__ == '__main__':
    main()
