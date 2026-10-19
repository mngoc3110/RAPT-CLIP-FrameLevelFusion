#!/usr/bin/env python3
"""
Evaluate CAER-S checkpoint with Standard vs Test-Time Augmentation (TTA).
Prints a side-by-side comparison table showing WAR, UAR, and per-class recall gains.
"""

import os
import sys
import argparse
import numpy as np
import torch
import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.checkpoint_utils import load_slim_checkpoint
from utils.builders import build_dataloaders, build_model, get_class_info
from utils.utils import plot_confusion_matrix


def evaluate_stream(loader, model, device, class_names, use_tta=False):
    model.eval()
    all_preds = []
    all_targets = []
    
    desc_str = f"Evaluating ({'TTA (Horizontal Flip)' if use_tta else 'Standard'})"
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc=desc_str):
            if len(batch) == 4:
                images_face, images_body, images_context, target = batch
                images_context = images_context.to(device)
            else:
                images_face, images_body, target = batch
                images_context = None

            images_face = images_face.to(device)
            images_body = images_body.to(device)
            target = target.to(device)

            if images_context is not None:
                output, _, _, _ = model(images_face, images_body, images_context)
                if use_tta:
                    f_face = torch.flip(images_face, dims=[-1])
                    f_body = torch.flip(images_body, dims=[-1])
                    f_ctx = torch.flip(images_context, dims=[-1])
                    output_flip, _, _, _ = model(f_face, f_body, f_ctx)
                    output = (output.softmax(dim=-1) + output_flip.softmax(dim=-1)) / 2.0
            else:
                output, _, _, _ = model(images_face, images_body)
                if use_tta:
                    f_face = torch.flip(images_face, dims=[-1])
                    f_body = torch.flip(images_body, dims=[-1])
                    output_flip, _, _, _ = model(f_face, f_body)
                    output = (output.softmax(dim=-1) + output_flip.softmax(dim=-1)) / 2.0

            predicted = output.argmax(dim=1)
            all_preds.append(predicted.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds, 0).numpy()
    all_targets = torch.cat(all_targets, 0).numpy()

    correct = (all_preds == all_targets).sum()
    war = 100.0 * correct / len(all_targets)

    cm = confusion_matrix(all_targets, all_preds, labels=range(len(class_names)))
    class_recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    class_recall[np.isnan(class_recall)] = 0.0
    uar = np.mean(class_recall) * 100.0

    cm_norm = (cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)) * 100.0
    return war, uar, class_recall * 100.0, cm_norm


def main():
    parser = argparse.ArgumentParser(description="CAER-S TTA Benchmark Evaluator")
    parser.add_argument('--checkpoint', type=str, default='/kaggle/input/models/traincollab/caerv2-funing/tensorflow2/default/1/model_best.pth', help='Path to checkpoint (model_best.pth or model.pth)')
    parser.add_argument('--data-dir', type=str, default='/kaggle/input/datasets/lcngtr/caer-s', help='CAER-S root dir')
    parser.add_argument('--annotation-dir', type=str, default='/kaggle/working/caers_data', help='Annotations dir')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Num workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output-dir', type=str, default='outputs/eval_tta', help='Output directory')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset paths
    train_txt = os.path.join(args.annotation_dir, 'train.txt')
    val_txt = os.path.join(args.annotation_dir, 'validation.txt')
    test_txt = os.path.join(args.annotation_dir, 'test.txt')
    if not os.path.isfile(val_txt) and os.path.isfile(test_txt):
        val_txt = test_txt

    face_box = os.path.join(args.annotation_dir, 'face.json')
    body_box = os.path.join(args.annotation_dir, 'body.json')

    # Construct mock args for builders
    class MockArgs:
        dataset = 'CAER-S'
        device = device
        batch_size = args.batch_size
        workers = args.workers
        root_dir = args.data_dir
        train_annotation = train_txt
        val_annotation = val_txt
        test_annotation = test_txt
        bounding_box_face = face_box
        bounding_box_body = body_box
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
        use_npy = False
        use_amp = True

    m_args = MockArgs()
    class_names, input_text = get_class_info(m_args)
    m_args.num_classes = len(class_names)

    print("=> Building model...")
    model = build_model(m_args, input_text)
    model = model.to(device)

    print(f"=> Loading weights from: {args.checkpoint}")
    load_slim_checkpoint(model, args.checkpoint, device=device)

    print("=> Building dataloader...")
    _, _, test_loader = build_dataloaders(m_args)

    # 1. Standard Run
    print("\n--- [1/2] Standard Evaluation (No Augmentation) ---")
    std_war, std_uar, std_recalls, std_cm = evaluate_stream(test_loader, model, device, class_names, use_tta=False)

    # 2. TTA Run
    print("\n--- [2/2] Test-Time Augmentation Evaluation (Horizontal Flip) ---")
    tta_war, tta_uar, tta_recalls, tta_cm = evaluate_stream(test_loader, model, device, class_names, use_tta=True)

    # 3. Print Comparison Table
    war_gain = tta_war - std_war
    uar_gain = tta_uar - std_uar

    print("\n" + "=" * 80)
    print("🏆 [CAER-S BENCHMARK COMPARISON: STANDARD vs TEST-TIME AUGMENTATION (TTA)]")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Standard':<12} | {'With TTA':<12} | {'Improvement':<15}")
    print("-" * 80)
    print(f"{'Overall WAR (Accuracy)':<25} | {std_war:>10.2f}% | {tta_war:>10.2f}% | {war_gain:>+13.2f}% {'🚀' if war_gain > 0 else ''}")
    print(f"{'Overall UAR (Balanced)':<25} | {std_uar:>10.2f}% | {tta_uar:>10.2f}% | {uar_gain:>+13.2f}% {'🚀' if uar_gain > 0 else ''}")
    print("-" * 80)
    print("Per-Class Recall:")
    for idx, name in enumerate(class_names):
        s_rec = std_recalls[idx]
        t_rec = tta_recalls[idx]
        g = t_rec - s_rec
        print(f"  [{idx}] {name:<18} | {s_rec:>10.2f}% | {t_rec:>10.2f}% | {g:>+13.2f}%")
    print("=" * 80)

    # 4. Save Confusion Matrices
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(tta_cm, classes=class_names, normalize=True, title=f"CAER-S Confusion Matrix with TTA (WAR: {tta_war:.2f}%)")
    cm_path = os.path.join(args.output_dir, "confusion_matrix_tta.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"\n=> Saved TTA confusion matrix plot to: {cm_path}\n")


if __name__ == '__main__':
    main()
