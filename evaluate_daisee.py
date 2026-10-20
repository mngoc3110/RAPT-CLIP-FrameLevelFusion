#!/usr/bin/env python3
"""
RAPT-CLIP DAiSEE Test Set Evaluation Script
Optimized for Apple Silicon (M1/M2/M3/M4, MPS) and CUDA/CPU.
"""

import os
import sys

# Ensure immediate real-time log printing in Kaggle/Colab notebooks
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Project imports
from utils.builders import get_class_info
from models.Generate_Model import GenerateModel
from models.clip import clip
from dataloader.video_dataloader import test_data_loader
from utils.checkpoint_utils import load_slim_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAPT-CLIP on DAiSEE Test Set")
    parser.add_argument('--checkpoint', type=str, default='model_daisee.zip',
                        help="Path to checkpoint (model_daisee.zip, model_clean.pth, or model_best.pth)")
    parser.add_argument('--root-dir', type=str, default='DAiSEE',
                        help="Path to DAiSEE root directory")
    parser.add_argument('--test-annotation', type=str, default='DAiSEE/daisee_test.txt',
                        help="Path to daisee_test.txt")
    parser.add_argument('--bounding-box-face', type=str, default='DAiSEE/face_bbox_dummy.json',
                        help="Face bbox JSON file")
    parser.add_argument('--bounding-box-body', type=str, default='DAiSEE/body_bbox_dummy.json',
                        help="Body bbox JSON file")
    parser.add_argument('--batch-size', type=int, default=16,
                        help="Batch size for evaluation (default: 16)")
    parser.add_argument('--workers', type=int, default=2,
                        help="Number of dataloader workers")
    parser.add_argument('--num-segments', type=int, default=8,
                        help="Number of temporal segments/frames sampled")
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help="Device to run inference on (default: auto)")
    parser.add_argument('--clip-path', type=str, default='ViT-B/16',
                        help="CLIP backbone architecture")
    parser.add_argument('--daisee-merge-low', action='store_true', default=True,
                        help="Merge class 0 and 1 into 3-class setup (Low, High, Very High)")
    return parser.parse_args()


def select_device(device_arg):
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    return device


def resolve_checkpoint(path):
    if os.path.exists(path):
        return path
    fallbacks = ['model_clean.pth', 'model_daisee.zip', 'model_best.pth']
    for fb in fallbacks:
        if os.path.exists(fb):
            print(f"[Warning] '{path}' not found. Falling back to '{fb}'.")
            return fb
    # Check in /kaggle/input if running on Kaggle
    if os.path.isdir('/kaggle/input'):
        import glob
        matches = glob.glob(f"/kaggle/input/**/{os.path.basename(path)}", recursive=True)
        if matches:
            print(f"[Auto-Detect] Found checkpoint at: {matches[0]}")
            return matches[0]
    raise FileNotFoundError(f"Checkpoint not found at '{path}' and no fallback available.")


def auto_detect_daisee_paths(root_dir, test_annotation, bbox_face):
    """
    Intelligently discovers DAiSEE root directory, annotation file, and face boxes
    across Kaggle (/kaggle/input), Colab (/content), and local environments.
    """
    import glob

    # 1. Resolve root_dir
    possible_roots = [
        root_dir,
        "DAiSEE",
        "DAiSEE/DataSet",
        "/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE",
        "/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE/DataSet",
        "/kaggle/input/datasets/olgaparfenova/daisee",
        "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data",
        "/kaggle/input/datasets/mngochocsupham/daisee",
        "/kaggle/input/daisee/DAiSEE_data",
        "/kaggle/input/daisee/DAiSEE",
        "/kaggle/input/daisee",
        "/kaggle/input/DAiSEE_data",
        "/kaggle/input/DAiSEE",
        "/content/DAiSEE",
        "/content/drive/MyDrive/DAiSEE"
    ]
    resolved_root = root_dir
    for r in possible_roots:
        if r and os.path.exists(r) and (
            os.path.isdir(os.path.join(r, "DataSet")) or
            os.path.isdir(os.path.join(r, "Labels")) or
            os.path.exists(os.path.join(r, "daisee_test.txt")) or
            os.path.exists(os.path.join(r, "Validation.txt"))
        ):
            resolved_root = r
            break
            
    if (not os.path.exists(resolved_root) or resolved_root == "DAiSEE") and os.path.isdir("/kaggle/input"):
        matches = glob.glob("/kaggle/input/**/daisee_test.txt", recursive=True) + \
                  glob.glob("/kaggle/input/**/TestLabels.csv", recursive=True) + \
                  glob.glob("/kaggle/input/**/Validation.txt", recursive=True)
        if matches:
            parent = os.path.dirname(matches[0])
            if os.path.basename(parent) in ["DataSet", "Labels"]:
                resolved_root = os.path.dirname(parent)
            else:
                resolved_root = parent

    # 2. Resolve test_annotation
    resolved_test = test_annotation
    if not os.path.exists(resolved_test):
        candidate_tests = [
            os.path.join(resolved_root, "daisee_test.txt"),
            os.path.join(resolved_root, "DataSet", "Test.txt"),
            os.path.join(resolved_root, "Test.txt"),
            os.path.join(resolved_root, "Labels", "TestLabels.csv"),
            os.path.join("DAiSEE", "daisee_test.txt"),
            "daisee_test.txt",
        ]
        for t in candidate_tests:
            if os.path.exists(t):
                resolved_test = t
                break
                
        if not os.path.exists(resolved_test) and os.path.isdir("/kaggle/input"):
            matches = glob.glob("/kaggle/input/**/daisee_test.txt", recursive=True) + \
                      glob.glob("/kaggle/input/**/TestLabels.csv", recursive=True) + \
                      glob.glob("/kaggle/input/**/Test.txt", recursive=True)
            if matches:
                resolved_test = matches[0]

    # 3. Resolve bbox_face
    resolved_bbox = bbox_face
    if not os.path.exists(resolved_bbox) or os.path.getsize(resolved_bbox) < 100:
        candidate_bboxes = [
            os.path.join(resolved_root, "daisee_face_bbox.json"),
            os.path.join(resolved_root, "face_bbox.json"),
            "/kaggle/working/daisee_face_bbox.json",
            "DAiSEE/daisee_face_bbox.json",
            "daisee_face_bbox.json",
        ]
        resolved_bbox = "none"
        for b in candidate_bboxes:
            if os.path.exists(b) and os.path.getsize(b) > 100:
                resolved_bbox = b
                break

    return resolved_root, resolved_test, resolved_bbox


def main():
    args = parse_args()
    device = select_device(args.device)
    
    # Auto-resolve dataset & annotation paths
    args.root_dir, args.test_annotation, args.bounding_box_face = auto_detect_daisee_paths(
        args.root_dir, args.test_annotation, args.bounding_box_face
    )

    print(f"\n===========================================================")
    print(f"🚀 RAPT-CLIP DAiSEE EVALUATION")
    print(f"===========================================================")
    print(f"• Hardware Device:    {device} ({'Apple Silicon Metal' if device.type == 'mps' else device.type.upper()})")
    
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"• Checkpoint Path:    {ckpt_path}")
    print(f"• Dataset Root:       {args.root_dir}")
    print(f"• Test Annotation:    {args.test_annotation}")
    print(f"• Face Bounding Box:  {args.bounding_box_face}")
    print(f"• Batch Size:         {args.batch_size}")
    print(f"• Num Segments:       {args.num_segments}")
    print(f"===========================================================\n")

    if not os.path.exists(args.test_annotation):
        print(f"❌ [LỖI] Không tìm thấy file test annotation tại: '{args.test_annotation}'")
        print("👉 Vui lòng truyền đường dẫn chính xác bằng:")
        print("   python evaluate_daisee.py --checkpoint ... --root-dir <THƯ_MỤC_DAiSEE> --test-annotation <FILE_TEST.txt>")
        sys.exit(1)

    # Model configuration namespace
    model_args = argparse.Namespace(
        dataset='DAiSEE',
        root_dir=args.root_dir,
        test_annotation=args.test_annotation,
        bounding_box_face=args.bounding_box_face,
        bounding_box_body=args.bounding_box_body,
        batch_size=args.batch_size,
        workers=args.workers,
        num_segments=args.num_segments,
        duration=1,
        image_size=224,
        use_context=False,
        crop_body=False,
        mask_context_face=False,
        daisee_merge_low=args.daisee_merge_low,
        use_npy=False,
        text_type='prompt_ensemble',
        clip_path=args.clip_path,
        temporal_layers=1,
        fusion_type='cmaf',
        fusion_stage='frame',
        contexts_number=8,
        class_token_position='end',
        class_specific_contexts=True,
        load_and_tune_prompt_learner=True,
        drop_path_rate=0.0,
        modality_dropout=0.0,
        temperature=0.07,
        freeze_image_encoder=False,
        ablation_no_text=False,
        device=device
    )

    class_names, input_text = get_class_info(model_args)
    model_args.num_classes = len(class_names)
    print(f"=> Classes ({model_args.num_classes}): {class_names}")

    # Build model & ensure float32 for MPS stability
    print(f"=> Loading CLIP {args.clip_path} backbone...")
    clip_model, _ = clip.load(model_args.clip_path, device=device, jit=False)
    clip_model.float()

    model = GenerateModel(input_text, clip_model, model_args)
    model = model.float().to(device)

    # Load weights
    print(f"=> Loading model weights from: {ckpt_path}...")
    ckpt = load_slim_checkpoint(model, ckpt_path, device=device)
    epoch_info = ckpt.get('epoch', 'N/A')
    best_acc_info = ckpt.get('best_acc', 'N/A')
    print(f"   ✓ Checkpoint metadata: Epoch = {epoch_info}, Validation Best Acc = {best_acc_info}")

    # Build test loader
    print("\n=> Initializing DAiSEE Test DataLoader...")
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

    if len(test_data) == 0:
        print("\n❌ [LỖI] DataLoader không tìm thấy video test nào (0 samples loaded)!")
        print(f"   • Root dir đã kiểm tra: {model_args.root_dir}")
        print(f"   • File annotation:     {model_args.test_annotation}")
        print("👉 Vui lòng chỉ định đúng đường dẫn DAiSEE trên máy hoặc Kaggle bằng cách thêm tham số:")
        print("   --root-dir /path/to/DAiSEE --test-annotation /path/to/daisee_test.txt")
        sys.exit(1)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_data,
        batch_size=model_args.batch_size,
        shuffle=False,
        num_workers=model_args.workers,
        pin_memory=(device.type == 'cuda')
    )
    print(f"=> Total Test samples: {len(test_data)} across {len(test_loader)} batches")

    # Run inference
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []

    total_batches = len(test_loader)
    print(f"\n=> Running inference on Test set ({len(test_data)} videos, {total_batches} batches) with {device}...", flush=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 4:
                images_face, images_body, images_context, target = batch
            else:
                images_face, images_body, target = batch
                images_context = None

            images_face = images_face.to(device)
            images_body = images_body.to(device)

            if images_context is not None:
                images_context = images_context.to(device)
                output, _, _, _ = model(images_face, images_body, images_context)
            else:
                output, _, _, _ = model(images_face, images_body)

            all_logits.append(output.cpu())
            all_preds.append(output.argmax(dim=1).cpu())
            all_targets.append(target.cpu())

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                processed = min((batch_idx + 1) * model_args.batch_size, len(test_data))
                pct = (batch_idx + 1) / total_batches * 100.0
                print(f"   ⏳ [{pct:5.1f}%] Batch {batch_idx + 1:2d}/{total_batches} | Đã xử lý {processed:4d}/{len(test_data)} videos...", flush=True)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0).numpy()

    # Calculate metrics
    cm = confusion_matrix(all_targets, all_preds, labels=range(model_args.num_classes))
    row_sums = np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    cm_percent = (cm.astype(float) / row_sums) * 100.0
    diag = cm_percent.diagonal()
    uar = float(np.mean(diag))
    war = float(100.0 * np.sum(all_preds == all_targets) / len(all_targets))
    harmonic = (2.0 * war * uar) / (war + uar + 1e-6)

    print("\n" + "=" * 65)
    print(f"📊 [KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC TRÊN TEST SET (1,784 VIDEO)]")
    print("=" * 65)
    print(f"• WAR (Accuracy / Weighted Recall):  {war:.2f}%")
    print(f"• UAR (Balanced / Unweighted Recall): {uar:.2f}%")
    print(f"• Harmonic Score:                     {harmonic:.2f}%")
    print("\n• Chi tiết Recall từng lớp cảm xúc / mức độ chú ý:")
    for idx, cname in enumerate(class_names):
        count_correct = cm[idx, idx]
        total_class = cm.sum(axis=1)[idx]
        print(f"   [{idx}] {cname:<24}: {diag[idx]:6.2f}%  ({count_correct}/{total_class} videos)")

    print("\n• Confusion Matrix (Số lượng mẫu):")
    print(cm)
    print("\n• Confusion Matrix (Tỉ lệ chuẩn hoá % theo hàng):")
    print(np.round(cm_percent, 2))
    print("=" * 65)

    # Prior Calibration Analysis
    print("\n=> [Thử nghiệm hiệu chỉnh Bias phân phối - Prior Calibration]:")
    train_counts = np.array([247, 2617, 2494], dtype=float)
    train_prior = train_counts / train_counts.sum()
    probs = np.exp(all_logits) / np.sum(np.exp(all_logits), axis=1, keepdims=True)
    adj_probs = probs / (train_prior ** 0.5)
    adj_preds = adj_probs.argmax(axis=1)

    adj_cm = confusion_matrix(all_targets, adj_preds, labels=range(model_args.num_classes))
    adj_row_sums = np.maximum(adj_cm.sum(axis=1)[:, np.newaxis], 1)
    adj_cm_pct = (adj_cm.astype(float) / adj_row_sums) * 100.0
    adj_war = 100.0 * np.sum(adj_preds == all_targets) / len(all_targets)
    adj_uar = np.mean(adj_cm_pct.diagonal())
    print(f"   • WAR sau hiệu chỉnh: {adj_war:.2f}%")
    print(f"   • UAR sau hiệu chỉnh: {adj_uar:.2f}%")
    print(f"   • Recall từng lớp:    {np.round(adj_cm_pct.diagonal(), 2).tolist()}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
