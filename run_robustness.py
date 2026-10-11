"""
Robustness evaluation script for RAER-FrameLevelFusion model.
Evaluates model performance under:
1. Modality Occlusion (Masked Face, Masked Body, Masked Context)
2. Gaussian Noise Injection (std: 0.0, 0.1, 0.3, 0.5)
3. Structural Disturbances (Temporal Shuffling, Random Erasing)

Optimized for Single-Pass DataLoader iteration (9x faster).
"""
import os
import argparse
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import recall_score, accuracy_score

from models.clip import clip
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import test_data_loader
from utils.builders import get_class_info
import warnings

warnings.filterwarnings('ignore')

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']


def evaluate_all(model, dataloader, device):
    conditions = [
        "normal",
        "mask_context",
        "mask_face",
        "mask_body",
        "noise_0.1",
        "noise_0.3",
        "noise_0.5",
        "temporal_shuffle",
        "random_erasing_body"
    ]
    preds = {c: [] for c in conditions}
    targets = []

    model.eval()
    with torch.no_grad():
        for face, body, context, label in tqdm(dataloader, desc="Evaluating All Robustness Conditions"):
            face = face.to(device).float()
            body = body.to(device).float()
            context = context.to(device).float()
            targets.extend(label.numpy())

            # 1. Normal
            out = model(face, body, context)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["normal"].extend(l)

            # 2. Mask Context
            out = model(face, body, torch.zeros_like(context))
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["mask_context"].extend(l)

            # 3. Mask Face
            out = model(torch.zeros_like(face), body, context)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["mask_face"].extend(l)

            # 4. Mask Body
            out = model(face, torch.zeros_like(body), context)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["mask_body"].extend(l)

            # 5. Gaussian Noise 0.1
            out = model(face + torch.randn_like(face)*0.1, body + torch.randn_like(body)*0.1, context + torch.randn_like(context)*0.1)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["noise_0.1"].extend(l)

            # 6. Gaussian Noise 0.3
            out = model(face + torch.randn_like(face)*0.3, body + torch.randn_like(body)*0.3, context + torch.randn_like(context)*0.3)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["noise_0.3"].extend(l)

            # 7. Gaussian Noise 0.5
            out = model(face + torch.randn_like(face)*0.5, body + torch.randn_like(body)*0.5, context + torch.randn_like(context)*0.5)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["noise_0.5"].extend(l)

            # 8. Temporal Shuffling
            perm = torch.randperm(face.size(1))
            out = model(face[:, perm], body[:, perm], context[:, perm])
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["temporal_shuffle"].extend(l)

            # 9. Random Erasing on Body
            b_erased = body.clone()
            h, w = b_erased.shape[-2], b_erased.shape[-1]
            b_erased[:, :, :, h//4:3*h//4, w//4:3*w//4] = 0.0
            out = model(face, b_erased, context)
            l = (out[0] if isinstance(out, (tuple, list)) else out).argmax(dim=-1).cpu().numpy()
            preds["random_erasing_body"].extend(l)

    y_true = np.array(targets)
    metrics = {}
    for c in conditions:
        y_pred = np.array(preds[c])
        war = accuracy_score(y_true, y_pred) * 100.0
        uar = recall_score(y_true, y_pred, average='macro') * 100.0
        metrics[c] = (war, uar)

    return metrics


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"=> Using device: {device}")

    ckpt_path = 'outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    args = argparse.Namespace(
        clip_path='ViT-B/16',
        drop_path_rate=0.0,
        dataset='RAER',
        use_context=True,
        fusion_type='cmaf',
        fusion_stage='frame',
        temporal_layers=1,
        contexts_number=8,
        class_token_position='end',
        class_specific_contexts=True,
        load_and_tune_prompt_learner=True,
        text_type='prompt_ensemble',
        temperature=0.07,
        lr=2e-5,
        lr_image_encoder=1e-6,
        lr_prompt_learner=3e-4,
        lr_adapter=1e-4,
        lambda_vad=0.0,
        modality_dropout=0.0,
        crop_body=True,
        freeze_image_encoder=False,
        use_moco=False
    )

    class_names, input_text = get_class_info(args)
    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model = clip_model.float()
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=args).to(device).float()

    print(f"=> Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_dataset = test_data_loader(
        root_dir='./',
        list_file='RAER/annotation/test.txt',
        num_segments=16,
        duration=1,
        image_size=224,
        bounding_box_face='RAER/bounding_box/face.json',
        bounding_box_body='RAER/bounding_box/body.json',
        crop_body=True,
        num_classes=5,
        dataset_name='RAER'
    )
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=False
    )

    os.makedirs('outputs/robustness', exist_ok=True)

    metrics = evaluate_all(model, dataloader, device)

    print("\n" + "="*50)
    print("1. MODALITY ABLATION / OCCLUSION RESULTS")
    print("="*50)
    for c in ["normal", "mask_context", "mask_face", "mask_body"]:
        w, u = metrics[c]
        print(f"  {c:<22}: WAR = {w:.2f}%, UAR = {u:.2f}%")

    with open('outputs/robustness/modality_ablation.txt', 'w') as f:
        f.write(f"Normal (Face + Body + Context)  : {metrics['normal'][0]:.2f}% (UAR: {metrics['normal'][1]:.2f}%)\n")
        f.write(f"Masked Context (Zeros)          : {metrics['mask_context'][0]:.2f}% (UAR: {metrics['mask_context'][1]:.2f}%)\n")
        f.write(f"Masked Face (Zeros)             : {metrics['mask_face'][0]:.2f}% (UAR: {metrics['mask_face'][1]:.2f}%)\n")
        f.write(f"Masked Body (Zeros)             : {metrics['mask_body'][0]:.2f}% (UAR: {metrics['mask_body'][1]:.2f}%)\n")

    print("\n" + "="*50)
    print("2. GAUSSIAN NOISE INJECTION RESULTS")
    print("="*50)
    for std_str in ["noise_0.1", "noise_0.3", "noise_0.5"]:
        w, u = metrics[std_str]
        print(f"  {std_str:<22}: WAR = {w:.2f}%, UAR = {u:.2f}%")

    print("\n" + "="*50)
    print("3. STRUCTURAL DISTURBANCES RESULTS")
    print("="*50)
    for s_str in ["temporal_shuffle", "random_erasing_body"]:
        w, u = metrics[s_str]
        print(f"  {s_str:<22}: WAR = {w:.2f}%, UAR = {u:.2f}%")

    with open('outputs/robustness/advanced_robustness_results.txt', 'w') as f:
        f.write("Noise Injection Results:\n")
        f.write(f"Std 0.0 (Normal): {metrics['normal'][0]:.2f}% (UAR: {metrics['normal'][1]:.2f}%)\n")
        for std_str in ["noise_0.1", "noise_0.3", "noise_0.5"]:
            f.write(f"{std_str}: {metrics[std_str][0]:.2f}% (UAR: {metrics[std_str][1]:.2f}%)\n")
        f.write("\nStructural Disturbances:\n")
        f.write(f"Normal: {metrics['normal'][0]:.2f}% (UAR: {metrics['normal'][1]:.2f}%)\n")
        f.write(f"Temporal Shuffling: {metrics['temporal_shuffle'][0]:.2f}% (UAR: {metrics['temporal_shuffle'][1]:.2f}%)\n")
        f.write(f"Random Erasing (Body): {metrics['random_erasing_body'][0]:.2f}% (UAR: {metrics['random_erasing_body'][1]:.2f}%)\n")

    print("\n=> All robustness results successfully saved to outputs/robustness/!")


if __name__ == '__main__':
    main()
