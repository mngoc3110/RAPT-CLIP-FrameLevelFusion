"""
Robustness evaluation script for RAER-FrameLevelFusion model.
Evaluates model performance under:
1. Modality Occlusion (Masked Face, Masked Body, Masked Context)
2. Gaussian Noise Injection (std: 0.0, 0.1, 0.3, 0.5)
3. Structural Disturbances (Temporal Shuffling, Random Erasing)

Saves results to:
- outputs/robustness/modality_ablation.txt
- outputs/robustness/advanced_robustness_results.txt
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

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

def evaluate(model, dataloader, device, condition="normal", noise_std=0.0):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for face, body, context, label in dataloader:
            face = face.to(device).float()
            body = body.to(device).float()
            context = context.to(device).float()
            
            # 1. Condition: Modality Masking
            if condition == "mask_face":
                face = torch.zeros_like(face)
            elif condition == "mask_body":
                body = torch.zeros_like(body)
            elif condition == "mask_context":
                context = torch.zeros_like(context)
                
            # 2. Condition: Gaussian Noise
            if noise_std > 0.0:
                face = face + torch.randn_like(face) * noise_std
                body = body + torch.randn_like(body) * noise_std
                context = context + torch.randn_like(context) * noise_std
                
            # 3. Condition: Temporal Shuffling
            if condition == "temporal_shuffle":
                # Permute temporal dimension (dim=1) randomly
                perm = torch.randperm(face.size(1))
                face = face[:, perm]
                body = body[:, perm]
                context = context[:, perm]
                
            # 4. Condition: Random Erasing on Body (occlude a random patch of frames)
            if condition == "random_erasing_body":
                b, t, c, h, w = body.shape
                # Erase a 100x100 patch in the center
                body[:, :, :, h//4:3*h//4, w//4:3*w//4] = 0.0

            out = model(face, body, context)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(label.numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    war = accuracy_score(y_true, y_pred) * 100.0
    uar = recall_score(y_true, y_pred, average='macro') * 100.0
    return war, uar

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

    print("\n" + "="*50)
    print("1. EVALUATING MODALITY ABLATION / OCCLUSION")
    print("="*50)
    
    war_normal, uar_normal = evaluate(model, dataloader, device, condition="normal")
    print(f"Normal (Face + Body + Context)  : WAR = {war_normal:.2f}%, UAR = {uar_normal:.2f}%")
    
    war_mask_ctx, uar_mask_ctx = evaluate(model, dataloader, device, condition="mask_context")
    print(f"Masked Context (Zeros)          : WAR = {war_mask_ctx:.2f}%, UAR = {uar_mask_ctx:.2f}%")
    
    war_mask_face, uar_mask_face = evaluate(model, dataloader, device, condition="mask_face")
    print(f"Masked Face (Zeros)             : WAR = {war_mask_face:.2f}%, UAR = {uar_mask_face:.2f}%")
    
    war_mask_body, uar_mask_body = evaluate(model, dataloader, device, condition="mask_body")
    print(f"Masked Body (Zeros)             : WAR = {war_mask_body:.2f}%, UAR = {uar_mask_body:.2f}%")

    with open('outputs/robustness/modality_ablation.txt', 'w') as f:
        f.write(f"Normal (Face + Body + Context)  : {war_normal:.2f}% (UAR: {uar_normal:.2f}%)\n")
        f.write(f"Masked Context (Zeros)          : {war_mask_ctx:.2f}% (UAR: {uar_mask_ctx:.2f}%)\n")
        f.write(f"Masked Face (Zeros)             : {war_mask_face:.2f}% (UAR: {uar_mask_face:.2f}%)\n")
        f.write(f"Masked Body (Zeros)             : {war_mask_body:.2f}% (UAR: {uar_mask_body:.2f}%)\n")

    print("\n" + "="*50)
    print("2. EVALUATING GAUSSIAN NOISE INJECTION")
    print("="*50)
    noise_results = {}
    for std in [0.0, 0.1, 0.3, 0.5]:
        w, u = evaluate(model, dataloader, device, noise_std=std)
        noise_results[std] = (w, u)
        print(f"Noise Std {std:.1f}: WAR = {w:.2f}%, UAR = {u:.2f}%")

    print("\n" + "="*50)
    print("3. EVALUATING STRUCTURAL DISTURBANCE")
    print("="*50)
    w_shuf, u_shuf = evaluate(model, dataloader, device, condition="temporal_shuffle")
    print(f"Temporal Shuffling: WAR = {w_shuf:.2f}%, UAR = {u_shuf:.2f}%")
    
    w_erase, u_erase = evaluate(model, dataloader, device, condition="random_erasing_body")
    print(f"Random Erasing (Body): WAR = {w_erase:.2f}%, UAR = {u_erase:.2f}%")

    with open('outputs/robustness/advanced_robustness_results.txt', 'w') as f:
        f.write("Noise Injection Results:\n")
        for std, (w, u) in noise_results.items():
            f.write(f"Std {std:.1f}: {w:.2f}% (UAR: {u:.2f}%)\n")
        f.write("\nStructural Disturbances:\n")
        f.write(f"Normal: {war_normal:.2f}% (UAR: {uar_normal:.2f}%)\n")
        f.write(f"Temporal Shuffling: {w_shuf:.2f}% (UAR: {u_shuf:.2f}%)\n")
        f.write(f"Random Erasing (Body): {w_erase:.2f}% (UAR: {u_erase:.2f}%)\n")

    print("\n=> All robustness results successfully saved to outputs/robustness/!")

if __name__ == '__main__':
    main()
