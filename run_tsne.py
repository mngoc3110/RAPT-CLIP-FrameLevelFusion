"""
t-SNE visualization of RAER features using RAER-FrameLevelFusion model.
Loads the SOTA checkpoint, extracts full fused video embeddings through
Triple-Stream (Face+Body+Context) -> CMAF -> Temporal Transformer -> Projection FC,
then runs t-SNE and saves the plot.
"""
import os
import argparse
import numpy as np
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from models.clip import clip
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import test_data_loader
from utils.builders import get_class_info

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
CLASS_COLORS = ['#3b82f6', '#f59e0b', '#ef4444', '#06b6d4', '#10b981']

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
        modality_dropout=0.0,
        crop_body=True,
        freeze_image_encoder=False,
        use_moco=False
    )

    class_names, input_text = get_class_info(args)
    print("=> Loading CLIP ViT-B/16 backbone...")
    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model = clip_model.float()

    print("=> Initializing GenerateModel (Triple-Stream FrameLevelFusion)...")
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=args).to(device).float()

    print(f"=> Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("=> Initializing RAER test dataloader...")
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

    cache_feat = 'outputs/tsne_rampdown/extracted_features.npy'
    cache_lbl = 'outputs/tsne_rampdown/extracted_labels.npy'
    os.makedirs('outputs/tsne_rampdown', exist_ok=True)

    if os.path.exists(cache_feat) and os.path.exists(cache_lbl):
        print("=> Found cached embeddings! Loading from disk...")
        features = np.load(cache_feat)
        labels = np.load(cache_lbl)
    else:
        all_features = []
        all_labels = []
        print(f"=> Extracting video embeddings for {len(test_dataset)} test samples...")
        with torch.no_grad():
            for face, body, context, label in tqdm(dataloader):
                face = face.to(device).float()
                body = body.to(device).float()
                context = context.to(device).float()
                
                # Forward pass
                out = model(face, body, context)
                vid_feat = model.last_video_features # (B, 512)
                
                all_features.append(vid_feat.cpu().numpy())
                all_labels.append(label.numpy())

        features = np.concatenate(all_features, axis=0) # (528, 512)
        labels = np.concatenate(all_labels, axis=0)     # (528,)
        np.save(cache_feat, features)
        np.save(cache_lbl, labels)

    print(f"=> Feature matrix shape: {features.shape}")

    print("=> Computing t-SNE 2D projection (perplexity=30, random_state=42)...")
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(9, 7), dpi=300)
    for class_idx, (cname, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = (labels == class_idx)
        count = np.sum(mask)
        plt.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=color, label=f"{cname} (n={count})",
            alpha=0.85, edgecolors='none', s=45
        )

    plt.title("t-SNE Feature Space: RAPT-CLIP-FrameLevelFusion on RAER Test Set", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight='medium')
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight='medium')
    plt.legend(loc='best', frameon=True, fontsize=10, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()

    out_file1 = "image/tsne_finetuned.png"
    out_file2 = "outputs/tsne_rampdown/tsne_finetuned_fused.png"
    os.makedirs(os.path.dirname(out_file2), exist_ok=True)
    
    plt.savefig(out_file1, dpi=300)
    plt.savefig(out_file2, dpi=300)
    plt.close()
    print(f"=> t-SNE plot successfully saved to:\n  - {out_file1}\n  - {out_file2}")

if __name__ == '__main__':
    main()
