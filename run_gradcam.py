"""
Grad-CAM / Token-level CLIP Similarity Heatmap Visualization for RAPT-CLIP FrameLevelFusion
Checkpoint: outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth
Architecture: Triple-Stream ViT-B/16 (Face + EAA, Body, Context)
Dataset: RAER (5 Emotion Classes)
"""
import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info
from dataloader.video_dataloader import test_data_loader
import warnings

warnings.filterwarnings('ignore')


def compute_stream_token_similarity(model, img_4d, target_class, all_class_text):
    """
    Compute spatial token-level CLIP similarity for a single image tensor (1, C, H, W).
    Projects spatial patches (14x14 = 196 tokens) to the vision-language latent space,
    computes cosine similarity with class text embeddings, and performs
    class-discriminative suppression (target_sim - mean_sim).
    """
    c, h, w = img_4d.shape[1], img_4d.shape[2], img_4d.shape[3]
    
    transformer_output = {}
    def hook_fn(module, input, output):
        transformer_output['out'] = output.detach()

    handle = model.image_encoder.transformer.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model.image_encoder(img_4d)
    handle.remove()

    if 'out' not in transformer_output:
        return np.zeros((14, 14), dtype=np.float32)

    with torch.no_grad():
        # transformer output: (197, B, 768) -> permute to (B, 197, 768)
        feat = transformer_output['out'].permute(1, 0, 2)
        spatial = feat[0, 1:]  # (196, 768) - skip CLS token

        # Project spatial patches through ln_post + proj
        spatial_n = model.image_encoder.ln_post(spatial)
        if model.image_encoder.proj is not None:
            spatial_p = spatial_n @ model.image_encoder.proj
        else:
            spatial_p = spatial_n
        spatial_p = spatial_p / (spatial_p.norm(dim=-1, keepdim=True) + 1e-6)

        # Cosine similarity with all classes: (196, num_classes)
        all_sim = spatial_p @ all_class_text.t()

        # Class-discriminative: target class similarity minus average similarity across all classes
        target_sim = all_sim[:, target_class]
        mean_sim = all_sim.mean(dim=1)
        disc_sim = (target_sim - mean_sim).cpu().numpy().reshape(14, 14)

        # Apply ReLU / positive only
        cam = np.maximum(disc_sim, 0)
        if cam.max() > 1e-7:
            cam = cam / cam.max()

        # Percentile thresholding (suppress lower 40% noise to highlight focal points)
        thresh = np.percentile(cam, 40)
        cam = np.where(cam >= thresh, cam, 0.0)
        if cam.max() > 1e-7:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(img_bgr, cam, alpha=0.55):
    """Intensity-weighted alpha blending with smooth colormap overlay."""
    h, w = img_bgr.shape[:2]
    cam_smooth = cv2.GaussianBlur(cam.astype(np.float32), (7, 7), sigmaX=2.0)
    cam_resized = cv2.resize(cam_smooth, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    blend_w = np.clip(cam_resized * 1.4, 0, 1) * alpha
    blend_3ch = np.stack([blend_w] * 3, axis=-1)

    result = img_bgr.astype(np.float32) * (1 - blend_3ch) + heatmap.astype(np.float32) * blend_3ch
    return np.uint8(np.clip(result, 0, 255))


def tensor_to_bgr(tensor):
    """Convert normalized (C, H, W) tensor to uint8 BGR image."""
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
    t = tensor.detach().cpu().numpy() * std + mean
    t = np.clip(t, 0, 1)
    img_hwc = np.transpose(t, (1, 2, 0))
    return cv2.cvtColor((img_hwc * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def create_tri_stream_canvas(img_f, img_b, img_c, cam_f, cam_b, cam_c, true_cls, pred_cls, conf):
    """Creates a side-by-side 2x3 panel comparing Face, Body, and Context streams."""
    s = 224
    pad_top = 50
    gap = 8
    pad_bottom = 45
    grid_w = s * 3 + gap * 4
    grid_h = s * 2 + pad_top + gap * 2 + pad_bottom

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 248  # clean light background

    font = cv2.FONT_HERSHEY_DUPLEX

    # Column titles
    col_titles = ["Face Stream (Micro-expressions)", "Body Stream (Posture/Gestures)", "Context Stream (Environment)"]
    for col, title in enumerate(col_titles):
        x = gap + col * (s + gap)
        tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.putText(canvas, title, (x + (s - tw) // 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1)

    # Row 1: Original Inputs
    y_row1 = pad_top
    for col, img in enumerate([img_f, img_b, img_c]):
        x = gap + col * (s + gap)
        canvas[y_row1:y_row1 + s, x:x + s] = cv2.resize(img, (s, s))

    # Row 2: Heatmap Overlays
    y_row2 = pad_top + s + gap
    for col, cam_img in enumerate([cam_f, cam_b, cam_c]):
        x = gap + col * (s + gap)
        canvas[y_row2:y_row2 + s, x:x + s] = cv2.resize(cam_img, (s, s))

    # Bottom Banner
    match = (true_cls == pred_cls)
    status_color = (30, 140, 30) if match else (30, 30, 200)
    status_text = f"Ground Truth: {true_cls}  |  Predicted: {pred_cls} ({conf:.1f}%)"
    cv2.putText(canvas, status_text, (gap + 10, grid_h - 18), font, 0.52, status_color, 1)

    return canvas


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
    print(f"=> Target Classes: {class_names}")

    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model = clip_model.float()
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=args).to(device).float()

    print(f"=> Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Precompute ensemble text features for all classes
    with torch.no_grad():
        prompts = model.prompt_learner()
        text_feats = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
        if model.is_ensemble:
            text_feats = text_feats.view(model.num_classes, model.num_prompts_per_class, -1)
            all_class_text = text_feats.mean(dim=1)
        else:
            all_class_text = text_feats
        all_class_text = all_class_text / (all_class_text.norm(dim=-1, keepdim=True) + 1e-6)

    print("=> Initializing test dataset...")
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

    out_dir = 'outputs/gradcam'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('image', exist_ok=True)

    # Pick 1 representative sample for each class
    found_classes = {}
    print("=> Scanning test set for representative samples of each class...")
    for idx in range(len(test_dataset)):
        face, body, context, label = test_dataset[idx]
        cls_idx = int(label)
        if cls_idx not in found_classes:
            found_classes[cls_idx] = idx
        if len(found_classes) == len(class_names):
            break

    print(f"=> Selected {len(found_classes)} samples across classes: {list(found_classes.keys())}")

    for cls_idx, sample_idx in found_classes.items():
        face, body, context, label = test_dataset[sample_idx]
        face_t = face.unsqueeze(0).to(device).float()
        body_t = body.unsqueeze(0).to(device).float()
        context_t = context.unsqueeze(0).to(device).float()

        # Run forward pass to get prediction
        with torch.no_grad():
            logits, _, _, _, _ = model(face_t, body_t, context_t)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = logits.argmax(dim=-1).item()
            conf = probs[pred_idx].item() * 100.0

        true_cls_name = class_names[cls_idx]
        pred_cls_name = class_names[pred_idx]

        # Use middle frame (t=8) for visualization
        mid = face.shape[0] // 2
        f_mid_tensor = face[mid].unsqueeze(0).to(device).float()
        b_mid_tensor = body[mid].unsqueeze(0).to(device).float()
        c_mid_tensor = context[mid].unsqueeze(0).to(device).float()

        # Compute heatmaps for all 3 streams
        cam_face = compute_stream_token_similarity(model, f_mid_tensor, pred_idx, all_class_text)
        cam_body = compute_stream_token_similarity(model, b_mid_tensor, pred_idx, all_class_text)
        cam_context = compute_stream_token_similarity(model, c_mid_tensor, pred_idx, all_class_text)

        # Convert tensors to BGR images
        img_f = tensor_to_bgr(face[mid])
        img_b = tensor_to_bgr(body[mid])
        img_c = tensor_to_bgr(context[mid])

        # Overlays
        ov_f = overlay_heatmap(img_f, cam_face)
        ov_b = overlay_heatmap(img_b, cam_body)
        ov_c = overlay_heatmap(img_c, cam_context)

        canvas = create_tri_stream_canvas(img_f, img_b, img_c, ov_f, ov_b, ov_c, true_cls_name, pred_cls_name, conf)

        save_path = os.path.join(out_dir, f"gradcam_sample_class_{cls_idx}_{true_cls_name}.png")
        cv2.imwrite(save_path, canvas)
        print(f"  [Saved] {save_path} (True: {true_cls_name}, Pred: {pred_cls_name})")

        # Save the first sample as the main presentation image
        if cls_idx == 1 or cls_idx == list(found_classes.keys())[0]:
            cv2.imwrite('image/gradcam_sample.png', canvas)

    print("\n=> Grad-CAM visualization complete! Visuals saved to outputs/gradcam/ and image/gradcam_sample.png.")


if __name__ == '__main__':
    main()
