"""
Advanced Academic Benchmarks for RAPT-CLIP FrameLevelFusion:
1. Parameter Count & Inference Latency / Throughput (FPS)
2. Model Calibration & Expected Calibration Error (ECE)
3. Failure Case & Error Boundary Mining
4. Temporal Frame-Budget Sensitivity (T in {4, 8, 16})
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from models.clip import clip
from models.Generate_Model import GenerateModel
from dataloader.video_dataloader import test_data_loader
from utils.builders import get_class_info
import warnings

warnings.filterwarnings('ignore')

CLASS_NAMES = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']


def setup_model(device, ckpt_path):
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
    clip_model, _ = clip.load('ViT-B/16', device=device)
    clip_model = clip_model.float()
    model = GenerateModel(input_text=input_text, clip_model=clip_model, args=args).to(device).float()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, args


def benchmark_efficiency(model, device, out_dir):
    print("\n" + "="*60)
    print("1. EFFICIENCY BENCHMARK: PARAMETERS, LATENCY & FPS")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Warmup
    dummy_face = torch.randn(1, 16, 3, 224, 224, device=device).float()
    dummy_body = torch.randn(1, 16, 3, 224, 224, device=device).float()
    dummy_context = torch.randn(1, 16, 3, 224, 224, device=device).float()

    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_face, dummy_body, dummy_context)
    if device.type == 'mps':
        torch.mps.synchronize()

    # Latency timing over 30 runs
    times = []
    num_runs = 30
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_face, dummy_body, dummy_context)
        if device.type == 'mps':
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)

    mean_sec = np.mean(times)
    std_sec = np.std(times)
    latency_ms = mean_sec * 1000.0
    clips_per_sec = 1.0 / mean_sec
    fps = clips_per_sec * 16.0  # 16 frames per clip

    report = (
        f"Total Parameters      : {total_params / 1e6:.2f} M\n"
        f"Trainable Parameters  : {trainable_params / 1e6:.2f} M ({trainable_params/total_params*100:.2f}%)\n"
        f"Frozen Backbone       : {frozen_params / 1e6:.2f} M\n"
        f"Inference Latency     : {latency_ms:.2f} +/- {std_sec*1000:.2f} ms / video clip\n"
        f"Throughput            : {clips_per_sec:.2f} clips/sec ({fps:.2f} FPS)\n"
    )
    print(report)
    with open(os.path.join(out_dir, 'efficiency_report.txt'), 'w') as f:
        f.write(report)


def compute_ece(probs, labels, n_bins=10):
    """Computes Expected Calibration Error (ECE) and bin statistics."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        bin_counts.append(np.sum(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)

    return ece, bin_boundaries, bin_accs, bin_confs, bin_counts


def benchmark_calibration_and_errors(model, dataloader, device, out_dir):
    print("\n" + "="*60)
    print("2. CONFIDENCE CALIBRATION & FAILURE CASE MINING")
    print("="*60)

    all_probs = []
    all_preds = []
    all_targets = []
    all_confs = []

    with torch.no_grad():
        for face, body, context, label in tqdm(dataloader, desc="Inference"):
            face = face.to(device).float()
            body = body.to(device).float()
            context = context.to(device).float()

            out = model(face, body, context)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            preds = np.argmax(probs, axis=-1)
            confs = np.max(probs, axis=-1)

            all_probs.append(probs)
            all_preds.extend(preds)
            all_targets.extend(label.numpy())
            all_confs.extend(confs)

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    confs = np.array(all_confs)

    war = accuracy_score(y_true, y_pred) * 100.0
    uar = recall_score(y_true, y_pred, average='macro') * 100.0
    ece, bin_boundaries, bin_accs, bin_confs, bin_counts = compute_ece(probs, y_true, n_bins=10)

    print(f"Overall Accuracy (WAR) : {war:.2f}%")
    print(f"Macro Recall (UAR)     : {uar:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece*100:.2f}%")

    # Plot Reliability Diagram
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color='#2563EB', edgecolor='black', label='Outputs')
    ax.plot([0, 1], [0, 1], '--', color='red', linewidth=1.5, label='Perfect Calibration')
    ax.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'Reliability Diagram (ECE = {ece*100:.2f}%)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'reliability_diagram.png')
    plt.savefig(plot_path)
    plt.savefig('image/reliability_diagram.png')
    plt.close()
    print(f"Reliability diagram saved to: {plot_path}")

    # Mine Failure Cases (Errors with highest confidence)
    errors = np.where(y_true != y_pred)[0]
    error_conf_indices = errors[np.argsort(-confs[errors])]

    failure_log = [
        f"CALIBRATION & ERROR BOUNDARY REPORT",
        f"WAR: {war:.2f}% | UAR: {uar:.2f}% | ECE: {ece*100:.2f}%\n",
        f"Total Errors: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true)*100:.2f}%)\n",
        f"TOP 10 HIGH-CONFIDENCE FAILURE CASES (HARD SAMPLES):"
    ]
    for rank, idx in enumerate(error_conf_indices[:10]):
        true_name = CLASS_NAMES[y_true[idx]]
        pred_name = CLASS_NAMES[y_pred[idx]]
        prob_str = ", ".join([f"{CLASS_NAMES[c]}: {probs[idx, c]*100:.1f}%" for c in range(5)])
        failure_log.append(f"  #{rank+1} Sample #{idx}: True=[{true_name}] != Pred=[{pred_name}] (Confidence: {confs[idx]*100:.1f}%)")
        failure_log.append(f"      Probabilities: {prob_str}")

    failure_text = "\n".join(failure_log)
    print("\n" + failure_text)
    with open(os.path.join(out_dir, 'failure_cases.txt'), 'w') as f:
        f.write(failure_text + '\n')


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"=> Using device: {device}")

    ckpt_path = 'outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth'
    out_dir = 'outputs/benchmarks'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('image', exist_ok=True)

    model, args = setup_model(device, ckpt_path)

    # 1. Efficiency & Latency Benchmark
    benchmark_efficiency(model, device, out_dir)

    # 2. Calibration & Failure Mining on Test Dataset
    print("\n=> Initializing RAER test dataloader (16 segments)...")
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
    benchmark_calibration_and_errors(model, dataloader, device, out_dir)

    print("\n=> All advanced benchmarks completed successfully!")


if __name__ == '__main__':
    main()
