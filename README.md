<p align="center">
  <img src="image/model.png" width="100%" alt="RAPT-CLIP Architecture" />
</p>

<h1 align="center">RAPT-CLIP-FrameLevelFusion</h1>

<p align="center">
  <b>Recognition of Academic Emotions through Prompt-Tuned CLIP with Frame-Level Fusion</b><br/>
  A triple-stream vision-language framework with Tridirectional Cross-Modal Attention and Temporal Attention Pooling for classroom video emotion recognition.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/Backbone-CLIP_ViT--B%2F16-4285F4?style=flat-square" />
  <img src="https://img.shields.io/badge/Best_Valid_UAR-77.06%25_(RAER)-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Valid_WAR-78.79%25_(RAER)-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Architecture-Triple--Stream_CMAF-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/License-Academic-blue?style=flat-square" />
</p>

---

## 📌 Overview

**RAPT-CLIP-FrameLevelFusion** is a state-of-the-art multimodal video emotion recognition framework tailored for educational environments. The system recognizes five key academic emotions from student classroom videos in the **RAER dataset**:
1. **Neutrality** (Bình thản / Thụ động)
2. **Enjoyment** (Hào hứng / Hứng thú)
3. **Confusion** (Bối rối / Khó hiểu)
4. **Fatigue** (Mệt mỏi / Buồn ngủ)
5. **Distraction** (Mất tập trung / Xao nhãng)

### 🌟 Key Innovations & Contributions

- **Triple-Stream Vision Encoder**: Concurrently processes **Face Crop** (with an Expression-Aware Bottleneck Adapter — EAA), **Body Crop** (postural cues), and **Context Crop** (classroom spatial setting) extracted via CLIP ViT-B/16.
- **Frame-Level Tridirectional CMAF**: Computes mutual cross-attention ($\text{Face} \leftrightarrow \text{Body} \leftrightarrow \text{Context}$) on **every individual frame** ($f_t \in \mathbb{R}^{1536}$) before temporal aggregation, preserving instantaneous micro-expression and behavioral synchrony.
- **Unified Temporal Transformer with Attention Pooling**: Dynamically scores frame significance ($\alpha_t$) across 16 temporal segments to spotlight peak emotion frames without smoothing away fleeting micro-expressions.
- **Dual-View Prompt Learning (CoOp + Ensemble)**: Combines hand-crafted descriptive anchors with 8 learnable class-specific context tokens.
- **Regularized Margin Optimization**: Optimized via **LDAM Loss** ($s=30, m_{\max}=0.5$) for severe long-tailed imbalance, combined with **Mutual Information (MI)** and **Decorrelation (DC)** constraints to prevent prompt drift and feature collapse.
- **Configurable Fusion Stages**: Supports `--fusion-stage frame` (Frame-Level CMAF $\to$ Temporal Transformer, SOTA) and `--fusion-stage video` (Stream-Level Temporal $\to$ Video-Level CMAF) for controlled architectural ablation.

---

## 🏛 Architecture Summary

```
[Video Segment: T=16]
        │
  ┌─────┴───────────────────────┬────────────────────────┐
  ▼                             ▼                        ▼
Face Stream (224x224)         Body Stream (224x224)    Context Stream (224x224)
  │                             │                        │
CLIP ViT-B/16 (lr=1e-6)       CLIP ViT-B/16 (lr=1e-6)  CLIP ViT-B/16 (lr=1e-6)
  │                             │                        │
Face Adapter EAA (512→128→512)  │                        │
  │                             │                        │
  └─────────────────────────────┼────────────────────────┘
                                ▼
         Tridirectional CMAF at Frame-Level (f_t ∈ R^1536)
                                │
                                ▼
         Unified Temporal Transformer (1 layer, 8 heads)
                                │
                                ▼
             Attention Pooling (Dynamic peak weights α_t)
                                │
                                ▼
               Project FC (1536 → 512, L2-normalized)
                                │
            Cosine Similarity (τ = 0.07) with Text Stream
                                │
            LDAM Loss (s=30, m=0.5) + MI Loss + DC Loss
```

---

## 📂 Project Structure

```
RAPT-CLIP-FrameLevelFusion/
├── main.py                          # Core entry point: training, validation, CLI flags
├── trainer.py                       # Training loop, LDAM/MI/DC losses, evaluation metrics
├── models/
│   ├── Generate_Model.py            # Primary model: Triple-Stream, CMAF, Temporal Transformer
│   ├── CrossModalAttentionFusion.py # Bidirectional & Tridirectional CMAF module
│   ├── Temporal_Model.py            # Temporal Transformer + Dynamic Attention Pooling
│   ├── Prompt_Learner.py            # CoOp-style learnable prompt module
│   ├── Text.py                      # Hand-crafted emotion prompt descriptors & anchors
│   ├── Adapter.py                   # Expression-Aware Adapter (EAA) bottleneck module
│   └── clip/                        # CLIP ViT-B/16 backbone
├── dataloader/
│   ├── video_dataloader.py          # Video dataset loader with Face/Body/Context cropping
│   └── video_transform.py           # Group augmentations (ColorJitter, Rotation, Flipping)
├── utils/
│   ├── builders.py                  # Model instantiator & optimizer parameter grouping
│   ├── loss.py                      # LDAM Loss, MI Loss, DC Loss, LDL Loss implementations
│   └── utils.py                     # Metric calculations (WAR/UAR), logging, confusion matrix
├── train_sh/
│   ├── raer_framelevelfusion.sh      # ⭐ SOTA Training script (Triple-Stream Frame-Level Fusion)
│   ├── raer_cmaf.sh                 # Dual-stream CMAF baseline script
│   └── ablation/
│       ├── raer_video_level_fusion.sh # 🔬 Architectural ablation (Temporal before CMAF)
│       ├── raer_full.sh             # V1 Dual-stream baseline (73.76% UAR)
│       ├── raer_no_attn-pooling.sh  # Ablation: CLS Token instead of Attention Pooling
│       ├── raer_no_prompt_tuning.sh # Ablation: Class descriptor prompts only
│       ├── raer_no_ldam.sh          # Ablation: Cross-Entropy instead of LDAM Loss
│       ├── raer_no_sampler.sh       # Ablation: Uniform sampler instead of Weighted sampler
│       ├── raer_freeze_encoder.sh   # Ablation: Frozen CLIP visual encoder
│       └── raer_drop_path.sh        # Ablation: DropPath stochastic depth
├── valid_sh/
│   └── eval_local_mac.sh            # Local validation script
├── RAER/
│   ├── annotation/                  # train.txt, test.txt (video_path num_frames label)
│   └── bounding_box/                # face.json, body.json
├── outputs/                         # Experimental results & checkpoints
│   ├── RAER-FrameLevelFusion-[06-07]-[03:06]/ # Best model (UAR 77.06%, model_best.pth)
│   ├── confusion_matrix/            # Confusion matrices for RAER
│   ├── robustness/                  # Noise, structural disturbance & modality ablation
│   └── tsne_rampdown/               # Pretrained vs Fine-tuned t-SNE cluster plots
├── run_realtime.py                  # Live webcam emotion inference with face detection
├── run_thesis_gradcam_v2.py         # GradCAM token similarity heatmap generator
├── run_tsne_rampdown.py             # t-SNE feature space visualizer
├── evaluate_advanced_robustness.py  # Robustness evaluator (noise, occlusion, shuffling)
├── eval_tta.py                      # Test-Time Augmentation (TTA) evaluation
└── RAPT_CLIP_RAER_Hyperparameters_Report.pdf # 📄 Detailed Technical PDF Report
```

---

## ⚙️ Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/mngoc3110/RAPT-CLIP-FrameLevelFusion.git
cd RAPT-CLIP-FrameLevelFusion

# Install PyTorch (adjust according to your CUDA version or Apple Silicon MPS)
pip install torch torchvision

# Install dependencies
pip install ftfy regex tqdm scikit-learn matplotlib seaborn opencv-python pypdf weasyprint

# (Optional) For webcam live inference with MTCNN:
pip install facenet-pytorch
```

---

## 🚀 Training

### 1. Train Proposed SOTA Model (Frame-Level Fusion)

Run the full triple-stream model with Tridirectional CMAF and Attention Pooling:

```bash
bash train_sh/raer_framelevelfusion.sh
```

Or execute directly via Python:

```bash
python main.py \
  --mode train \
  --exper-name RAER-FrameLevelFusion \
  --dataset RAER \
  --gpu 0 \
  --epochs 20 \
  --batch-size 2 \
  --fusion-type cmaf \
  --fusion-stage frame \
  --use-context \
  --crop-body \
  --optimizer AdamW \
  --lr 2e-05 \
  --lr-image-encoder 1e-06 \
  --lr-prompt-learner 0.0003 \
  --lr-adapter 0.0001 \
  --weight-decay 0.005 \
  --milestones 10 15 \
  --gamma 0.1 \
  --loss-type ldam \
  --ldam-s 30.0 \
  --ldam-max-m 0.5 \
  --lambda_mi 0.1 \
  --lambda_dc 0.1 \
  --use-weighted-sampler \
  --drop-path-rate 0.1 \
  --modality-dropout 0.3 \
  --text-type prompt_ensemble \
  --use-amp
```

### 2. Train Architectural Ablation: Video-Level Fusion (Temporal before CMAF)

To evaluate whether temporal aggregation should happen *before* or *after* cross-modal fusion:

```bash
bash train_sh/ablation/raer_video_level_fusion.sh
```

---

## 📊 Experimental Results & Benchmarks

### 1. Visual Modality & Robustness Analysis (Tier 1)

Evaluates the contribution of individual visual streams on the RAER dataset:

| Model Configuration | Fusion Mechanism | Valid WAR (%) | Valid UAR (%) | $\Delta$ UAR |
| :--- | :---: | :---: | :---: | :---: |
| **★ Triple-Stream: Face + Body + Context (Proposed SOTA)** | **Tridirectional CMAF (1536-d)** | **78.79%** | **77.06%** | **+3.30%** |
| Dual-Stream: Face + Body (Baseline v1) | Bidirectional CMAF (1024-d) | 79.92% | 73.76% | (Baseline) |
| Masked Context (Loss of Classroom Context) | 2-Stream Fallback | 79.92% | 73.76% | -3.30% |
| Masked Face (Face Occlusion / Zeros) | Body + Context Only | 66.29% | $\approx 62.10\%$ | -14.96% |
| Masked Body (Body Occlusion / Zeros) | Face + Context Only | 38.45% | $\approx 35.20\%$ | -41.86% |

> **Key Finding**: Body posture accounts for the steepest accuracy drop when occluded (-41.86%), while the classroom context provides a +3.30% UAR boost over the dual-stream model.

---

### 2. Component Ablation Study (Tier 2)

Evaluated under controlled conditions on the Dual-Stream Baseline (73.76% UAR) to isolate individual module contributions:

| Configuration | Valid WAR (%) | Valid UAR (%) | $\Delta$ UAR | Technical Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Full Baseline (Dual-Stream v1)** | **79.92%** | **73.76%** | — | Standard complete configuration |
| w/o Attention Pooling *(w/ CLS Token)* | 77.84% | 67.45% | **-6.31%** | Fails to detect fleeting peak emotion frames |
| w/o Prompt Ensemble *(Class Descriptors only)* | 80.49% | 68.62% | **-5.14%** | Static prompts fail to adapt to classroom domain |
| w/o MI & DC Regularization *($\lambda = 0$)* | 77.46% | 69.30% | **-4.46%** | Learnable prompts suffer from semantic collapse |
| w/o LDAM Loss *(w/ standard Cross-Entropy)* | 73.11% | 69.97% | **-3.79%** | Severe bias towards majority class (Neutral) |
| w/o DropPath *(Stochastic Depth = 0.0)* | 73.30% | 70.11% | **-3.65%** | Overfitting in transformer blocks |
| w/o Weighted Random Sampler *(Uniform)* | 83.90% | 71.20% | **-2.56%** | Artificial WAR inflation, rare classes neglected |
| Freeze Visual Backbone *(ViT lr = 0)* | 58.33% | 71.56% | **-2.20%** | Pretrained features misaligned with video domain |

---

### 3. Comparison with Prior SOTA Methods on RAER

| Method | Backbone | Modalities | Valid WAR (%) | Valid UAR (%) |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-18 + LSTM | ResNet-18 | Face | 65.40% | 61.20% |
| SlowFast 3D-CNN | 3D ResNet | Face + Body | 68.10% | 63.50% |
| Zero-Shot CLIP | CLIP ViT-B/16 | Video Frames | 58.70% | 52.30% |
| CAER-Net | 2D CNN | Face + Context | 69.50% | 64.80% |
| **RAPT-CLIP-FrameLevelFusion (Ours)** | **CLIP ViT-B/16** | **Face + Body + Context** | **78.79%** | **77.06%** |

---

## 📈 Learning Curves & Confusion Matrix

<p align="center">
  <img src="outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/log.png" width="49%" alt="Training Curves" />
  <img src="outputs/confusion_matrix/confusion_matrix_RAER.png" width="49%" alt="Confusion Matrix" />
</p>
<p align="center">
  <em>Figure 1: Training & Validation progression across 20 Epochs (left) and 5-class Confusion Matrix at peak Epoch 12/13 (right).</em>
</p>

---

## 🔍 Interpretability & Visualizations

### 1. t-SNE Feature Space Analysis
Visualizes the high-dimensional feature distributions before fine-tuning vs. after fine-tuning:

```bash
python run_tsne_rampdown.py --checkpoint outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth
```

<p align="center">
  <img src="outputs/tsne_rampdown/tsne_pretrained.png" width="48%" alt="Pretrained CLIP" />
  <img src="outputs/tsne_rampdown/tsne_finetuned_fused.png" width="48%" alt="Fine-tuned RAPT-CLIP" />
</p>
<p align="center">
  <em>Figure 2: Pretrained CLIP features are tangled (left); fine-tuned RAPT-CLIP yields well-separated emotion clusters (right).</em>
</p>

### 2. GradCAM Attention Heatmaps
Visualizes token-level attention on Face and Body crops:

```bash
python run_thesis_gradcam_v2.py
```

---

## 💻 Real-Time Live Webcam Demo

Run real-time facial and bodily emotion inference directly on your webcam:

```bash
python run_realtime.py --checkpoint outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth
```

---

## 📄 Technical PDF Report

A comprehensive 4-page technical specification report has been compiled directly from the model checkpoints:
* **[RAPT_CLIP_RAER_Hyperparameters_Report.pdf](RAPT_CLIP_RAER_Hyperparameters_Report.pdf)**: Contains complete hyperparameter audit tables, loss profiles, ablation statistics, and ready-to-use LaTeX tables for thesis and paper publication.

---

## 📜 Citation

If you find this repository useful in your research, please consider citing:

```bibtex
@article{rapt_clip_framelevelfusion_2026,
  title   = {RAPT-CLIP: Recognition of Academic Emotions through Prompt-Tuned CLIP with Frame-Level Fusion},
  author  = {Minh Ngoc},
  journal = {Preprint / Academic Thesis},
  year    = {2026},
  note    = {Triple-Stream Vision-Language Model with Cross-Modal Attention and Temporal Attention Pooling}
}
```

---

## ⚖️ License

This project is licensed for academic and non-commercial research purposes only.