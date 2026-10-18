#!/usr/bin/env python3
"""
tools/render_model_architecture.py
Generates a publication-grade, high-resolution visual architecture diagram
matching the LaTeX TikZ specification in rapt_clip_architecture.tex.
Explicitly highlights the intermediate operations between CMAF and Temporal Transformer:
  1. Channel-wise Concatenation [f'_F || f'_B || f'_C] -> 1536-d
  2. Temporal Sequence Reshaping / Stacking: (B*T, 1536) -> (B, 16, 1536)
  3. Learnable Temporal Positional Embedding Addition (+ E_pos)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os

def draw_rounded_box(ax, xy, width, height, box_color, border_color, title, subtitle=None, dims=None, corner_radius=0.012, alpha=1.0, title_color='#1A252C', title_weight='bold', title_size=8.5):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.010,rounding_size={corner_radius}",
        facecolor=box_color,
        edgecolor=border_color,
        linewidth=1.2,
        alpha=alpha,
        zorder=3
    )
    ax.add_patch(box)
    
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    
    if subtitle is None and dims is None:
        ax.text(center_x, center_y, title, ha='center', va='center', fontsize=title_size, fontweight=title_weight, color=title_color, zorder=5)
    elif dims is None:
        ax.text(center_x, center_y + height * 0.18, title, ha='center', va='center', fontsize=title_size, fontweight=title_weight, color=title_color, zorder=5)
        ax.text(center_x, center_y - height * 0.18, subtitle, ha='center', va='center', fontsize=title_size-1.5, color='#444444', zorder=5)
    elif subtitle is None:
        ax.text(center_x, center_y + height * 0.18, title, ha='center', va='center', fontsize=title_size, fontweight=title_weight, color=title_color, zorder=5)
        ax.text(center_x, center_y - height * 0.18, dims, ha='center', va='center', fontsize=title_size-2.0, fontfamily='monospace', color='#555555', zorder=5)
    else:
        ax.text(center_x, center_y + height * 0.28, title, ha='center', va='center', fontsize=title_size, fontweight=title_weight, color=title_color, zorder=5)
        ax.text(center_x, center_y, subtitle, ha='center', va='center', fontsize=title_size-1.8, color='#333333', zorder=5)
        ax.text(center_x, center_y - height * 0.28, dims, ha='center', va='center', fontsize=title_size-2.2, fontfamily='monospace', color='#555555', zorder=5)

def draw_arrow(ax, start, end, label=None, label_pos='above', color='#2C3E50', lw=1.2, dashed=False, rad=0.0):
    ls = '--' if dashed else '-'
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            ls=ls,
            connectionstyle=connectionstyle,
            mutation_scale=11,
            shrinkA=0, shrinkB=0
        ),
        zorder=4
    )
    if label:
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        offset_y = 0.015 if label_pos == 'above' else -0.018
        ax.text(mid_x, mid_y + offset_y, label, ha='center', va='center', fontsize=7.2, fontweight='bold', color=color, zorder=6,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.9))

def main():
    os.makedirs('image', exist_ok=True)
    fig = plt.figure(figsize=(21, 14.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    # Color Palette
    vis_bg = '#F6FAFF'
    text_bg = '#F5FDF7'
    clip_blue = '#EBF3FE'
    clip_border = '#2980B9'
    eaa_green = '#EBFBF1'
    eaa_border = '#27AE60'
    cmaf_orange = '#FEF6ED'
    cmaf_border = '#E67E22'
    bridge_amber = '#FFF9E6'
    bridge_border = '#D4AC0D'
    temp_purple = '#F7F0FC'
    temp_border = '#8E44AD'
    pool_pink = '#FDEFF7'
    pool_border = '#D63384'
    proj_gray = '#F2F4F4'
    proj_border = '#7F8C8D'
    loss_red = '#FDEDEC'
    loss_border = '#E74C3C'
    midc_yellow = '#FEF9E7'
    midc_border = '#F39C12'
    prompt_green = '#E6F9F5'
    prompt_border = '#16A085'
    desc_purple = '#F3F0FD'
    desc_border = '#6C5CE7'

    # Title Banner
    ax.text(0.5, 0.985, "RAPT-CLIP: Unified Framework for Context-Aware Video Emotion Recognition", 
            ha='center', va='center', fontsize=17, fontweight='bold', color='#1A252C')
    ax.text(0.5, 0.968, "Detailed Micro-Architectural Topology with Explicit Intermediate Stages between CMAF and Temporal Transformer", 
            ha='center', va='center', fontsize=10.5, fontstyle='italic', color='#555555')

    # =========================================================================
    # 1. TOP CONTAINER: VISUAL BRANCH (y: 0.535 -> 0.95)
    # =========================================================================
    vis_panel = FancyBboxPatch((0.015, 0.535), 0.97, 0.415, boxstyle="round,pad=0.015,rounding_size=0.02",
                               facecolor=vis_bg, edgecolor='#B0C8E8', linewidth=1.5, zorder=1)
    ax.add_patch(vis_panel)
    ax.text(0.03, 0.932, "VISUAL BRANCH: Multi-Stream Extraction, Frame-Level CMAF Fusion, Temporal Bridging & Unified Modeling",
            fontsize=12, fontweight='bold', color='#184C78', zorder=2)

    # 1.1 Inputs (Face, Body, Context)
    draw_rounded_box(ax, (0.025, 0.81), 0.08, 0.085, 'white', '#7F8C8D', 
                     "Face Stream", r"$F_{1..T}$ Crops", r"(B, T, 3, 224, 224)", title_size=8.0)
    draw_rounded_box(ax, (0.025, 0.68), 0.08, 0.085, 'white', '#7F8C8D', 
                     "Body Stream", r"$B_{1..T}$ Crops", r"(B, T, 3, 224, 224)", title_size=8.0)
    draw_rounded_box(ax, (0.025, 0.55), 0.08, 0.085, 'white', '#7F8C8D', 
                     "Context Stream", r"$C_{1..T}$ Scene", r"(B, T, 3, 224, 224)", title_size=8.0)

    # 1.2 CLIP ViT-B/16 Encoders (Shared Weights)
    draw_rounded_box(ax, (0.13, 0.81), 0.095, 0.085, clip_blue, clip_border, 
                     "CLIP ViT-B/16", "12 Layers, 196 Patches", r"$\mathbf{z}_t^{\mathrm{face}} \in \mathbb{R}^{B \cdot T \times 512}$", title_size=8.0)
    draw_rounded_box(ax, (0.13, 0.68), 0.095, 0.085, clip_blue, clip_border, 
                     "CLIP ViT-B/16", "Shared Weights $\\theta_{\\mathrm{ViT}}$", r"$\mathbf{z}_t^{\mathrm{body}} \in \mathbb{R}^{B \cdot T \times 512}$", title_size=8.0)
    draw_rounded_box(ax, (0.13, 0.55), 0.095, 0.085, clip_blue, clip_border, 
                     "CLIP ViT-B/16", "Shared Weights $\\theta_{\\mathrm{ViT}}$", r"$\mathbf{z}_t^{\mathrm{ctx}} \in \mathbb{R}^{B \cdot T \times 512}$", title_size=8.0)

    # Dashed connection for shared weights
    ax.plot([0.177, 0.177], [0.81, 0.765], color='#2980B9', linestyle=':', lw=1.4, zorder=4)
    ax.plot([0.177, 0.177], [0.68, 0.635], color='#2980B9', linestyle=':', lw=1.4, zorder=4)
    ax.text(0.177, 0.787, "Shared", ha='center', va='center', fontsize=6.2, color='#2980B9', bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax.text(0.177, 0.657, "Shared", ha='center', va='center', fontsize=6.2, color='#2980B9', bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))

    # Input to CLIP arrows
    draw_arrow(ax, (0.105, 0.852), (0.13, 0.852))
    draw_arrow(ax, (0.105, 0.722), (0.13, 0.722))
    draw_arrow(ax, (0.105, 0.592), (0.13, 0.592))

    # 1.3 Face Adapter (EAA)
    draw_rounded_box(ax, (0.25, 0.81), 0.095, 0.085, eaa_green, eaa_border, 
                     "Face Adapter (EAA)", "Bottleneck $512 \\rightarrow 128 \\rightarrow 512$\nResidual + ReLU", r"$\mathbf{f}_t^{\mathrm{face}} \in \mathbb{R}^{B \cdot T \times 512}$", title_size=8.0)
    draw_arrow(ax, (0.225, 0.852), (0.25, 0.852))

    # 1.4 CMAF Cross-Attention Core
    draw_rounded_box(ax, (0.375, 0.55), 0.115, 0.345, cmaf_orange, cmaf_border, 
                     "CMAF Cross-Attention\nPer-Frame Core", 
                     "Tridirectional Cross-Attn ($h=4$):\n" +
                     "• Q_F x [K_B, K_C] -> f'_F\n" +
                     "• Q_B x [K_F, K_C] -> f'_B\n" +
                     "• Q_C x [K_F, K_B] -> f'_C\n\n" +
                     "3 Residual LayerNorm\nOutputs (each 512-d)", 
                     r"$\mathbf{f}_t^{\prime F}, \mathbf{f}_t^{\prime B}, \mathbf{f}_t^{\prime C}$", title_size=8.5)

    draw_arrow(ax, (0.345, 0.852), (0.375, 0.852), label="Mod. Drop (p=0.3)", label_pos='above')
    draw_arrow(ax, (0.225, 0.722), (0.375, 0.722))
    draw_arrow(ax, (0.225, 0.592), (0.375, 0.592))

    # =========================================================================
    # [HIGHLIGHTED INTERMEDIATE STAGES BETWEEN CMAF AND TEMPORAL TRANSFORMER]
    # =========================================================================
    
    # Bridge Stage 1: Channel-wise Concatenation
    draw_rounded_box(ax, (0.515, 0.66), 0.08, 0.125, bridge_amber, bridge_border, 
                     "Stage 1: Channel\nConcatenation", 
                     r"$\mathbf{f}_t = [\mathbf{f}_t^{\prime F} \,\|\, \mathbf{f}_t^{\prime B} \,\|\, \mathbf{f}_t^{\prime C}]$" + "\n" +
                     "torch.cat(dim=-1)\n$512 \\times 3 \\rightarrow 1536$", 
                     r"$\mathbf{f}_t \in \mathbb{R}^{(B \cdot T) \times 1536}$", title_size=8.0)
    draw_arrow(ax, (0.490, 0.722), (0.515, 0.722), label="3x512", label_pos='above')

    # Bridge Stage 2: Temporal Sequence Reshaping (Stacking frames into video sequence)
    draw_rounded_box(ax, (0.615, 0.66), 0.085, 0.125, bridge_amber, bridge_border, 
                     "Stage 2: Temporal\nSequence Reshape", 
                     r"$\text{view}(B, T=16, 1536)$" + "\n" +
                     "Frames -> Video Series\n" +
                     r"$[\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_{16}]$", 
                     r"$\mathbf{F}_{\mathrm{seq}} \in \mathbb{R}^{B \times 16 \times 1536}$", title_size=8.0)
    draw_arrow(ax, (0.595, 0.722), (0.615, 0.722), label=r"$\mathbf{f}_t$", label_pos='above')

    # Bridge Stage 3: Temporal Positional Encoding Addition
    # Box for learnable parameter E_pos
    draw_rounded_box(ax, (0.695, 0.81), 0.075, 0.075, '#FFFDF0', '#B7950B',
                     "Temporal Pos.\nEmbedding", r"$\mathbf{E}_{\mathrm{pos}}$ Learnable", r"(1, 16, 1536)", title_size=7.5)
    
    # Add circle symbol
    ax.add_patch(patches.Circle((0.732, 0.722), 0.013, facecolor='#FEF9E7', edgecolor='#B7950B', lw=1.3, zorder=5))
    ax.text(0.732, 0.722, r"$\oplus$", ha='center', va='center', fontsize=11, fontweight='bold', color='#7D6608', zorder=6)
    
    # Arrow from Stage 2 into Add circle
    draw_arrow(ax, (0.700, 0.722), (0.719, 0.722))
    # Arrow from E_pos down into Add circle
    draw_arrow(ax, (0.732, 0.81), (0.732, 0.735))

    # 1.5 Unified Temporal Transformer
    draw_rounded_box(ax, (0.765, 0.66), 0.095, 0.125, temp_purple, temp_border, 
                     "Unified Temporal\nTransformer", 
                     "1 Layer, $T=16$, $h=8$\nPre-LN + MHSA(8)\n+ Residual FFN(1024)", 
                     r"$\mathbf{H} \in \mathbb{R}^{B \times 16 \times 1536}$", title_size=8.0)
    draw_arrow(ax, (0.745, 0.722), (0.765, 0.722), label=r"$\mathbf{X}$", label_pos='above')

    # 1.6 Attention Pooling
    draw_rounded_box(ax, (0.875, 0.66), 0.065, 0.125, pool_pink, pool_border, 
                     "Attention\nPooling", 
                     r"$\mathbf{u}_t = \tanh(\mathbf{W}_1 \mathbf{h}_t)$" + "\n" +
                     r"$\alpha_t = \mathrm{softmax}(\mathbf{w}_2^\top \mathbf{u}_t)$" + "\n" +
                     r"$\mathbf{F} = \sum \alpha_t \mathbf{h}_t$", 
                     r"(B, 1536)", title_size=7.8)
    draw_arrow(ax, (0.860, 0.722), (0.875, 0.722), label=r"$\mathbf{h}_t$", label_pos='above')

    # Vector F
    ax.add_patch(patches.Circle((0.950, 0.722), 0.011, facecolor='#FDEDEC', edgecolor='#C0392B', lw=1.2, zorder=5))
    ax.text(0.950, 0.722, r"$\mathbf{F}$", ha='center', va='center', fontsize=9, fontweight='bold', color='#922B21', zorder=6)
    draw_arrow(ax, (0.940, 0.722), (0.939, 0.722))

    # Linear Proj & L2-Norm
    draw_rounded_box(ax, (0.968, 0.66), 0.025, 0.125, proj_gray, proj_border, 
                     r"$\mathbf{v}$", r"$\mathbf{W}_{\mathrm{p}}$" + "\n" + r"$\ell_2$", r"512", title_size=9.0)
    draw_arrow(ax, (0.961, 0.722), (0.968, 0.722))

    # =========================================================================
    # 2. MIDDLE CONTAINER: TEXT BRANCH (y: 0.285 -> 0.515)
    # =========================================================================
    text_panel = FancyBboxPatch((0.015, 0.285), 0.97, 0.23, boxstyle="round,pad=0.015,rounding_size=0.02",
                                facecolor=text_bg, edgecolor='#A3E4D7', linewidth=1.5, zorder=1)
    ax.add_patch(text_panel)
    ax.text(0.03, 0.495, "TEXT BRANCH: Dual-Prompt Learning (CoOp Tuning + Hand-Crafted Clinical Anchors)",
            fontsize=12, fontweight='bold', color='#1E824C', zorder=2)

    # 2.1 View 1: Prompt Ensemble
    draw_rounded_box(ax, (0.025, 0.395), 0.14, 0.08, prompt_green, prompt_border, 
                     "Prompt Ensemble (View 1)", "K=5 classes x P=3 templates\ne.g., 'a photo of [CLS] student'", r"(K \cdot P, \text{tokens})", title_size=8.0)

    # 2.2 Prompt Learner (CoOp)
    draw_rounded_box(ax, (0.19, 0.395), 0.13, 0.08, 'white', prompt_border, 
                     "Prompt Learner (CoOp)", "M=8 Learnable Soft Tokens\n$[SOS] \oplus [V]_1..[V]_M \oplus [CLS]$", r"(K \cdot P, 77, 512)", title_size=8.0)
    draw_arrow(ax, (0.165, 0.435), (0.19, 0.435))

    # 2.3 View 2: Detailed Clinical Descriptors
    draw_rounded_box(ax, (0.025, 0.30), 0.295, 0.08, desc_purple, desc_border, 
                     "Semantic Clinical Descriptors (View 2 - Anchors)", 
                     "Detailed physiological cues: e.g., 'droopy eyelids, head tilting down, yawning' (Fatigue)", 
                     r"(K, 77, \text{tokens})", title_size=8.0)

    # 2.4 CLIP Text Encoder
    draw_rounded_box(ax, (0.355, 0.30), 0.12, 0.175, clip_blue, clip_border, 
                     "CLIP Text Encoder\n(12-Layer Transformer)", 
                     "Shared Embedding Matrix\nEOT Token Extraction\nText Projection $\\rightarrow \\ell_2$-norm", 
                     r"out: (K, 512)", title_size=8.5)
    draw_arrow(ax, (0.32, 0.435), (0.355, 0.435))
    draw_arrow(ax, (0.32, 0.34), (0.355, 0.34))

    # 2.5 Text Output Embeddings
    draw_rounded_box(ax, (0.505, 0.405), 0.09, 0.07, prompt_green, prompt_border, 
                     "View 1: $\mathbf{T}^{\mathrm{learn}}$", "Ensemble Averaged\nLearned Embeddings", r"(K, 512)", title_size=7.5)
    draw_rounded_box(ax, (0.505, 0.305), 0.09, 0.07, desc_purple, desc_border, 
                     "View 2: $\mathbf{T}^{\mathrm{desc}}$", "Frozen Semantic\nAnchor Embeddings", r"(K, 512)", title_size=7.5)
    draw_arrow(ax, (0.475, 0.44), (0.505, 0.44))
    draw_arrow(ax, (0.475, 0.34), (0.505, 0.34))

    # 2.6 Regularization Losses: MI & DC Loss
    draw_rounded_box(ax, (0.635, 0.30), 0.145, 0.175, midc_yellow, midc_border, 
                     "Semantic Regularization\n(MI Loss + DC Loss)", 
                     "• L_MI: Symmetrical InfoNCE\n" +
                     "   Aligns T^learn <-> T^desc\n" +
                     "• L_DC: Decorrelation Penalty\n" +
                     "   Minimizes sum_{i!=j} (t_i^T t_j)^2", 
                     "L_reg = lambda_MI * L_MI + lambda_DC * L_DC", title_size=8.5)
    draw_arrow(ax, (0.595, 0.44), (0.635, 0.44))
    draw_arrow(ax, (0.595, 0.34), (0.635, 0.34))

    # 2.7 Primary Classification Loss: LDAM Loss
    draw_rounded_box(ax, (0.825, 0.30), 0.155, 0.175, loss_red, loss_border, 
                     "LDAM Classification Loss\n(Imbalance-Aware)", 
                     "Logits: z_k = (v^T t_k) / tau\n" +
                     "Margin: Delta_y = C / N_y^(1/4)\n" +
                     "scale s=30, temp tau=0.07\n" +
                     "Balances rare vs majority classes", 
                     "L_total = L_LDAM + 0.1*L_MI + 0.1*L_DC", title_size=8.5)

    # Route View 1 text features into LDAM Loss
    draw_arrow(ax, (0.595, 0.46), (0.825, 0.46), rad=-0.15, color='#16A085', lw=1.3)
    ax.text(0.710, 0.478, "Text Prototypes $\mathbf{t}_k$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#16A085',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9))

    # Visual Vector v to LDAM Loss (Dashed Curve from Visual Branch)
    ax.annotate(
        '', xy=(0.89, 0.475), xytext=(0.98, 0.66),
        arrowprops=dict(
            arrowstyle="-|>",
            color='#E74C3C',
            lw=1.5,
            ls='--',
            connectionstyle="arc3,rad=-0.22",
            mutation_scale=12
        ),
        zorder=5
    )
    ax.text(0.965, 0.575, r"Cosine Sim: $\mathbf{v}^\top \mathbf{t}_k$", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#C0392B',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E74C3C', lw=1.0))

    # =========================================================================
    # 3. BOTTOM PANEL: ZOOM-IN SUB-MODULE ARCHITECTURES (y: 0.015 -> 0.265)
    # =========================================================================
    inset_panel = FancyBboxPatch((0.015, 0.015), 0.97, 0.25, boxstyle="round,pad=0.015,rounding_size=0.02",
                                 facecolor='#FAFAFA', edgecolor='#D5D8DC', linewidth=1.2, zorder=1)
    ax.add_patch(inset_panel)
    ax.text(0.03, 0.245, "DETAILED INTERNAL MICRO-ARCHITECTURES (Bridging & Sub-Module Breakdowns)",
            fontsize=11.5, fontweight='bold', color='#2C3E50', zorder=2)

    # Inset (a): EAA Bottleneck
    draw_rounded_box(ax, (0.025, 0.03), 0.21, 0.195, eaa_green, eaa_border, 
                     "(a) EAA Bottleneck Adapter", 
                     "• Input: $\mathbf{z}_t^{\mathrm{face}} \in \mathbb{R}^{512}$ (CLIP Token)\n" +
                     "• Linear Down: $512 \\rightarrow 128$ (dim // 4)\n" +
                     "• Activation: $\\mathrm{ReLU} + \\mathrm{Dropout}(0.5)$\n" +
                     "• Linear Up: $128 \\rightarrow 512$\n" +
                     "• Residual Add: $\\mathbf{z}_t^{\\mathrm{face}} + \\Delta \\mathbf{z}_t^{\\mathrm{face}}$\n" +
                     "• Modality Dropout: $p=0.3$ (Occlusion-proof)", 
                     "Preserves base CLIP semantic space", title_size=8.5)

    # Inset (b): CMAF-to-Temporal Bridge Stages (NEW DETAILED BREAKDOWN!)
    draw_rounded_box(ax, (0.250, 0.03), 0.245, 0.195, bridge_amber, bridge_border, 
                     "(b) CMAF-to-Temporal Bridging Flow", 
                     "1. Cross-Attn: Q_F, Q_B, Q_C attend pairs -> f'_F, f'_B, f'_C\n" +
                     "2. Channel Concat: [f'_F || f'_B || f'_C] -> f_t in R^(B*T x 1536)\n" +
                     "3. Temporal Stacking: contiguous().view(B, T=16, 1536)\n" +
                     "   Groups flat per-frame vectors into video sequence F_seq\n" +
                     "4. Positional Add: X = F_seq + E_pos (1 x 16 x 1536)\n" +
                     "   Breaks permutation-invariance for chronological attention", 
                     "Essential bridge from frame-level to video-level", title_size=8.5)

    # Inset (c): Attention Pooling
    draw_rounded_box(ax, (0.510, 0.03), 0.235, 0.195, pool_pink, pool_border, 
                     "(c) Dynamic Attention Pooling", 
                     "• Input Tokens: $\\mathbf{H} = [\\mathbf{h}_1, ..., \\mathbf{h}_{16}] \\in \\mathbb{R}^{16 \\times 1536}$\n" +
                     r"• Scoring MLP: $\mathbf{u}_t = \tanh(\mathbf{W}_1 \mathbf{h}_t + \mathbf{b}_1) \in \mathbb{R}^{512}$" + "\n" +
                     r"• Softmax Weights: $\alpha_t = \frac{\exp(\mathbf{w}_2^\top \mathbf{u}_t)}{\sum_j \exp(\mathbf{w}_2^\top \mathbf{u}_j)}$" + "\n" +
                     r"• Aggregated Vector: $\mathbf{F} = \sum_{t=1}^{16} \alpha_t \mathbf{h}_t$" + "\n" +
                     "• Linear Head: $\\mathbf{W}_{\\mathrm{proj}} (1536 \\rightarrow 512) + \\ell_2\\text{-norm}$", 
                     "Pinpoints dynamic peak emotional keyframes", title_size=8.5)

    # Inset (d): Multi-Objective Optimization
    draw_rounded_box(ax, (0.760, 0.03), 0.225, 0.195, loss_red, loss_border, 
                     "(d) Multi-Objective Loss Joint", 
                     "• L_LDAM: Tail-margin penalty (s=30, tau=0.07)\n" +
                     "• L_MI: Symmetrical InfoNCE (T=0.07)\n" +
                     "• L_DC: Text feature orthogonalization\n" +
                     "• L_total = L_LDAM + 0.1*L_MI + 0.1*L_DC\n" +
                     "• Canonical SOTA: 77.06% UAR / 77.65% WAR", 
                     "Guarantees high UAR without class collapse", title_size=8.5)

    plt.tight_layout()
    fig.savefig('image/model_detailed.png', dpi=300, bbox_inches='tight')
    fig.savefig('image/model_detailed.pdf', bbox_inches='tight')
    print("=> Re-rendered image/model_detailed.png and image/model_detailed.pdf with explicit CMAF-to-Temporal stages!")

if __name__ == '__main__':
    main()
