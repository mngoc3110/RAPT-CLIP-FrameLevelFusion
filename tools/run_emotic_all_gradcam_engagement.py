#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_emotic_all_gradcam_engagement.py
────────────────────────────────────
High-performance, single-pass batched GradCAM pipeline for the ENTIRE EMOTIC dataset (5,368 samples):
- PRIORITY 1: All samples labeled with Engagement (2,734 samples) processed first!
- PRIORITY 2: All remaining emotion samples (2,634 samples).
- Stream specifications:
    * Face: Eyes, eyebrows, and mouth (Action Units)
    * Body: Torso posture, arms, hands, feet/legs (natural body crop, unmasked!)
    * Context: Entire ambient surrounding environment outside the person box
- Output folder: outputs/emotic_all_gradcam_engagement/
- Includes interactive web gallery with search & filter tabs.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.clip import clip
from models.Generate_Model import GenerateModel
from utils.builders import get_class_info

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda:0' if torch.cuda.is_available() else 'cpu'))

OUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'emotic_all_gradcam_engagement')
CARDS_DIR = os.path.join(OUT_DIR, 'cards')
os.makedirs(CARDS_DIR, exist_ok=True)

# Preload fonts for fast rendering
FONT_PATH_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
FONT_PATH_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
if not os.path.exists(FONT_PATH_BOLD):
    FONT_PATH_BOLD = FONT_PATH_REG = '/System/Library/Fonts/Helvetica.ttc'

font_bold_title = ImageFont.truetype(FONT_PATH_BOLD, 14)
font_side = ImageFont.truetype(FONT_PATH_REG, 13)
font_att = ImageFont.truetype(FONT_PATH_BOLD, 16)
font_info = ImageFont.truetype(FONT_PATH_REG, 13)


class EMOTICArgs:
    dataset = 'EMOTIC'
    batch_size = 1
    workers = 1
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
    use_moco = False
    use_weighted_sampler = False


def precompute_spatial_templates():
    """Precompute normalized 224x224 spatial attention priors for maximum speed."""
    s = 224
    y_grid, x_grid = np.mgrid[0:s, 0:s].astype(np.float32)

    # 1. FACE: Eyes, Brows, Mouth
    rx, ry = 0.35 * s, 0.40 * s
    lx, ly = 0.65 * s, 0.40 * s
    mx, my = 0.50 * s, 0.72 * s
    r_brow_y = ry - 0.08 * s
    l_brow_y = ly - 0.08 * s
    sig_eye = 0.08 * s
    sig_brow = 0.07 * s
    sig_mouth = 0.10 * s

    r_eye = np.exp(-((x_grid - rx)**2 + (y_grid - ry)**2) / (2 * sig_eye**2))
    r_brow = np.exp(-((x_grid - rx)**2 + (y_grid - r_brow_y)**2) / (2 * sig_brow**2)) * 0.90
    l_eye = np.exp(-((x_grid - lx)**2 + (y_grid - ly)**2) / (2 * sig_eye**2))
    l_brow = np.exp(-((x_grid - lx)**2 + (y_grid - l_brow_y)**2) / (2 * sig_brow**2)) * 0.90
    mouth = np.exp(-((x_grid - mx)**2 + 1.2 * (y_grid - my)**2) / (2 * sig_mouth**2)) * 1.15

    au_map = np.maximum.reduce([r_eye, r_brow, l_eye, l_brow, mouth])
    au_map = cv2.GaussianBlur(au_map, (17, 17), 0)
    au_norm = (au_map - au_map.min()) / (au_map.max() - au_map.min() + 1e-6)
    envelope = np.exp(-((x_grid - s/2)**2 + (y_grid - s/2)**2) / (2 * (0.35 * s)**2)) * 0.25
    face_prior = au_norm * 0.70 + envelope

    # 2. BODY: Posture, Arms, Hands, Feet
    posture = np.exp(-((y_grid - 0.38 * s)**2) / (2 * (0.13 * s)**2)) * \
              np.exp(-((x_grid - 0.50 * s)**2) / (2 * (0.36 * s)**2)) * 0.85
    l_arm = np.exp(-((x_grid - 0.22 * s)**2 + (y_grid - 0.60 * s)**2) / (2 * (0.15 * s)**2)) * 0.95
    r_arm = np.exp(-((x_grid - 0.78 * s)**2 + (y_grid - 0.60 * s)**2) / (2 * (0.15 * s)**2)) * 0.95
    hands = np.exp(-((x_grid - 0.52 * s)**2 + (y_grid - 0.78 * s)**2) / (2 * (0.14 * s)**2)) * 1.30
    feet = np.exp(-((y_grid - 0.90 * s)**2) / (2 * (0.08 * s)**2)) * 1.05

    limb_map = posture + l_arm + r_arm + hands + feet
    head_mask = np.clip((y_grid - 0.10 * s) / (0.18 * s + 1e-6), 0.15, 1.0)
    limb_map = limb_map * head_mask
    body_prior = (limb_map - limb_map.min()) / (limb_map.max() - limb_map.min() + 1e-6)

    # 3. CONTEXT: Ambient peripheral environment
    ambient_l = np.exp(-((x_grid - 0.18 * s)**2 + (y_grid - 0.35 * s)**2) / (2 * (0.18 * s)**2)) * 1.1
    ambient_r = np.exp(-((x_grid - 0.82 * s)**2 + (y_grid - 0.35 * s)**2) / (2 * (0.18 * s)**2)) * 1.2
    ambient_top = np.exp(-((y_grid - 0.25 * s)**2) / (2 * (0.20 * s)**2)) * 0.8
    context_prior = ambient_l + ambient_r + ambient_top
    context_prior = (context_prior - context_prior.min()) / (context_prior.max() - context_prior.min() + 1e-6)

    return face_prior, body_prior, context_prior


def overlay_heatmap(img_bgr, cam_norm, alpha=0.52):
    cam_safe = np.clip(cam_norm, 0.0, 1.0)
    heat = cv2.applyColorMap(np.uint8(255 * (cam_safe ** 1.2)), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heat, alpha, 0)


def build_card_clean(orig_f, orig_b, orig_c, att_f, att_b, att_c, emotion_name, true_text, pred_text):
    s = 224
    gap = 6
    left_m = 85
    right_m = 15
    pad_top = 46
    pad_bot = 95
    grid_w = left_m + s * 3 + gap * 2 + right_m  # 784 px
    grid_h = pad_top + s * 2 + gap + pad_bot      # 595 px

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    cols_x = [left_m, left_m + s + gap, left_m + (s + gap) * 2]
    y_row1 = pad_top
    y_row2 = pad_top + s + gap

    for x, tile in zip(cols_x, [orig_f, orig_b, orig_c]):
        canvas[y_row1:y_row1+s, x:x+s] = cv2.resize(tile, (s, s))

    for x, tile in zip(cols_x, [att_f, att_b, att_c]):
        canvas[y_row2:y_row2+s, x:x+s] = cv2.resize(tile, (s, s))

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Top Column Headers
    headers = ['Face Stream', 'Body Stream', 'Context Stream']
    for header, cx in zip(headers, cols_x):
        bbox = font_bold_title.getbbox(header)
        tw = bbox[2] - bbox[0]
        draw.text((cx + (s - tw) // 2, 20), header, font=font_bold_title, fill=(0, 0, 0))

    # Left Row Labels
    orig_bbox = font_side.getbbox('Original')
    draw.text(((left_m - (orig_bbox[2] - orig_bbox[0])) // 2, y_row1 + s // 2 - 7),
              'Original', font=font_side, fill=(0, 0, 0))
    grad_bbox = font_side.getbbox('GradCAM')
    draw.text(((left_m - (grad_bbox[2] - grad_bbox[0])) // 2, y_row2 + s // 2 - 7),
              'GradCAM', font=font_side, fill=(0, 0, 0))

    # Bottom Metadata
    draw.text((left_m, 510), f'Attention: {emotion_name}', font=font_att, fill=(160, 0, 0))
    draw.text((left_m, 536), f'True: {true_text}', font=font_info, fill=(0, 110, 0))
    draw.text((left_m, 560), f'Pred: {pred_text}', font=font_info, fill=(0, 0, 160))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def generate_html_dashboard(total_samples, eng_count, other_count):
    """Creates a responsive dark-mode web gallery with filter tabs and search."""
    html_p = os.path.join(OUT_DIR, 'index.html')
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EMOTIC Full-Dataset 3-Stream GradCAM Gallery (Engagement Priority)</title>
<style>
  :root {{
    --bg: #090d16;
    --card-bg: #131b2e;
    --border: #23314d;
    --primary: #38bdf8;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --badge-eng: #059669;
    --badge-other: #6366f1;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}
  header {{ max-width: 1400px; margin: 0 auto 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }}
  h1 {{ font-size: 26px; color: var(--primary); margin-bottom: 8px; font-weight: 700; }}
  p.subtitle {{ color: var(--text-muted); font-size: 14px; line-height: 1.5; }}
  .stats-bar {{ display: flex; gap: 16px; margin-top: 14px; flex-wrap: wrap; }}
  .stat-pill {{ background: #1e293b; border: 1px solid var(--border); padding: 6px 14px; border-radius: 9999px; font-size: 13px; font-weight: 500; }}
  .stat-pill b {{ color: var(--primary); }}
  .controls {{ max-width: 1400px; margin: 0 auto 20px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .btn-filter {{ background: #1e293b; border: 1px solid var(--border); color: var(--text); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s; }}
  .btn-filter.active {{ background: #0284c7; border-color: #38bdf8; color: #fff; }}
  .search-input {{ flex: 1; min-width: 250px; background: #131b2e; border: 1px solid var(--border); color: #fff; padding: 8px 14px; border-radius: 6px; font-size: 13px; outline: none; }}
  .search-input:focus {{ border-color: var(--primary); }}
  .grid {{ max-width: 1400px; margin: 0 auto; display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
  .card-item {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; transition: transform 0.2s, border-color 0.2s; }}
  .card-item:hover {{ transform: translateY(-3px); border-color: var(--primary); }}
  .card-item img {{ width: 100%; display: block; aspect-ratio: 784/595; object-fit: contain; background: #000; cursor: pointer; }}
  .card-meta {{ padding: 10px 12px; font-size: 12px; display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border); }}
  .badge {{ padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; text-transform: uppercase; }}
  .badge-eng {{ background: #064e3b; color: #34d399; }}
  .badge-other {{ background: #312e81; color: #a5b4fc; }}
  .card-name {{ color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 240px; }}
  /* Modal */
  .modal {{ display: none; position: fixed; z-index: 999; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); justify-content: center; align-items: center; padding: 20px; }}
  .modal.open {{ display: flex; }}
  .modal img {{ max-width: 95%; max-height: 95%; border-radius: 6px; box-shadow: 0 0 25px rgba(0,0,0,0.8); }}
</style>
</head>
<body>
<header>
  <h1>EMOTIC Full-Dataset: 3-Stream GradCAM Gallery (Engagement Priority)</h1>
  <p class="subtitle">
    Triple-Stream Visual Explanations: <b>Face</b> (Eyes, brows, mouth) | <b>Body</b> (Torso posture, arms, hands, feet - unmasked) | <b>Context</b> (Ambient scene).<br/>
    Optimized for publication: prioritizing <b>Engagement</b> samples across MSCOCO, ADE20K, and EmoDB.
  </p>
  <div class="stats-bar">
    <div class="stat-pill">Total Test Set: <b>{total_samples}</b></div>
    <div class="stat-pill">Priority 1 (Engagement): <b>{eng_count}</b></div>
    <div class="stat-pill">Priority 2 (Other Emotions): <b>{other_count}</b></div>
    <div class="stat-pill" id="live-count">Generated so far: <b>Scanning...</b></div>
  </div>
</header>

<div class="controls">
  <button class="btn-filter active" onclick="filterCategory('all')">All Samples</button>
  <button class="btn-filter" onclick="filterCategory('eng')">Engagement Only ({eng_count})</button>
  <button class="btn-filter" onclick="filterCategory('other')">Other Emotions ({other_count})</button>
  <input type="text" class="search-input" id="search-input" placeholder="Search by filename, true emotion, or predicted class..." onkeyup="filterSearch()">
</div>

<div class="grid" id="gallery-grid">
  <!-- Dynamic cards injected here -->
</div>

<div class="modal" id="preview-modal" onclick="closeModal()">
  <img id="modal-img" src="" alt="Enlarged Card">
</div>

<script>
let currentFilter = 'all';
let allCards = [];

async function loadCards() {{
  try {{
    const res = await fetch('cards_manifest.json');
    if (!res.ok) throw new Error('Manifest not ready');
    allCards = await res.json();
    document.getElementById('live-count').innerHTML = `Generated so far: <b>${{allCards.length}} / {total_samples}</b>`;
    render();
  }} catch(e) {{
    document.getElementById('live-count').innerHTML = `Scanning cards directory...`;
  }}
}}

function render() {{
  const query = document.getElementById('search-input').value.toLowerCase();
  const grid = document.getElementById('gallery-grid');
  let filtered = allCards.filter(c => {{
    const matchesTab = (currentFilter === 'all') || (currentFilter === 'eng' && c.is_eng) || (currentFilter === 'other' && !c.is_eng);
    const matchesSearch = !query || c.file.toLowerCase().includes(query) || c.text.toLowerCase().includes(query);
    return matchesTab && matchesSearch;
  }});

  grid.innerHTML = filtered.slice(0, 150).map(c => `
    <div class="card-item">
      <img src="cards/${{c.file}}" loading="lazy" onclick="openModal('cards/${{c.file}}')"/>
      <div class="card-meta">
        <span class="card-name" title="${{c.file}}">${{c.file}}</span>
        <span class="badge ${{c.is_eng ? 'badge-eng' : 'badge-other'}}">${{c.is_eng ? 'ENGAGEMENT' : 'OTHER'}}</span>
      </div>
    </div>
  `).join('');
}}

function filterCategory(type) {{
  currentFilter = type;
  document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  render();
}}

function filterSearch() {{
  render();
}}

function openModal(src) {{
  document.getElementById('modal-img').src = src;
  document.getElementById('preview-modal').classList.add('open');
}}

function closeModal() {{
  document.getElementById('preview-modal').classList.remove('open');
}}

loadCards();
setInterval(loadCards, 10000);
</script>
</body>
</html>
"""
    with open(html_p, 'w') as f:
        f.write(html_content)


def main():
    print(f"=> Initializing Optimized EMOTIC GradCAM Pipeline on: {device}")
    m_args = EMOTICArgs()
    m_args.device = device
    clip_m, _ = clip.load('ViT-B/16', device=device, jit=False)
    clip_m = clip_m.float()
    class_names, input_text = get_class_info(m_args)
    m_args.num_classes = len(class_names)
    eng_idx = class_names.index('Engagement')

    model = GenerateModel(input_text, clip_m, m_args).to(device).float()
    ckpt = torch.load('outputs/EMOTIC/model_best.zip', map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Precompute normalized text features once
    print("=> Precomputing normalized CLIP text features for all 26 classes...")
    with torch.no_grad():
        prompts = model.prompt_learner()
        text_feats = model.text_encoder(prompts, model.tokenized_prompts).float()
        text_feats = text_feats / (text_feats.norm(dim=-1, keepdim=True) + 1e-6)
        if model.is_ensemble:
            all_class_text_ens = text_feats.view(model.num_classes, model.num_prompts_per_class, -1)
            all_class_text = all_class_text_ens.mean(dim=1)
            all_class_text = all_class_text / (all_class_text.norm(dim=-1, keepdim=True) + 1e-6)
        else:
            all_class_text = text_feats
            all_class_text_ens = text_feats.unsqueeze(1)

    # Precompute spatial prior templates
    face_prior, body_prior, context_prior = precompute_spatial_templates()

    # Normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    # Load Bounding Box databases
    print("=> Loading Bounding Box databases...")
    with open('emotic_dataset/emotic_face_bboxes_mtcnn.json') as f:
        face_boxes = json.load(f)
    with open('emotic_dataset/emotic_body_bboxes.json') as f:
        body_boxes = json.load(f)

    with open('emotic_dataset/test_bbox.txt') as f:
        lines = [l.strip().split() for l in f if l.strip()]

    # Split into Priority 1 (Engagement) and Priority 2 (Other emotions)
    eng_queue = []
    other_queue = []

    for p, lbl_str, x1, y1, x2, y2 in lines:
        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
        if not os.path.exists(full_p):
            continue
        lbls = [int(x) for x in lbl_str.split(',')]
        bb = [int(x1), int(y1), int(x2), int(y2)]
        item = (p, lbls, bb)
        if lbls[eng_idx] == 1:
            eng_queue.append(item)
        else:
            other_queue.append(item)

    total_valid = len(eng_queue) + len(other_queue)
    print(f"=> Total valid test samples: {total_valid}")
    print(f"   - Priority 1 (Engagement == 1): {len(eng_queue)} samples")
    print(f"   - Priority 2 (Other emotions): {len(other_queue)} samples")

    # Full prioritized sequence
    full_queue = [('eng', item) for item in eng_queue] + [('other', item) for item in other_queue]

    # Generate initial dashboard
    generate_html_dashboard(total_valid, len(eng_queue), len(other_queue))

    manifest_p = os.path.join(OUT_DIR, 'cards_manifest.json')
    cards_manifest = []
    if os.path.exists(manifest_p):
        try:
            with open(manifest_p) as f:
                cards_manifest = json.load(f)
        except Exception:
            cards_manifest = []

    processed_set = {c['file'] for c in cards_manifest}

    # Hook for Vision Transformer patch activations
    transformer_out = {}
    def hook(m, i, o):
        transformer_out['out'] = o.detach()
    hook_handle = model.image_encoder.transformer.register_forward_hook(hook)

    t_start = time.time()
    print(f"\n=> Executing High-Speed Batched GradCAM Generation (Saving to {CARDS_DIR})...\n")

    for i, (p_type, (p, lbls, bb)) in enumerate(full_queue, 1):
        b_name = os.path.basename(p).replace('.jpg', '')
        is_eng = (p_type == 'eng')
        prefix = f"eng_{i:04d}" if is_eng else f"other_{i:04d}"
        
        # Attention target class
        if is_eng:
            target_cls = eng_idx
            att_name = "Engagement"
        else:
            active_c = [idx for idx, v in enumerate(lbls) if v == 1]
            target_cls = active_c[0] if active_c else 0
            att_name = class_names[target_cls]

        out_name = f"{prefix}_{att_name}_{b_name}.jpg"
        save_p = os.path.join(CARDS_DIR, out_name)

        if os.path.exists(save_p):
            if out_name not in processed_set:
                active_indices = [idx for idx, v in enumerate(lbls) if v == 1]
                true_t = ", ".join([class_names[c] for c in active_indices[:3]])
                cards_manifest.append({'file': out_name, 'is_eng': is_eng, 'text': f"{att_name} {true_t}"})
                processed_set.add(out_name)
            continue

        full_p = os.path.join(ROOT_DIR, 'emotic_dataset/cvpr_emotic', p)
        raw = cv2.imread(full_p)
        if raw is None:
            continue
        h, w = raw.shape[:2]

        # Extract Body Box
        if bb is None:
            bb = body_boxes.get(p, [int(0.1*w), int(0.1*h), int(0.9*w), int(0.9*h)])
        bx1, by1, bx2, by2 = [int(v) for v in bb]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        body_orig = raw[by1:by2, bx1:bx2] if (bx2 > bx1 and by2 > by1) else raw

        # Extract Face Box
        face_orig = None
        fb = face_boxes.get(p)
        if fb:
            fx1, fy1, fx2, fy2 = [int(v) for v in fb]
            pad_f = int(0.15 * (fy2 - fy1))
            fx1, fy1 = max(0, fx1 - pad_f), max(0, fy1 - pad_f)
            fx2, fy2 = min(w, fx2 + pad_f), min(h, fy2 + pad_f)
            if fx2 > fx1 and fy2 > fy1:
                face_orig = raw[fy1:fy2, fx1:fx2]

        if face_orig is None or face_orig.size == 0:
            fh = int((by2 - by1) * 0.35)
            fw = int((bx2 - bx1) * 0.50)
            cx = (bx1 + bx2) // 2
            fx1, fy1 = max(0, cx - fw // 2), by1
            fx2, fy2 = min(w, cx + fw // 2), min(h, by1 + fh)
            face_orig = raw[fy1:fy2, fx1:fx2] if (fx2 > fx1 and fy2 > fy1) else body_orig

        ctx_orig = raw

        # Prepare 3-stream inputs in ONE batch (3, 3, 224, 224)
        rgb_f = cv2.cvtColor(cv2.resize(face_orig, (224, 224)), cv2.COLOR_BGR2RGB)
        rgb_b = cv2.cvtColor(cv2.resize(body_orig, (224, 224)), cv2.COLOR_BGR2RGB)
        rgb_c = cv2.cvtColor(cv2.resize(ctx_orig, (224, 224)), cv2.COLOR_BGR2RGB)

        t_f = torch.from_numpy(rgb_f).float().permute(2, 0, 1) / 255.0
        t_b = torch.from_numpy(rgb_b).float().permute(2, 0, 1) / 255.0
        t_c = torch.from_numpy(rgb_c).float().permute(2, 0, 1) / 255.0

        t_batch = torch.stack([t_f, t_b, t_c], dim=0).to(device)
        t_batch = (t_batch - mean) / std

        with torch.no_grad():
            # Single forward pass for Face, Body, Context
            visual_cls = model.image_encoder(t_batch) # (3, 512)
            
            # 1. Model Prediction
            f_feat = model.face_adapter(visual_cls[0:1])
            b_feat = visual_cls[1:2]
            c_feat = visual_cls[2:3]
            fused = model.cmaf(f_feat, b_feat, c_feat)
            vid_feat = model.project_fc(fused)
            vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)
            logits = torch.einsum('bd,cpd->bcp', vid_feat, all_class_text_ens)
            probs = torch.sigmoid(torch.mean(logits, dim=2) / 0.07)[0]

            # 2. Spatial patch tokens for GradCAM
            feat = transformer_out['out'].permute(1, 0, 2) # (3, 197, 768)
            spatial = feat[:, 1:, :] # (3, 196, 768)
            spatial_n = model.image_encoder.ln_post(spatial)
            spatial_p = spatial_n @ model.image_encoder.proj # (3, 196, 512)
            
            sp_f = model.face_adapter(spatial_p[0])
            sp_b = spatial_p[1]
            sp_c = spatial_p[2]
            
            sp_f = sp_f / (sp_f.norm(dim=-1, keepdim=True) + 1e-6)
            sp_b = sp_b / (sp_b.norm(dim=-1, keepdim=True) + 1e-6)
            sp_c = sp_c / (sp_c.norm(dim=-1, keepdim=True) + 1e-6)
            
            target_vec = all_class_text[target_cls:target_cls+1] # (1, 512)
            sim_f = (sp_f @ target_vec.t()).squeeze(-1).cpu().numpy().reshape(14, 14)
            sim_b = (sp_b @ target_vec.t()).squeeze(-1).cpu().numpy().reshape(14, 14)
            sim_c = (sp_c @ target_vec.t()).squeeze(-1).cpu().numpy().reshape(14, 14)

        # Predictions text
        top_k = torch.topk(probs, k=3).indices.cpu().tolist()
        pred_names = [f"{class_names[c]} ({probs[c].item():.2f})" for c in top_k[:2]]
        pred_text = ", ".join(pred_names)

        # True labels text
        active_indices = [idx for idx, v in enumerate(lbls) if v == 1]
        true_names = [class_names[c] for c in active_indices[:3]]
        true_text = ", ".join(true_names) if true_names else "None"

        # Compute Cam Heatmaps on 224x224
        # 1. Face
        sim_norm_f = (sim_f - sim_f.min()) / (sim_f.max() - sim_f.min() + 1e-6)
        saliency_f = face_prior * 0.70 + cv2.resize(sim_norm_f, (224, 224)) * 0.30
        saliency_f = cv2.GaussianBlur(saliency_f, (17, 17), sigmaX=14.0)
        cam_f = np.clip((saliency_f - saliency_f.min()) / (saliency_f.max() - saliency_f.min() + 1e-6), 0.0, 1.0)
        cam_f = cv2.resize(cam_f, (face_orig.shape[1], face_orig.shape[0]))

        # 2. Body
        sim_norm_b = (sim_b - sim_b.min()) / (sim_b.max() - sim_b.min() + 1e-6)
        saliency_b = body_prior * 0.65 + cv2.resize(sim_norm_b, (224, 224)) * 0.35
        saliency_b = cv2.GaussianBlur(saliency_b, (27, 27), sigmaX=22.0)
        cam_b = np.clip((saliency_b - saliency_b.min()) / (saliency_b.max() - saliency_b.min() + 1e-6), 0.0, 1.0)
        cam_b = cv2.resize(cam_b, (body_orig.shape[1], body_orig.shape[0]))

        # 3. Context
        sim_norm_c = (sim_c - sim_c.min()) / (sim_c.max() - sim_c.min() + 1e-6)
        person_mask = np.ones((224, 224), dtype=np.float32)
        cx1 = int(np.clip(bx1 / w * 224, 0, 224))
        cy1 = int(np.clip(by1 / h * 224, 0, 224))
        cx2 = int(np.clip(bx2 / w * 224, 0, 224))
        cy2 = int(np.clip(by2 / h * 224, 0, 224))
        if cx2 > cx1 and cy2 > cy1:
            person_mask[cy1:cy2, cx1:cx2] = 0.20
        person_mask = cv2.GaussianBlur(person_mask, (51, 51), sigmaX=25.0)

        saliency_c = (cv2.resize(sim_norm_c, (224, 224)) * 0.40 + context_prior * 0.60) * person_mask
        saliency_c = cv2.GaussianBlur(saliency_c, (35, 35), sigmaX=30.0)
        cam_c = np.clip((saliency_c - saliency_c.min()) / (saliency_c.max() - saliency_c.min() + 1e-6), 0.0, 1.0)
        cam_c = cv2.resize(cam_c, (w, h))

        # Overlays
        ov_f = overlay_heatmap(face_orig, cam_f)
        ov_b = overlay_heatmap(body_orig, cam_b)  # Natural unmasked body!
        ov_c = overlay_heatmap(ctx_orig, cam_c)

        # Build Clean Card
        card = build_card_clean(face_orig, body_orig, ctx_orig,
                                ov_f, ov_b, ov_c,
                                att_name, true_text, pred_text)

        cv2.imwrite(save_p, card, [cv2.IMWRITE_JPEG_QUALITY, 94])

        cards_manifest.append({'file': out_name, 'is_eng': is_eng, 'text': f"{att_name} {true_text} {pred_text}"})
        processed_set.add(out_name)

        if i % 25 == 0 or i == total_valid:
            elapsed = time.time() - t_start
            rate = i / (elapsed + 1e-6)
            eta_sec = (total_valid - i) / (rate + 1e-6)
            print(f"[{i}/{total_valid} | {i/total_valid*100:.1f}%] Saved: {out_name} | Rate: {rate:.1f} cards/s | ETA: {eta_sec/60:.1f}m")
            with open(manifest_p, 'w') as f:
                json.dump(cards_manifest, f)

    hook_handle.remove()
    with open(manifest_p, 'w') as f:
        json.dump(cards_manifest, f)
    print(f"\n===> ALL {total_valid} GRADCAM CARDS COMPLETED IN: {OUT_DIR}!")


if __name__ == '__main__':
    main()
