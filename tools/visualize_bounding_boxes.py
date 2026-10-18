#!/usr/bin/env python3
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

# Artifacts output directory
ARTIFACT_DIR = "/Users/macbook/.gemini/antigravity-ide/brain/edd9a27d-6868-4e16-a02a-191a9c0986f9"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1. CAER-S Static Samples
caer_s_faces_path = "CAER-S/annotations/caer_s_faces.json"
caer_s_faces = json.load(open(caer_s_faces_path))

sample_keys = [
    ("Happy", "test/Happy/1804.png"),
    ("Anger", "test/Anger/1804.png"),
    ("Sad", "test/Sad/1804.png"),
    ("Surprise", "test/Surprise/1804.png"),
    ("Neutral", "test/Neutral/1804.png"),
    ("Fear", "test/Fear/1804.png"),
]

print("=" * 80)
print(f"{'EMOTION':<10} | {'IMAGE PATH':<25} | {'IMAGE SIZE':<12} | {'FACE BBOX [x1, y1, x2, y2]':<25} | {'BODY BBOX [x1, y1, x2, y2]'}")
print("=" * 80)

caer_s_results = []

for emotion, rel_key in sample_keys:
    full_path = os.path.join("CAER-S", rel_key)
    if not os.path.exists(full_path):
        continue

    img = Image.open(full_path).convert('RGB')
    w, h = img.size

    # Raw Face box
    raw_face = caer_s_faces.get(rel_key, [int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.75)])
    if isinstance(raw_face[0], list):
        raw_face = raw_face[0]
    fx1, fy1, fx2, fy2 = raw_face[:4]

    # Padded Face box (15% padding used in RAPT-CLIP dataloader)
    bw = fx2 - fx1
    bh = fy2 - fy1
    pad_w = int(bw * 0.15)
    pad_h = int(bh * 0.15)
    pfx1 = max(0, fx1 - pad_w)
    pfy1 = max(0, fy1 - pad_h)
    pfx2 = min(w, fx2 + pad_w)
    pfy2 = min(h, fy2 + pad_h)

    # Torso/Body Box estimation (anatomical ratio used in RAPT-CLIP)
    bx1 = max(0, int(fx1 - bw * 0.8))
    by1 = max(0, int(fy1))
    bx2 = min(w, int(fx2 + bw * 0.8))
    by2 = min(h, int(fy2 + bh * 2.5))

    # Crops (resized to 224x224 for ViT input)
    face_crop = img.crop((pfx1, pfy1, pfx2, pfy2)).resize((224, 224), Image.BILINEAR)
    body_crop = img.crop((bx1, by1, bx2, by2)).resize((224, 224), Image.BILINEAR)

    # Masked Context (masking face region to prevent face leakage into scene features)
    masked_context = img.copy()
    draw = ImageDraw.Draw(masked_context)
    draw.rectangle([max(0, fx1 - 5), max(0, fy1 - 5), min(w, fx2 + 5), min(h, fy2 + 5)], fill=(0, 0, 0))
    masked_context_resized = masked_context.resize((224, 224), Image.BILINEAR)

    print(f"{emotion:<10} | {rel_key:<25} | {f'{w}x{h}':<12} | {str([fx1, fy1, fx2, fy2]):<25} | {str([bx1, by1, bx2, by2])}")

    caer_s_results.append({
        'emotion': emotion,
        'rel_key': rel_key,
        'orig_img': img,
        'face_crop': face_crop,
        'body_crop': body_crop,
        'masked_context': masked_context_resized,
        'raw_face': [fx1, fy1, fx2, fy2],
        'padded_face': [pfx1, pfy1, pfx2, pfy2],
        'body_box': [bx1, by1, bx2, by2],
        'w': w, 'h': h
    })

print("=" * 80)

# 2. Render CAER-S Multi-Stream Comparison Grid
fig, axes = plt.subplots(len(caer_s_results), 4, figsize=(18, 4.2 * len(caer_s_results)))
if len(caer_s_results) == 1:
    axes = np.expand_dims(axes, 0)

col_titles = [
    "Full Scene with BBoxes\n(Cyan: Face, Lime: Body/Torso)",
    "Face Stream (ViT Input)\n[Padded 15% Crop -> 224x224]",
    "Body Stream (ViT Input)\n[Torso Anatomy Crop -> 224x224]",
    "Context Stream (ViT Input)\n[Face Masked Out -> 224x224]"
]

for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold', pad=12)

for row_idx, res in enumerate(caer_s_results):
    orig = res['orig_img']
    fx1, fy1, fx2, fy2 = res['raw_face']
    pfx1, pfy1, pfx2, pfy2 = res['padded_face']
    bx1, by1, bx2, by2 = res['body_box']

    # Subplot 0: Original image with Bounding Boxes
    ax0 = axes[row_idx, 0]
    ax0.imshow(orig)

    # Face BBox: Cyan solid box + label
    face_rect = patches.Rectangle(
        (fx1, fy1), fx2 - fx1, fy2 - fy1,
        linewidth=2.5, edgecolor='#00E5FF', facecolor='none', linestyle='-'
    )
    ax0.add_patch(face_rect)
    ax0.text(
        fx1, max(0, fy1 - 6),
        f"Face [{fx1},{fy1},{fx2},{fy2}]",
        color='white', fontsize=9, fontweight='bold',
        bbox=dict(facecolor='#00E5FF', edgecolor='none', alpha=0.8, pad=1.5)
    )

    # Body BBox: Lime solid box + label
    body_rect = patches.Rectangle(
        (bx1, by1), bx2 - bx1, by2 - by1,
        linewidth=2.5, edgecolor='#00E676', facecolor='none', linestyle='--'
    )
    ax0.add_patch(body_rect)
    ax0.text(
        bx1, min(res['h'] - 4, by2 + 14),
        f"Body [{bx1},{by1},{bx2},{by2}]",
        color='black', fontsize=9, fontweight='bold',
        bbox=dict(facecolor='#00E676', edgecolor='none', alpha=0.8, pad=1.5)
    )

    ax0.set_ylabel(f"{res['emotion']}\n({res['w']}x{res['h']})", fontsize=12, fontweight='bold')
    ax0.axis('off')

    # Subplot 1: Face Crop
    ax1 = axes[row_idx, 1]
    ax1.imshow(res['face_crop'])
    ax1.axis('off')

    # Subplot 2: Body Crop
    ax2 = axes[row_idx, 2]
    ax2.imshow(res['body_crop'])
    ax2.axis('off')

    # Subplot 3: Masked Context
    ax3 = axes[row_idx, 3]
    ax3.imshow(res['masked_context'])
    ax3.axis('off')

plt.tight_layout()
overview_out = os.path.join(ARTIFACT_DIR, "caer_s_bounding_boxes_overview.png")
plt.savefig(overview_out, dpi=180, bbox_inches='tight')
plt.close()
print(f"\n=> Đã lưu hình tổng quan Bounding Box CAER-S tại: {overview_out}")

# 3. CAER Video Multi-Frame Interpolation Visualization
video_path = "CAER/test/Happy/0207.avi"
fv_data = json.load(open("CAER_Video/face_boxes.json")).get("test/Happy/0207", {})
bv_data = json.load(open("CAER_Video/body_boxes.json")).get("test/Happy/0207", {})

if os.path.exists(video_path) and fv_data:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=> Đang render CAER Video: {video_path} (Tổng số frame: {total_frames})")

    # Frames 0, 1, 2, 3 (Keyframes at 0 and 3; Interpolated at 1 and 2)
    frames_to_show = [0, 1, 2, 3]
    video_figs, v_axes = plt.subplots(1, 4, figsize=(20, 5.5))

    avail_f_indices = sorted([int(k.split('.')[0]) for k in fv_data.keys() if k.replace('.jpg', '').isdigit()])
    avail_b_indices = sorted([int(k.split('.')[0]) for k in bv_data.keys() if k.replace('.jpg', '').isdigit()])

    for idx, p in enumerate(frames_to_show):
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw, _ = frame_rgb.shape

        # Face interpolation logic
        is_f_key = f"{p}.jpg" in fv_data or str(p) in fv_data
        lowers = [k for k in avail_f_indices if k <= p]
        uppers = [k for k in avail_f_indices if k >= p]
        f_box = None
        if lowers and uppers:
            il, iu = lowers[-1], uppers[0]
            bl = fv_data.get(f"{il}.jpg", fv_data.get(str(il)))
            bu = fv_data.get(f"{iu}.jpg", fv_data.get(str(iu)))
            if bl and bu:
                if il == iu:
                    f_box = bl
                else:
                    alpha = (p - il) / float(iu - il)
                    f_box = [bl[i] * (1.0 - alpha) + bu[i] * alpha for i in range(4)]

        # Body interpolation logic
        is_b_key = f"{p}.jpg" in bv_data or str(p) in bv_data
        b_lowers = [k for k in avail_b_indices if k <= p]
        b_uppers = [k for k in avail_b_indices if k >= p]
        b_box = None
        if b_lowers and b_uppers:
            il, iu = b_lowers[-1], b_uppers[0]
            bl = bv_data.get(f"{il}.jpg", bv_data.get(str(il)))
            bu = bv_data.get(f"{iu}.jpg", bv_data.get(str(iu)))
            if bl and bu:
                if il == iu:
                    b_box = bl
                else:
                    alpha = (p - il) / float(iu - il)
                    b_box = [bl[i] * (1.0 - alpha) + bu[i] * alpha for i in range(4)]

        ax = v_axes[idx]
        ax.imshow(frame_rgb)

        tag = "Keyframe" if is_f_key else "Linear Interpolated"
        title_color = "#00E5FF" if is_f_key else "#FF9100"
        ax.set_title(f"Frame #{p} ({tag})", fontsize=12, fontweight='bold', color=title_color, pad=8)

        if f_box:
            fx1, fy1, fx2, fy2 = [int(v) for v in f_box[:4]]
            rect = patches.Rectangle((fx1, fy1), fx2 - fx1, fy2 - fy1, linewidth=2.5, edgecolor='#00E5FF', facecolor='none')
            ax.add_patch(rect)
            ax.text(fx1, max(0, fy1 - 5), f"Face {'[Key]' if is_f_key else '[Interp]'}",
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='#00E5FF', edgecolor='none', alpha=0.8, pad=1.5))

        if b_box:
            bx1, by1, bx2, by2 = [int(v) for v in b_box[:4]]
            rect = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, linewidth=2.5, edgecolor='#00E676', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(bx1, min(fh - 4, by2 + 14), f"Body {'[Key]' if is_b_key else '[Interp]'}",
                    color='black', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='#00E676', edgecolor='none', alpha=0.8, pad=1.5))

        ax.axis('off')

    cap.release()
    plt.tight_layout()
    video_out = os.path.join(ARTIFACT_DIR, "caer_video_interpolation_frames.png")
    plt.savefig(video_out, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"=> Đã lưu hình nội suy Bounding Box CAER Video tại: {video_out}")

print("\n=> Hoàn tất quá trình trực quan hóa bounding boxes!")
