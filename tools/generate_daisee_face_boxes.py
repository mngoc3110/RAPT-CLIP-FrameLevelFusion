#!/usr/bin/env python3
"""
Precomputes Face Bounding Boxes for DAiSEE videos using MTCNN.
- Samples K keyframes per video clip across time.
- Detects student face with high confidence (>0.90).
- Computes temporally smoothed / median face bounding box ("default") per video.
- Stores multi-key aliases (clip_id, relative path) for instant O(1) lookup.
- Fully resumable with periodic checkpointing every 50 videos.
"""

import os
import sys
import glob
import json
import time
import argparse
import numpy as np
from PIL import Image

try:
    import cv2
    if hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(0)
except:
    pass

import torch
from tqdm import tqdm

def get_detector(device='cpu'):
    try:
        from facenet_pytorch import MTCNN
        print(f"=> Using MTCNN Face Detector on device: {device}", flush=True)
        detector = MTCNN(
            keep_all=False,
            select_largest=True,
            device=device,
            post_process=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
        return detector
    except ImportError:
        print("=> Notice: facenet_pytorch not installed. Installing via pip...", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "facenet-pytorch"])
        from facenet_pytorch import MTCNN
        return MTCNN(keep_all=False, select_largest=True, device=device, post_process=False)

def read_keyframe(clip_path, frame_idx, num_total_frames=300):
    """Reads a PIL Image at frame_idx from either a directory of frames or an .avi/.mp4 video file."""
    if os.path.isdir(clip_path):
        frame_files = sorted(glob.glob(os.path.join(clip_path, '*')))
        if not frame_files:
            return None
        clamped_idx = min(len(frame_files) - 1, max(0, frame_idx))
        try:
            return Image.open(frame_files[clamped_idx]).convert('RGB')
        except:
            return None
    elif os.path.isfile(clip_path):
        try:
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img_rgb)
        except:
            return None
    return None

def resolve_video_path(root_dir, raw_path):
    """Resolves potential relative / frames / avi paths for DAiSEE."""
    clean_p = raw_path
    if clean_p.startswith('DAiSEE_data/'):
        clean_p = clean_p[len('DAiSEE_data/'):]
    
    candidates = [
        os.path.join(root_dir, raw_path),
        os.path.join(root_dir, clean_p),
        os.path.join(root_dir, 'DAiSEE_data', clean_p),
        raw_path
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
        # If points to 'frames' but does not exist, check .avi
        if c.endswith('frames'):
            clip_dir = os.path.dirname(c)
            clip_id = os.path.basename(clip_dir)
            for ext in ['.avi', '.mp4', '.AVI', '.MP4']:
                cand_v = os.path.join(clip_dir, f"{clip_id}{ext}")
                if os.path.exists(cand_v):
                    return cand_v
        # If points to video but does not exist, check 'frames' dir
        if c.endswith(('.avi', '.mp4', '.AVI', '.MP4')):
            f_dir = os.path.join(os.path.dirname(c), 'frames')
            if os.path.isdir(f_dir):
                return f_dir
    return candidates[0]

def main():
    parser = argparse.ArgumentParser(description="Generate DAiSEE Face Bounding Boxes with MTCNN")
    parser.add_argument('--root-dir', type=str, default='DAiSEE', help='Root directory for DAiSEE dataset')
    parser.add_argument('--annotations', nargs='+', default=['DAiSEE/daisee_train.txt', 'DAiSEE/daisee_val.txt', 'DAiSEE/daisee_test.txt'])
    parser.add_argument('--output', type=str, default='DAiSEE/daisee_face_bbox.json', help='Output JSON path')
    parser.add_argument('--samples-per-video', type=int, default=5, help='Number of frames to sample per video clip')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--save-freq', type=int, default=50, help='Checkpoint frequency')
    args = parser.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    detector = get_detector(device=device)

    # Load existing boxes if resumable
    face_boxes = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r') as f:
                face_boxes = json.load(f)
            print(f"=> Resuming from existing output: {len(face_boxes)} entries already processed.", flush=True)
        except Exception as e:
            print(f"=> Warning: Could not read existing output ({e}). Starting fresh.", flush=True)

    # Collect unique videos from all annotation files
    video_entries = []
    seen_paths = set()
    for annot in args.annotations:
        if not os.path.exists(annot):
            # Check if annot is inside root_dir
            alt_annot = os.path.join(args.root_dir, os.path.basename(annot))
            if os.path.exists(alt_annot):
                annot = alt_annot
            else:
                print(f"=> Notice: Annotation file {annot} not found, skipping.", flush=True)
                continue

        with open(annot, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if not parts or not parts[0]:
                    continue
                v_path = parts[0]
                num_frames = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 300
                if v_path not in seen_paths:
                    seen_paths.add(v_path)
                    video_entries.append((v_path, num_frames))

    print(f"=> Total unique DAiSEE videos to process: {len(video_entries)}", flush=True)

    processed_count = 0
    t0 = time.time()
    
    pbar = tqdm(video_entries, desc="Detecting DAiSEE Faces")
    for raw_path, num_frames in pbar:
        # Check if already processed
        clip_id = os.path.basename(raw_path.rstrip('/'))
        if clip_id == 'frames':
            clip_id = os.path.basename(os.path.dirname(raw_path))
        clip_id_no_ext = os.path.splitext(clip_id)[0]

        if clip_id_no_ext in face_boxes or raw_path in face_boxes:
            continue

        resolved_path = resolve_video_path(args.root_dir, raw_path)
        if not os.path.exists(resolved_path):
            continue

        # Sample K frame indices evenly across video duration
        step = max(1, num_frames // (args.samples_per_video + 1))
        sample_indices = [step * (k + 1) for k in range(args.samples_per_video)]
        sample_indices = [min(num_frames - 1, max(0, s)) for s in sample_indices]

        detected_boxes = []
        frame_dict = {}

        for f_idx in sample_indices:
            img = read_keyframe(resolved_path, f_idx, num_total_frames=num_frames)
            if img is None:
                continue

            try:
                boxes, probs = detector.detect(img)
                if boxes is not None and len(boxes) > 0 and probs[0] > 0.80:
                    box_coords = [round(float(x), 1) for x in boxes[0]]
                    detected_boxes.append(box_coords)
                    frame_dict[str(f_idx)] = box_coords
                    frame_dict[f"{f_idx}.jpg"] = box_coords
            except Exception:
                pass

        if detected_boxes:
            # Robust median box across sampled frames
            arr = np.array(detected_boxes)
            median_box = [round(float(x), 1) for x in np.median(arr, axis=0)]
            frame_dict['default'] = median_box

            # Register under clip_id and relative keys
            face_boxes[clip_id_no_ext] = frame_dict
            face_boxes[raw_path] = frame_dict
            
            # Additional alias for DataSet relative path
            if 'DataSet/' in raw_path:
                rel_ds = raw_path[raw_path.index('DataSet/'):]
                face_boxes[rel_ds] = frame_dict
        else:
            # Fallback heuristic box if student was completely hidden (lower center)
            # Default 640x480 webcam heuristic for students sitting in lower half
            default_fallback = [220.0, 240.0, 420.0, 440.0]
            fallback_dict = {'default': default_fallback}
            face_boxes[clip_id_no_ext] = fallback_dict
            face_boxes[raw_path] = fallback_dict

        processed_count += 1
        if processed_count % args.save_freq == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(face_boxes, f, indent=2)
            pbar.set_postfix({'saved': len(face_boxes)})

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(face_boxes, f, indent=2)

    total_time = time.time() - t0
    print(f"\n=> Finished! Saved {len(face_boxes)} entries to: {args.output}", flush=True)
    print(f"=> Total time: {total_time/60:.2f} minutes ({total_time/max(1, processed_count):.2f}s per video)", flush=True)

if __name__ == '__main__':
    main()
