#!/usr/bin/env python3
"""
Preprocess and generate high-precision, temporally-smoothed Body Bounding Boxes
for CAER / CAER-S datasets based on face bounding box geometry and human anatomy.

Usage:
  python tools/generate_caer_body_boxes.py \
    --face-json /kaggle/input/datasets/bearmn/caers-dataset-rapt-clip/face.json \
    --output /kaggle/working/caer_body_boxes.json
"""

import argparse
import json
import os
import tqdm


def generate_body_boxes(face_json_path, output_path, ema_alpha=0.6):
    print(f"=> Loading face boxes from: {face_json_path}")
    with open(face_json_path, 'r') as f:
        face_data = json.load(f)

    print(f"=> Processing {len(face_data)} videos...")
    body_data = {}

    for video_key, frames_dict in tqdm.tqdm(face_data.items(), desc="Generating Body BBoxes"):
        if not isinstance(frames_dict, dict):
            # Flat dict format: {"image_path": [x1, y1, x2, y2]}
            box = frames_dict
            if isinstance(box, list) and len(box) == 4:
                fx1, fy1, fx2, fy2 = box
                w = fx2 - fx1
                h = fy2 - fy1
                bx1 = max(0, int(fx1 - 0.7 * w))
                by1 = max(0, int(fy1 - 0.1 * h))
                bx2 = int(fx2 + 0.7 * w)
                by2 = int(fy1 + 3.2 * h)
                body_data[video_key] = [bx1, by1, bx2, by2]
            continue

        # Video format: {"video_key": {"0.jpg": [x1, y1, x2, y2], ...}}
        video_body_dict = {}
        # Sort frame keys by frame index
        try:
            sorted_frame_keys = sorted(frames_dict.keys(), key=lambda k: int(k.split('.')[0]))
        except:
            sorted_frame_keys = sorted(frames_dict.keys())

        prev_box = None
        for fk in sorted_frame_keys:
            raw_box = frames_dict[fk]
            if raw_box is None or not (isinstance(raw_box, list) and len(raw_box) == 4):
                if prev_box is not None:
                    video_body_dict[fk] = prev_box
                continue

            fx1, fy1, fx2, fy2 = raw_box
            w = fx2 - fx1
            h = fy2 - fy1
            bx1 = max(0, int(fx1 - 0.7 * w))
            by1 = max(0, int(fy1 - 0.1 * h))
            bx2 = int(fx2 + 0.7 * w)
            by2 = int(fy1 + 3.2 * h)
            curr_box = [bx1, by1, bx2, by2]

            # Temporal EMA smoothing to eliminate inter-frame bounding box jitter
            if prev_box is not None:
                smoothed_box = [
                    int(ema_alpha * curr_box[i] + (1.0 - ema_alpha) * prev_box[i])
                    for i in range(4)
                ]
            else:
                smoothed_box = curr_box

            video_body_dict[fk] = smoothed_box
            prev_box = smoothed_box

        # Store with both original key and normalized key (without CAER/ prefix)
        body_data[video_key] = video_body_dict
        if video_key.startswith('CAER/'):
            clean_key = video_key[len('CAER/'):]
            body_data[clean_key] = video_body_dict

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"=> Writing generated body boxes to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(body_data, f)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"=> Successfully generated body boxes! File size: {file_size_mb:.2f} MB ({len(body_data)} keys)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and generate CAER body bounding boxes")
    parser.add_argument('--face-json', type=str, default='CAER/face.json', help='Path to face.json')
    parser.add_argument('--output', type=str, default='CAER/body.json', help='Output path for body bounding boxes')
    parser.add_argument('--ema-alpha', type=float, default=0.6, help='Smoothing factor across consecutive video frames')
    args = parser.parse_args()

    generate_body_boxes(args.face_json, args.output, ema_alpha=args.ema_alpha)
