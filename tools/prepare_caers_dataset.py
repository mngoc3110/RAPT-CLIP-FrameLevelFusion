#!/usr/bin/env python3
"""
Preprocess and prepare CAER-S static image annotations for RAPT-CLIP training.
Converts Kaggle/CSV layout (bbox_*.csv) into:
  - train.txt, validation.txt, test.txt
  - face.json
  - body.json
"""

import argparse
import ast
import json
import os
import glob
import pandas as pd
from tqdm import tqdm

LABEL_MAP = {
    'anger': 0, 'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CAER-S dataset for RAPT-CLIP")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory of CAER-S dataset (e.g., /kaggle/input/datasets/lcngtr/caer-s)")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/caers_data",
                        help="Directory to save generated txt and json files")
    return parser.parse_args()

def find_csv_files(data_dir, split_name):
    patterns = [
        os.path.join(data_dir, split_name, split_name, f"bbox_{split_name}_*.csv"),
        os.path.join(data_dir, split_name, f"bbox_{split_name}_*.csv"),
        os.path.join(data_dir, split_name, split_name, "bbox_*.csv"),
        os.path.join(data_dir, split_name, "bbox_*.csv"),
    ]
    files = []
    for p in patterns:
        files = glob.glob(p)
        if files:
            break
    if not files and split_name in ['val', 'validation']:
        alt = 'val' if split_name == 'validation' else 'validation'
        for p in [os.path.join(data_dir, alt, alt, f"bbox_{alt}_*.csv"),
                  os.path.join(data_dir, alt, f"bbox_{alt}_*.csv")]:
            files = glob.glob(p)
            if files:
                break
    return files

def find_image_path(data_dir, split_name, class_name, filename):
    candidates = [
        os.path.join(split_name, split_name, class_name, filename),
        os.path.join(split_name, class_name, filename),
    ]
    if split_name in ['val', 'validation']:
        alt = 'val' if split_name == 'validation' else 'validation'
        candidates.extend([
            os.path.join(alt, alt, class_name, filename),
            os.path.join(alt, class_name, filename),
        ])

    for rel in candidates:
        full = os.path.join(data_dir, rel)
        if os.path.exists(full):
            return rel
    return None

def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=> Input dataset root: {data_dir}")
    print(f"=> Output annotations: {output_dir}")

    all_face_boxes = {}
    all_body_boxes = {}

    splits = [('train', 'train.txt'), ('val', 'validation.txt'), ('test', 'test.txt')]

    for split_key, out_txt_name in splits:
        csv_files = find_csv_files(data_dir, split_key)
        if not csv_files and split_key == 'val':
            csv_files = find_csv_files(data_dir, 'validation')

        if not csv_files:
            print(f"=> [WARNING] No CSV files found for split '{split_key}'! Skipping.")
            continue

        print(f"\n=> Processing split: {split_key.upper()} ({len(csv_files)} CSV files found)")
        split_records = []

        for csv_p in tqdm(sorted(csv_files), desc=f"Reading {split_key} CSVs"):
            df = pd.read_csv(csv_p)
            bbox_col = None
            for col in ['box', 'bbox', 'bounding_box', 'coordinates']:
                if col in df.columns:
                    bbox_col = col
                    break

            for _, row in df.iterrows():
                raw_name = str(row['image_name'] if 'image_name' in row else row['image_path'])
                filename = os.path.basename(raw_name)
                raw_cls = str(row['class'] if 'class' in row else row['Emotions']).strip()
                label_id = LABEL_MAP.get(raw_cls.lower())
                if label_id is None:
                    continue

                rel_img_path = find_image_path(data_dir, split_key, raw_cls, filename)
                if not rel_img_path:
                    # Fallback to search without class folder
                    for cand in [os.path.join(split_key, split_key, filename), os.path.join(split_key, filename)]:
                        if os.path.exists(os.path.join(data_dir, cand)):
                            rel_img_path = cand
                            break

                if not rel_img_path:
                    continue

                split_records.append(f"{rel_img_path} {label_id}")

                # Bounding box parsing
                if bbox_col and pd.notna(row[bbox_col]):
                    val = row[bbox_col]
                    try:
                        box = ast.literal_eval(val) if isinstance(val, str) else list(val)
                        if isinstance(box, list) and len(box) == 4:
                            fx1, fy1, fx2, fy2 = [int(v) for v in box]
                            all_face_boxes[rel_img_path] = [fx1, fy1, fx2, fy2]

                            # Body box estimation
                            w = fx2 - fx1
                            h = fy2 - fy1
                            bx1 = max(0, int(fx1 - 0.7 * w))
                            by1 = max(0, int(fy1 - 0.1 * h))
                            bx2 = int(fx2 + 0.7 * w)
                            by2 = int(fy1 + 3.2 * h)
                            all_body_boxes[rel_img_path] = [bx1, by1, bx2, by2]
                    except Exception:
                        pass

        out_txt_path = os.path.join(output_dir, out_txt_name)
        with open(out_txt_path, 'w') as f:
            for rec in split_records:
                f.write(f"{rec}\n")
        print(f"   Saved {len(split_records)} samples to: {out_txt_path}")

    # Write face.json and body.json
    face_json_path = os.path.join(output_dir, "face.json")
    with open(face_json_path, 'w') as f:
        json.dump(all_face_boxes, f, indent=2)
    print(f"\n=> Saved {len(all_face_boxes)} Face bounding boxes to: {face_json_path}")

    body_json_path = os.path.join(output_dir, "body.json")
    with open(body_json_path, 'w') as f:
        json.dump(all_body_boxes, f, indent=2)
    print(f"=> Saved {len(all_body_boxes)} Body bounding boxes to: {body_json_path}")
    print("\n=> All CAER-S annotations prepared successfully!")

if __name__ == "__main__":
    main()
