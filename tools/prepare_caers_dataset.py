#!/usr/bin/env python3
"""
Preprocess and prepare CAER-S static image annotations for RAPT-CLIP training.
Converts Google Colab CSV paths (e.g. /content/drive/.../CAER-S/test/Angry/1001.png)
to valid Kaggle paths (e.g. /kaggle/input/datasets/lcngtr/caer-s/test/test/Angry/1001.png)
and generates:
  - train.txt, validation.txt, test.txt
  - face.json
  - body.json
  - (Optional) clean converted CSV files with Kaggle paths
"""

import argparse
import ast
import json
import os
import glob
import re
import csv
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

LABEL_MAP = {
    'anger': 0, 'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3, 'happiness': 3,
    'neutral': 4, 'neutrality': 4,
    'sad': 5, 'sadness': 5,
    'surprise': 6
}

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CAER-S dataset for RAPT-CLIP")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory of CAER-S dataset (e.g., /kaggle/input/datasets/lcngtr/caer-s)")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/caers_data",
                        help="Directory to save generated txt, json, and converted csv files")
    return parser.parse_args()

def extract_class_from_filename_or_path(csv_p, row, raw_name):
    """Trích xuất tên class kể cả khi CSV thiếu cột class"""
    for col in ['class', 'Emotions', 'emotion', 'label']:
        if col in row and row[col] is not None:
            val = str(row[col]).strip()
            if val and val.lower() != 'nan':
                return val
    
    # Thử trích xuất từ tên file CSV (ví dụ bbox_test_Angry.csv -> Angry)
    m = re.search(r'bbox_[a-zA-Z]+_([a-zA-Z]+)\.csv', os.path.basename(csv_p), re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    
    # Thử trích xuất từ đường dẫn ảnh Colab (ví dụ /.../Angry/1001.png)
    parts = raw_name.replace('\\', '/').split('/')
    if len(parts) >= 2:
        candidate = parts[-2]
        if candidate.lower() in LABEL_MAP:
            return candidate.capitalize()
            
    return None

def find_csv_files(data_dir, split_name):
    aliases = [split_name]
    if split_name.lower() in ['val', 'valid', 'validation']:
        aliases = ['valid', 'val', 'validation']
    elif split_name.lower() in ['train', 'training']:
        aliases = ['train', 'training']
    elif split_name.lower() in ['test', 'testing']:
        aliases = ['test', 'testing']

    files = []
    for s in aliases:
        for sub in aliases:
            patterns = [
                os.path.join(data_dir, s, s, f"bbox_{sub}_*.csv"),
                os.path.join(data_dir, s, f"bbox_{sub}_*.csv"),
                os.path.join(data_dir, s, s, f"bbox_*.csv"),
                os.path.join(data_dir, s, f"bbox_*.csv"),
                os.path.join(data_dir, f"bbox_{sub}_*.csv"),
                os.path.join(data_dir, f"bbox_*.csv"),
            ]
            for p in patterns:
                matched = glob.glob(p)
                if matched:
                    files.extend(matched)
        if files:
            break

    if not files:
        # Fallback recursive search
        for s in aliases:
            cand = glob.glob(os.path.join(data_dir, f"**/*{s}*/**/bbox_*.csv"), recursive=True)
            if not cand:
                cand = glob.glob(os.path.join(data_dir, f"**/bbox_{s}_*.csv"), recursive=True)
            if cand:
                files = cand
                break

    return sorted(list(set(files)))

import time

_DIR_CACHE = {}

def get_class_dir(data_dir, split_name, class_name, sample_filename=None):
    """
    Xác định thư mục tương đối chứa ảnh của (split, class) một lần duy nhất (O(1) cached).
    Giảm số lần gọi stat/os.path.exists trên Kaggle từ 1.5 triệu lần xuống < 20 lần!
    """
    cache_key = (split_name.lower(), class_name.lower())
    if cache_key in _DIR_CACHE:
        return _DIR_CACHE[cache_key]

    cls_candidates = [class_name]
    low_c = class_name.lower()
    if low_c in ['angry', 'anger']:
        cls_candidates.extend(['Angry', 'Anger', 'angry', 'anger'])
    elif low_c in ['happy', 'happiness']:
        cls_candidates.extend(['Happy', 'Happiness', 'happy', 'happiness'])
    elif low_c in ['sad', 'sadness']:
        cls_candidates.extend(['Sad', 'Sadness', 'sad', 'sadness'])
    elif low_c in ['neutral', 'neutrality']:
        cls_candidates.extend(['Neutral', 'Neutrality', 'neutral', 'neutrality'])
    elif low_c in ['surprise', 'surprised']:
        cls_candidates.extend(['Surprise', 'surprise', 'surprised'])
    elif low_c in ['disgust', 'disgusted']:
        cls_candidates.extend(['Disgust', 'disgust'])
    elif low_c in ['fear', 'fearful']:
        cls_candidates.extend(['Fear', 'fear'])

    cls_candidates = list(dict.fromkeys(cls_candidates))

    split_candidates = [split_name]
    low_s = split_name.lower()
    if low_s in ['val', 'valid', 'validation']:
        split_candidates = ['valid', 'val', 'validation']
    elif low_s in ['train', 'training']:
        split_candidates = ['train', 'training']
    elif low_s in ['test', 'testing']:
        split_candidates = ['test', 'testing']
    split_candidates = list(dict.fromkeys(split_candidates))

    for s in split_candidates:
        for c in cls_candidates:
            candidates = [
                os.path.join(s, s, c),
                os.path.join(s, c),
                os.path.join('CAER-S', s, s, c),
                os.path.join('CAER-S', s, c),
            ]
            for rel in candidates:
                full = os.path.join(data_dir, rel)
                if os.path.isdir(full):
                    # Nếu có file mẫu, kiểm tra file mẫu có tồn tại không
                    if sample_filename:
                        if os.path.exists(os.path.join(full, sample_filename)):
                            _DIR_CACHE[cache_key] = rel
                            return rel
                    else:
                        _DIR_CACHE[cache_key] = rel
                        return rel

    # Fallback nếu cấu trúc phẳng không có thư mục con:
    if sample_filename:
        for s in split_candidates:
            for c in cls_candidates:
                candidates = [
                    os.path.join(s, s, c, sample_filename),
                    os.path.join(s, c, sample_filename),
                    os.path.join('CAER-S', s, s, c, sample_filename),
                    os.path.join('CAER-S', s, c, sample_filename),
                    os.path.join(s, s, sample_filename),
                    os.path.join(s, sample_filename),
                ]
                for rel_file in candidates:
                    if os.path.exists(os.path.join(data_dir, rel_file)):
                        rel_dir = os.path.dirname(rel_file)
                        _DIR_CACHE[cache_key] = rel_dir
                        return rel_dir

    _DIR_CACHE[cache_key] = None
    return None

def find_image_path(data_dir, split_name, class_name, filename):
    rel_dir = get_class_dir(data_dir, split_name, class_name, sample_filename=filename)
    if rel_dir:
        return os.path.join(rel_dir, filename)
    return None

def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    converted_csv_dir = os.path.join(output_dir, "converted_csvs")
    os.makedirs(converted_csv_dir, exist_ok=True)

    print("=" * 70)
    print("=> CAER-S DATASET PREPROCESSOR (Colab Path -> Kaggle Path Converter)")
    print(f"=> Kaggle Dataset Root: {data_dir}")
    print(f"=> Target Output Dir:   {output_dir}")
    print("=" * 70)

    all_face_boxes = {}
    all_body_boxes = {}

    splits = [('train', 'train.txt'), ('valid', 'validation.txt'), ('test', 'test.txt')]
    verified_sample = None

    for split_key, out_txt_name in splits:
        csv_files = find_csv_files(data_dir, split_key)

        if not csv_files:
            print(f"=> [WARNING] No CSV files found for split '{split_key}'! Skipping.")
            continue

        print(f"\n=> Đang xử lý tập: {split_key.upper()} ({len(csv_files)} file CSV tìm thấy)")
        split_records = []

        for idx, csv_p in enumerate(sorted(csv_files), 1):
            t_file_start = time.time()
            csv_name = os.path.basename(csv_p)
            print(f"   [{idx}/{len(csv_files)}] Đang đọc {csv_name}...", end="", flush=True)

            converted_rows = []
            rows_added = 0
            with open(csv_p, mode='r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                bbox_col = None
                for col in ['box', 'bbox', 'bounding_box', 'coordinates']:
                    if col in fieldnames:
                        bbox_col = col
                        break

                for row in reader:
                    # Lấy tên file gốc từ đường dẫn Colab
                    raw_name = str(row.get('image_name') or row.get('image_path') or list(row.values())[0])
                    filename = os.path.basename(raw_name)

                    raw_cls = extract_class_from_filename_or_path(csv_p, row, raw_name)
                    if not raw_cls:
                        continue

                    label_id = LABEL_MAP.get(raw_cls.lower())
                    if label_id is None:
                        continue

                    # Tìm đường dẫn thực tế trên Kaggle (O(1) cached)
                    rel_img_path = find_image_path(data_dir, split_key, raw_cls, filename)
                    if not rel_img_path:
                        # Fallback tìm kiếm trực tiếp trong split
                        for cand in [os.path.join(split_key, split_key, filename), os.path.join(split_key, filename)]:
                            if os.path.exists(os.path.join(data_dir, cand)):
                                rel_img_path = cand
                                break

                    if not rel_img_path:
                        continue

                    full_kaggle_path = os.path.join(data_dir, rel_img_path)
                    if verified_sample is None:
                        verified_sample = (raw_name, full_kaggle_path)

                    # Dòng lưu vào train.txt / validation.txt / test.txt
                    split_records.append(f"{rel_img_path} {label_id}")
                    rows_added += 1

                    # Bounding box
                    if bbox_col and row.get(bbox_col):
                        val = row[bbox_col]
                        try:
                            box = ast.literal_eval(val) if isinstance(val, str) else list(val)
                            if isinstance(box, list) and len(box) == 4:
                                fx1, fy1, fx2, fy2 = [int(v) for v in box]
                                parsed_box = [fx1, fy1, fx2, fy2]
                                
                                # Lưu nhiều dạng keys để đảm bảo match_rate 100% trong Dataloader
                                for k in [rel_img_path, filename, os.path.splitext(rel_img_path)[0], os.path.splitext(filename)[0]]:
                                    all_face_boxes[k] = parsed_box

                                # Ước lượng Body Bounding Box dựa trên tỷ lệ giải phẫu học
                                w = fx2 - fx1
                                h = fy2 - fy1
                                bx1 = max(0, int(fx1 - 0.7 * w))
                                by1 = max(0, int(fy1 - 0.1 * h))
                                bx2 = int(fx2 + 0.7 * w)
                                by2 = int(fy1 + 3.2 * h)
                                body_box = [bx1, by1, bx2, by2]
                                for k in [rel_img_path, filename, os.path.splitext(rel_img_path)[0], os.path.splitext(filename)[0]]:
                                    all_body_boxes[k] = body_box
                        except Exception:
                            pass

                    # Lưu vào danh sách CSV đã convert đường dẫn
                    new_row = dict(row)
                    new_row['image_path_colab'] = raw_name
                    new_row['image_path'] = full_kaggle_path
                    new_row['image_name'] = filename
                    new_row['class'] = raw_cls
                    converted_rows.append(new_row)

            # Lưu lại CSV chuẩn hóa đường dẫn Kaggle
            if converted_rows:
                out_csv_p = os.path.join(converted_csv_dir, os.path.basename(csv_p))
                with open(out_csv_p, mode='w', encoding='utf-8', newline='') as f_out:
                    writer = csv.DictWriter(f_out, fieldnames=list(converted_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(converted_rows)

            t_elapsed = time.time() - t_file_start
            print(f" -> {rows_added:,} mẫu ({t_elapsed:.2f}s)")

        out_txt_path = os.path.join(output_dir, out_txt_name)
        with open(out_txt_path, 'w') as f:
            for rec in split_records:
                f.write(f"{rec}\n")
        print(f"   ✓ Đã tạo {out_txt_name}: {len(split_records):,} mẫu -> {out_txt_path}")

    # Đảm bảo validation.txt luôn tồn tại (CAER-S thường chỉ có train và test)
    val_txt_path = os.path.join(output_dir, "validation.txt")
    test_txt_path = os.path.join(output_dir, "test.txt")
    if (not os.path.exists(val_txt_path) or os.path.getsize(val_txt_path) == 0) and os.path.exists(test_txt_path):
        import shutil
        print(f"\n=> [THÔNG TIN] Dataset CAER-S không có tập val riêng biệt. Tự động sử dụng test.txt làm validation.txt...")
        shutil.copy(test_txt_path, val_txt_path)
        print(f"   ✓ Đã tạo validation.txt (sao chép từ {test_txt_path})")

    # Ghi face.json và body.json (Không dùng indent=2 để ghi cực nhanh và tiết kiệm dung lượng)
    face_json_path = os.path.join(output_dir, "face.json")
    print(f"\n=> Đang lưu {face_json_path}...", end="", flush=True)
    with open(face_json_path, 'w') as f:
        json.dump(all_face_boxes, f)
    print(f" xong! ({len(all_face_boxes):,} keys)")

    body_json_path = os.path.join(output_dir, "body.json")
    print(f"=> Đang lưu {body_json_path}...", end="", flush=True)
    with open(body_json_path, 'w') as f:
        json.dump(all_body_boxes, f)
    print(f" xong! ({len(all_body_boxes):,} keys)")

    print("\n" + "=" * 70)
    print("=> XÁC NHẬN CHUYỂN ĐỔI ĐƯỜNG DẪN THÀNH CÔNG:")
    if verified_sample:
        colab_p, kaggle_p = verified_sample
        print(f"   [Colab gốc]: {colab_p}")
        print(f"   [Kaggle mới]: {kaggle_p}")
        print(f"   [Tệp tồn tại trên đĩa]: {'✓ CÓ (TRUE)' if os.path.exists(kaggle_p) else '✗ KHÔNG'}")
    print(f"=> Tổng Face BBoxes lưu: {len(all_face_boxes):,}")
    print(f"=> Tổng Body BBoxes lưu: {len(all_body_boxes):,}")
    print(f"=> Các CSV đã convert:   {converted_csv_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
