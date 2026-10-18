#!/usr/bin/env python3
"""
CAER-S Path Converter & Bounding Box Verification Tool for Kaggle & Local
========================================================================
1. Chuyển đổi đường dẫn Google Colab trong CSV:
   '/content/drive/MyDrive/KG-Emo/Data/CAER-S/CAER-S/test/Angry/1001.png'
   thành đường dẫn thực tế trên Kaggle:
   '/kaggle/input/datasets/lcngtr/caer-s/test/test/Angry/1001.png' (hoặc test/Anger/1001.png)
2. Tự động kiểm tra file ảnh có tồn tại và hợp lệ không (kích thước W x H).
3. Kiểm tra tính hợp lệ của Bounding Box [x1, y1, x2, y2].
4. Vẽ Bounding Box trực tiếp lên ảnh và lưu file ảnh kiểm tra trực quan.
5. Tạo sẵn train.txt, test.txt, face.json, body.json để huấn luyện RAPT-CLIP.
"""

import os
import glob
import json
import ast
import argparse
import csv
from PIL import Image, ImageDraw, ImageFont

# Bản đồ nhãn cảm xúc và các biến thể tên thư mục (Anger vs Angry)
CLASS_ALIASES = {
    'anger': ['Angry', 'Anger', 'angry', 'anger'],
    'angry': ['Angry', 'Anger', 'angry', 'anger'],
    'disgust': ['Disgust', 'disgust'],
    'fear': ['Fear', 'fear'],
    'happy': ['Happy', 'Happiness', 'happy', 'happiness'],
    'neutral': ['Neutral', 'Neutrality', 'neutral', 'neutrality'],
    'sad': ['Sad', 'Sadness', 'sad', 'sadness'],
    'surprise': ['Surprise', 'surprise']
}

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
    default_out = "/kaggle/working/caers_verification" if os.path.exists("/kaggle/working") else "outputs/caers_verification"
    parser = argparse.ArgumentParser(description="Convert Colab paths to Kaggle and test BBoxes")
    parser.add_argument("--data-dir", type=str, default="",
                        help="Root directory of CAER-S dataset on Kaggle or local")
    parser.add_argument("--csv-file", type=str, default="",
                        help="Path to specific CSV file (e.g. .../bbox_test_Angry.csv)")
    parser.add_argument("--output-dir", type=str, default=default_out,
                        help="Output directory to save verified images and annotations")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test first few samples and visualize")
    return parser.parse_args()

def display_if_notebook(image_path):
    """Hiển thị ảnh trực tiếp trong cell của Kaggle Notebook (không cần pandas/matplotlib)."""
    try:
        from IPython.display import display, Image as IPImage
        display(IPImage(filename=image_path))
    except Exception:
        try:
            import matplotlib.pyplot as plt
            img = Image.open(image_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

def auto_detect_data_dir(custom_path=""):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        "/kaggle/input/datasets/lcngtr/caer-s",
        "/kaggle/input/caer-s",
        "/kaggle/input/datasets/bearmn/caers-dataset-rapt-clip",
        "CAER-S",
        "./CAER-S"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

def find_actual_image_path(data_dir, split_name, class_name, filename):
    """
    Tìm kiếm đường dẫn thực tế của ảnh trên Kaggle hoặc Local.
    Hỗ trợ cả trường hợp thư mục lồng nhau (test/test/Angry) hoặc đơn (test/Anger).
    """
    cls_candidates = CLASS_ALIASES.get(class_name.lower(), [class_name])
    split_candidates = [split_name]
    if split_name in ['val', 'validation']:
        split_candidates = ['val', 'validation']

    search_patterns = []
    for s in split_candidates:
        for c in cls_candidates:
            search_patterns.extend([
                os.path.join(s, s, c, filename),       # test/test/Angry/1001.png
                os.path.join(s, c, filename),          # test/Angry/1001.png hoặc test/Anger/1001.png
                os.path.join('CAER-S', s, c, filename), # CAER-S/test/Angry/1001.png
                os.path.join('CAER-S', s, s, c, filename)
            ])

    for rel in search_patterns:
        full_p = os.path.join(data_dir, rel)
        if os.path.exists(full_p):
            return rel, full_p

    # Fallback đệ quy nếu tên file tồn tại
    matches = glob.glob(os.path.join(data_dir, f"**/{filename}"), recursive=True)
    if matches:
        full_p = matches[0]
        rel = os.path.relpath(full_p, data_dir)
        return rel, full_p

    return None, None

def test_and_visualize_sample(img_path, raw_box, class_name, colab_path, save_path):
    """
    Vẽ Bounding Box lên ảnh thực tế và lưu kết quả kiểm tra.
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    x1, y1, x2, y2 = raw_box

    # Kiểm tra tính hợp lệ
    is_valid_x = (0 <= x1 < x2 <= w)
    is_valid_y = (0 <= y1 < y2 <= h)
    box_w = x2 - x1
    box_h = y2 - y1

    # Tạo bản vẽ trực quan
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)

    # 1. Vẽ Bounding Box màu cam/lime cho Person/Body từ CSV
    draw.rectangle([x1, y1, x2, y2], outline="#00FF66", width=4)

    # Label box
    label_txt = f"{class_name} BBox: [{x1}, {y1}, {x2}, {y2}] ({box_w}x{box_h})"
    draw.rectangle([x1, max(0, y1 - 25), min(w, x1 + len(label_txt) * 9), y1], fill="#00FF66")
    draw.text((x1 + 4, max(0, y1 - 22)), label_txt, fill="black")

    # 2. Tạo ảnh cắt (Crop)
    crop_img = img.crop((x1, y1, x2, y2)).resize((224, 224), Image.BILINEAR)

    # 3. Ghép ảnh so sánh (Original có Box + Cropped Region)
    combined = Image.new('RGB', (w + 240, max(h, 240)), color=(24, 24, 24))
    combined.paste(annotated, (0, 0))
    combined.paste(crop_img, (w + 8, (max(h, 240) - 224) // 2))

    draw_comb = ImageDraw.Draw(combined)
    draw_comb.text((w + 8, (max(h, 240) - 224) // 2 - 20), "ViT Crop (224x224)", fill="#00E5FF")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.save(save_path)
    display_if_notebook(save_path)

    return {
        'img_size': (w, h),
        'box': [x1, y1, x2, y2],
        'box_size': (box_w, box_h),
        'is_valid': is_valid_x and is_valid_y,
        'save_path': save_path
    }

def main():
    args = parse_args()
    data_dir = auto_detect_data_dir(args.data_dir)
    print("=" * 80)
    print("🚀 [CAER-S KAGGLE CONVERTER & BBOX TESTER]")
    print(f"   • Data Directory:   '{data_dir}' (Exists: {os.path.exists(data_dir)})")
    print(f"   • Output Directory: '{args.output_dir}'")
    print("=" * 80)

    # Tìm file CSV để kiểm tra
    csv_file = args.csv_file
    if not csv_file or not os.path.exists(csv_file):
        # Tự động tìm bbox_test_Angry.csv hoặc file CSV bất kỳ
        found_csvs = glob.glob(os.path.join(data_dir, "**/bbox_test_*.csv"), recursive=True)
        if not found_csvs:
            found_csvs = glob.glob(os.path.join(data_dir, "**/bbox_*.csv"), recursive=True)
        if found_csvs:
            csv_file = found_csvs[0]
            print(f"🔍 Tự động phát hiện CSV: '{csv_file}'")

    # Mẫu test mẫu nếu chạy độc lập
    sample_colab_path = "/content/drive/MyDrive/KG-Emo/Data/CAER-S/CAER-S/test/Angry/1001.png"
    sample_class = "Angry"
    sample_box = [245, 51, 584, 354]
    sample_filename = "1001.png"

    print("\n[BƯỚC 1] KIỂM TRA MẪU ĐƯỢC CUNG CẤP (User Sample):")
    print(f"  • Đường dẫn Colab gốc:  '{sample_colab_path}'")
    print(f"  • Bounding Box khai báo: {sample_box}")
    print(f"  • Nhãn cảm xúc:          '{sample_class}'")

    rel_path, full_path = find_actual_image_path(data_dir, "test", sample_class, sample_filename)
    if full_path and os.path.exists(full_path):
        print(f"  ✓ Đã map thành công sang Kaggle: '{full_path}'")
        res = test_and_visualize_sample(
            full_path, sample_box, sample_class, sample_colab_path,
            os.path.join(args.output_dir, "sample_1001_Angry_bbox_verified.png")
        )
        print(f"  ✓ Kích thước ảnh:        {res['img_size'][0]}x{res['img_size'][1]}")
        print(f"  ✓ Kích thước Box:        {res['box_size'][0]}x{res['box_size'][1]}")
        print(f"  ✓ Tọa độ nằm trong ảnh:  {'HỢP LỆ (TRUE)' if res['is_valid'] else 'KHÔNG HỢP LỆ (FALSE)'}")
        print(f"  ✓ Ảnh trực quan đã lưu:  '{res['save_path']}'")
    else:
        print(f"  ✗ Chưa tìm thấy file ảnh '{sample_filename}' trong '{data_dir}'. Hãy kiểm tra lại thư mục giải nén!")

    # Nếu có file CSV thực tế, chạy kiểm tra toàn bộ file CSV
    if csv_file and os.path.exists(csv_file):
        print(f"\n[BƯỚC 2] KIỂM TRA FILE CSV: '{csv_file}'")
        rows = []
        fieldnames = []
        with open(csv_file, mode='r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            col_img = 'image_name' if 'image_name' in fieldnames else ('image_path' if 'image_path' in fieldnames else fieldnames[0])
            col_box = 'box' if 'box' in fieldnames else ('bbox' if 'bbox' in fieldnames else ('bounding_box' if 'bounding_box' in fieldnames else fieldnames[-1]))
            col_cls = 'class' if 'class' in fieldnames else ('label' if 'label' in fieldnames else None)
            rows = list(reader)

        print(f"  • Tổng số dòng trong CSV: {len(rows)}")
        print(f"  • Các cột trong CSV:      {fieldnames}")

        valid_count = 0
        missing_count = 0
        invalid_box_count = 0

        for idx, row in enumerate(rows):
            raw_img_val = str(row.get(col_img, ''))
            fname = os.path.basename(raw_img_val)
            c_name = str(row.get(col_cls, sample_class)) if col_cls and row.get(col_cls) else sample_class
            box_val = row.get(col_box, '')
            try:
                b = ast.literal_eval(box_val) if isinstance(box_val, str) else list(box_val)
            except:
                b = None

            _, found_p = find_actual_image_path(data_dir, "test", c_name, fname)
            if found_p and os.path.exists(found_p):
                if b and len(b) == 4:
                    img_check = Image.open(found_p)
                    w, h = img_check.size
                    x1, y1, x2, y2 = [int(v) for v in b]
                    if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                        valid_count += 1
                        if idx < 3: # Lưu 3 ảnh mẫu đầu tiên để kiểm tra
                            test_and_visualize_sample(
                                found_p, [x1, y1, x2, y2], c_name, raw_img_val,
                                os.path.join(args.output_dir, f"csv_sample_{idx}_{fname}")
                            )
                    else:
                        invalid_box_count += 1
                else:
                    invalid_box_count += 1
            else:
                missing_count += 1

        print("\n" + "=" * 80)
        print("📊 [KẾT QUẢ KIỂM TRA FILE CSV KAGGLE]")
        print(f"  • Tổng số mẫu:               {len(rows)}")
        if len(rows) > 0:
            print(f"  • Mẫu hợp lệ 100%:           {valid_count} ({valid_count/len(rows)*100:.1f}%)")
        print(f"  • Mẫu thiếu ảnh trên đĩa:    {missing_count}")
        print(f"  • Mẫu BBox ngoài biên:       {invalid_box_count}")
        print("=" * 80)

if __name__ == "__main__":
    main()
