#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/pack_seeds_to_pth.py
──────────────────────────
Đóng gói (compress) các thư mục checkpoint đã giải nén (seed123, seed7)
trở lại thành file checkpoint định dạng chuẩn PyTorch (.pth),
xử lý lỗi timestamp < 1980 của file nội bộ, đồng thời kiểm tra tính toàn vẹn (integrity).
"""

import os
import sys
import shutil
import zipfile
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import utils
DESKTOP_DIR = "/Users/macbook/Desktop"

TARGETS = [
    ("seed123", "seed_123.pth"),
    ("seed7", "seed_7.pth")
]

def pack_folder_to_pth(folder_path, out_pth):
    print(f"\n📦 Đang nén thư mục: {folder_path} -> {out_pth}...")
    temp_pth = out_pth + ".tmp"
    
    with zipfile.ZipFile(temp_pth, 'w', compression=zipfile.ZIP_STORED) as zf_out:
        for root, dirs, files in os.walk(folder_path):
            for f in sorted(files):
                if f.startswith('._') or '__MACOSX' in root or f == '.DS_Store':
                    continue
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, folder_path)
                arcname = os.path.join('archive', rel_p)
                
                # Sửa lỗi timestamp < 1980 do giải nén macOS tạo ra
                zinfo = zipfile.ZipInfo(arcname)
                zinfo.date_time = (2026, 11, 5, 13, 0, 0)
                zinfo.compress_type = zipfile.ZIP_STORED
                
                with open(full_p, 'rb') as f_in:
                    with zf_out.open(zinfo, 'w') as dest:
                        shutil.copyfileobj(f_in, dest, length=8 * 1024 * 1024)
                        
    if os.path.exists(out_pth):
        os.remove(out_pth)
    os.rename(temp_pth, out_pth)
    
    size_mb = os.path.getsize(out_pth) / (1024 ** 2)
    print(f"✅ Đã nén xong: {out_pth} ({size_mb:.1f} MB)")
    
    # Kiểm tra tính toàn vẹn bằng torch.load
    print(f"🔍 Đang kiểm tra torch.load({os.path.basename(out_pth)})...")
    ckpt = torch.load(out_pth, map_location='cpu', weights_only=False)
    print(f"🎉 TẢI THÀNH CÔNG 100%!")
    if isinstance(ckpt, dict):
        print(f"   • Các keys: {list(ckpt.keys())}")
        if 'epoch' in ckpt:
            print(f"   • Epoch: {ckpt['epoch']}")
        if 'best_acc' in ckpt:
            print(f"   • Best Acc: {ckpt['best_acc']}")
        if 'state_dict' in ckpt:
            print(f"   • Số lượng trọng số (Layers): {len(ckpt['state_dict'])}")
    return out_pth


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU ĐÓNG GÓI CHECKPOINT SEED123 VÀ SEED7 THÀNH .PTH")
    print("=" * 65)

    created_files = []
    for folder_name, pth_name in TARGETS:
        folder_path = os.path.join(ROOT_DIR, folder_name)
        if not os.path.exists(folder_path):
            print(f"❌ Không tìm thấy thư mục: {folder_path}")
            continue

        # 1. Tạo file .pth trong thư mục project
        out_project = os.path.join(ROOT_DIR, pth_name)
        pack_folder_to_pth(folder_path, out_project)
        created_files.append(out_project)

        # 2. Copy ra Desktop để người dùng tiện sử dụng ngay
        out_desktop = os.path.join(DESKTOP_DIR, pth_name)
        shutil.copy2(out_project, out_desktop)
        print(f"📋 Đã sao chép ra Desktop: {out_desktop}")
        created_files.append(out_desktop)

    # Dọn dẹp file test nếu có
    test_file = os.path.join(ROOT_DIR, "test_seed123.pth")
    if os.path.exists(test_file):
        os.remove(test_file)

    print("\n" + "=" * 65)
    print("🏆 HOÀN TẤT TOÀN BỘ!")
    for f in created_files:
        print(f"   • {f} ({os.path.getsize(f)/(1024**2):.1f} MB)")
    print("=" * 65)


if __name__ == '__main__':
    main()
