import os
import glob
import json
import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils import data
from numpy.random import randint
from dataloader.video_transform import *

# Custom Transform for List of Images (Group Transform)
class GroupRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            # Convert to Grayscale (L) then back to RGB to keep 3 channels
            return [img.convert('L').convert('RGB') for img in img_group]
        return img_group

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        # Support multi-label (e.g. '1,4,12')
        lbl_str = str(self._data[2])
        if ',' in lbl_str:
            return [int(x.strip()) for x in lbl_str.split(',') if x.strip()]
        return int(lbl_str)
        
    @property
    def inline_bbox(self):
        # Returns [x1, y1, x2, y2] if available
        if len(self._data) >= 7:
            return [int(self._data[3]), int(self._data[4]), int(self._data[5]), int(self._data[6])]
        return None

class EmoticNpyDataset(data.Dataset):
    def __init__(self, npy_dir, mode='train', transform=None, num_classes=26):
        self.mode = mode
        self.transform = transform
        
        # Load npy files (using mmap_mode='r' for images to avoid loading gigabytes into RAM)
        print(f"Loading {mode} NPY files from {npy_dir}...")
        self.context_arr = np.load(os.path.join(npy_dir, f'{mode}_context_arr.npy'), mmap_mode='r')
        self.body_arr = np.load(os.path.join(npy_dir, f'{mode}_body_arr.npy'), mmap_mode='r')
        self.cat_arr = np.load(os.path.join(npy_dir, f'{mode}_cat_arr.npy'))
        
    def __len__(self):
        return len(self.cat_arr)
        
    def __getitem__(self, index):
        context_img = self.context_arr[index]
        body_img = self.body_arr[index]
        
        # Convert to PIL Image for transforms (arrays are RGB (H, W, 3))
        context_pil = Image.fromarray(context_img)
        body_pil = Image.fromarray(body_img)
        
        # Artificial face crop (60% center crop from context) to satisfy Triple-Stream
        w, h = context_pil.size
        face_pil = context_pil.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))

        # Synchronized Spatial Augmentations across streams (Train mode only)
        if self.mode == 'train':
            if random.random() < 0.5:
                context_pil = context_pil.transpose(Image.FLIP_LEFT_RIGHT)
                body_pil = body_pil.transpose(Image.FLIP_LEFT_RIGHT)
                face_pil = face_pil.transpose(Image.FLIP_LEFT_RIGHT)
            rot = random.uniform(-10, 10)
            if abs(rot) > 0.5:
                context_pil = context_pil.rotate(rot, resample=Image.BILINEAR)
                body_pil = body_pil.rotate(rot, resample=Image.BILINEAR)
                face_pil = face_pil.rotate(rot, resample=Image.BILINEAR)

        if self.transform:
            # Transform expects a list of PIL Images (for video segments)
            # We wrap single images in a list, then reshape back to (T, C, H, W) where T=1
            context_tensor = self.transform([context_pil]).view(-1, 3, 224, 224)
            body_tensor = self.transform([body_pil]).view(-1, 3, 224, 224)
            face_tensor = self.transform([face_pil]).view(-1, 3, 224, 224)
        else:
            context_tensor = torch.zeros((1, 3, 224, 224))
            body_tensor = torch.zeros((1, 3, 224, 224))
            face_tensor = torch.zeros((1, 3, 224, 224))
            
        target_tensor = torch.tensor(self.cat_arr[index], dtype=torch.float32)
        
        return face_tensor, body_tensor, context_tensor, target_tensor


def resolve_daisee_path(raw_path, root_dir, cached_rule=None):
    """
    Robustly resolves a DAiSEE video or frame directory from an annotation line.
    Handles missing 'DataSet/', '.avi'/'.mp4' extensions, case differences,
    and nested 'DAiSEE_data' folders on Kaggle/local environments.
    Returns (resolved_path, updated_rule).
    """
    clean_p = raw_path.replace('\\', '/').strip('/')
    while clean_p.startswith(('DAiSEE_data/', './DAiSEE_data/', 'DAiSEE/')):
        if clean_p.startswith('DAiSEE_data/'):
            clean_p = clean_p[len('DAiSEE_data/'):]
        elif clean_p.startswith('./DAiSEE_data/'):
            clean_p = clean_p[len('./DAiSEE_data/'):]
        elif clean_p.startswith('DAiSEE/'):
            clean_p = clean_p[len('DAiSEE/'):]

    parts = [p for p in clean_p.split('/') if p and p != '.']
    if parts and parts[-1] == 'frames':
        parts = parts[:-1]
    clip_id = parts[-1] if parts else ''
    user_id = parts[-2] if len(parts) >= 2 else ''
    split_name = parts[-3] if len(parts) >= 3 else ''

    # Fast path: check cached rule first (O(1) without repeated disk stat)
    if cached_rule is not None and clip_id:
        cand = cached_rule(split_name, user_id, clip_id)
        return cand, cached_rule

    clean_root = root_dir.rstrip('/')
    while clean_root.endswith('/DAiSEE_data') or clean_root.endswith('/DAiSEE'):
        if clean_root.endswith('/DAiSEE_data'):
            clean_root = clean_root[:-len('/DAiSEE_data')]
        elif clean_root.endswith('/DAiSEE'):
            clean_root = clean_root[:-len('/DAiSEE')]

    search_roots = [
        root_dir,
        os.path.join(root_dir, 'DataSet'),
        clean_root,
        os.path.join(clean_root, 'DataSet'),
        os.path.join(clean_root, 'DAiSEE_data'),
        os.path.join(clean_root, 'DAiSEE_data', 'DataSet'),
    ]
    if os.path.isdir('/kaggle/input'):
        for kd in [
            root_dir,
            '/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE',
            '/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE/DataSet',
            '/kaggle/input/datasets/olgaparfenova/daisee',
            '/kaggle/input/daisee/DAiSEE',
            '/kaggle/input/daisee/DAiSEE_data',
            '/kaggle/input/DAiSEE_data',
            '/kaggle/input/DAiSEE',
        ]:
            if os.path.isdir(kd) and kd not in search_roots:
                search_roots.append(kd)
                search_roots.append(os.path.join(kd, 'DataSet'))

    splits = [split_name]
    if split_name.lower() in ['validation', 'val']:
        splits = ['Validation', 'validation', 'Val', 'val']
    elif split_name.lower() == 'train':
        splits = ['Train', 'train']
    elif split_name.lower() == 'test':
        splits = ['Test', 'test']

    exts = ['.avi', '.mp4', '.mkv', '.AVI', '.MP4']

    for r in search_roots:
        if not os.path.exists(r):
            continue
        for s in splits:
            for sub in ['', 'DataSet/', 'DAiSEE_data/', 'DAiSEE_data/DataSet/']:
                base_dir = os.path.normpath(os.path.join(r, sub, s, user_id, clip_id))
                # 1. Frames dir
                cand_frames = os.path.join(base_dir, 'frames')
                if os.path.isdir(cand_frames) and len(os.listdir(cand_frames)) > 0:
                    base_prefix = os.path.normpath(os.path.join(r, sub))
                    def make_rule(prefix=base_prefix, split_choice=s):
                        return lambda sp, uid, cid: os.path.join(prefix, split_choice if sp.lower() == split_choice.lower() else sp, uid, cid, 'frames')
                    return cand_frames, make_rule()

                # 2. Video file
                for ext in exts:
                    cand_video = os.path.join(base_dir, f"{clip_id}{ext}")
                    if os.path.isfile(cand_video):
                        base_prefix = os.path.normpath(os.path.join(r, sub))
                        def make_rule(prefix=base_prefix, split_choice=s, e=ext):
                            return lambda sp, uid, cid: os.path.join(prefix, split_choice if sp.lower() == split_choice.lower() else sp, uid, cid, f"{cid}{e}")
                        return cand_video, make_rule()

                # 3. Direct base_dir if it exists
                if os.path.isdir(base_dir):
                    files = os.listdir(base_dir)
                    for f in files:
                        if f.lower().endswith(('.avi', '.mp4', '.mkv')):
                            cand_video = os.path.join(base_dir, f)
                            ext_found = os.path.splitext(f)[1]
                            base_prefix = os.path.normpath(os.path.join(r, sub))
                            def make_rule(prefix=base_prefix, split_choice=s, e=ext_found):
                                return lambda sp, uid, cid: os.path.join(prefix, split_choice if sp.lower() == split_choice.lower() else sp, uid, cid, f"{cid}{e}")
                            return cand_video, make_rule()
                    img_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                    if len(img_files) > 0:
                        base_prefix = os.path.normpath(os.path.join(r, sub))
                        def make_rule(prefix=base_prefix, split_choice=s):
                            return lambda sp, uid, cid: os.path.join(prefix, split_choice if sp.lower() == split_choice.lower() else sp, uid, cid)
                        return base_dir, make_rule()

    # Fallback search using os.walk if Sample 0 was not located in common candidates
    if clip_id:
        search_dirs = [root_dir, clean_root]
        if os.path.exists('/kaggle/input'):
            search_dirs.append('/kaggle/input')
        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            for w_root, dirs, files in os.walk(s_dir):
                for f in files:
                    if f.startswith(clip_id) and f.lower().endswith(('.avi', '.mp4', '.mkv')):
                        found_v = os.path.join(w_root, f)
                        return found_v, None
                if clip_id in dirs:
                    cand_c_dir = os.path.join(w_root, clip_id)
                    try:
                        c_files = os.listdir(cand_c_dir)
                        for f in c_files:
                            if f.lower().endswith(('.avi', '.mp4', '.mkv')):
                                return os.path.join(cand_c_dir, f), None
                            if f == 'frames':
                                return os.path.join(cand_c_dir, 'frames'), None
                        return cand_c_dir, None
                    except Exception:
                        pass

    # If truly not found, return default path
    return os.path.join(root_dir, raw_path), None


def normalize_colab_path(p):
    if not p or not isinstance(p, str):
        return p
    if os.path.exists(p):
        return p
    if p.startswith('content/'):
        alt = '/' + p
        if os.path.exists(alt):
            return alt
    elif p.startswith('drive/'):
        alt = '/content/' + p
        if os.path.exists(alt):
            return alt
    return p


def resolve_caer_path(raw_path, root_dir, cached_rule=None):
    """
    Robustly resolves a CAER video or frame directory from an annotation line.
    Handles nested 'CAER/', case differences (Anger vs anger), extensions (.avi vs .mp4 vs directory),
    and Kaggle directory structure variations.
    """
    root_dir = normalize_colab_path(root_dir)
    clean_p = raw_path.replace('\\', '/').strip('/')
    while clean_p.startswith(('CAER/', './CAER/')):
        clean_p = clean_p[len('CAER/'):] if clean_p.startswith('CAER/') else clean_p[len('./CAER/'):]

    parts = [p for p in clean_p.split('/') if p and p != '.']
    if len(parts) >= 3:
        split_name, emotion, filename = parts[-3], parts[-2], parts[-1]
    elif len(parts) == 2:
        split_name, emotion, filename = '', parts[-2], parts[-1]
    else:
        return os.path.join(root_dir, raw_path), None

    base_name = os.path.splitext(filename)[0]

    if cached_rule is not None:
        cand = cached_rule(split_name, emotion, base_name)
        if os.path.exists(cand) or os.path.isdir(cand):
            return cand, cached_rule

    clean_root = root_dir.rstrip('/')
    cand_roots = [
        root_dir,
        os.path.join(root_dir, 'CAER'),
        clean_root,
        os.path.join(clean_root, 'CAER'),
        os.path.dirname(clean_root),
        os.path.join(os.path.dirname(clean_root), 'CAER'),
        '/kaggle/input/caer-video-dataset/CAER',
        '/kaggle/input/caer-video-dataset',
        '/kaggle/input/caer-dataset/CAER',
        '/kaggle/input/caer-dataset',
        '/kaggle/input/caer-video/CAER',
        '/kaggle/input/caer-video',
        '/kaggle/input/caer/CAER',
        '/kaggle/input/caer'
    ]
    if os.path.exists('/kaggle/input'):
        for root_cand in glob.glob('/kaggle/input/**/train', recursive=True):
            cand_roots.append(os.path.dirname(root_cand))

    splits = [split_name, split_name.lower(), split_name.capitalize()]
    emotions = [emotion, emotion.lower(), emotion.capitalize()]
    exts = ['.avi', '.mp4', '.mkv', '', '.AVI', '.MP4']

    for r in cand_roots:
        if not os.path.exists(r):
            continue
        for s in splits:
            for e in emotions:
                for ext in exts:
                    cand = os.path.join(r, s, e, base_name + ext) if s else os.path.join(r, e, base_name + ext)
                    if os.path.exists(cand) or os.path.isdir(cand):
                        def make_rule(r_f=r, s_f=s, e_f=e, ext_f=ext):
                            return lambda sp, emo, bn: os.path.join(r_f, s_f if sp.lower() == s_f.lower() else sp, e_f if emo.lower() == e_f.lower() else emo, bn + ext_f)
                        return cand, make_rule()

    # Search with os.walk if not found in standard paths
    if base_name and os.path.exists(root_dir):
        for w_root, dirs, files in os.walk(root_dir):
            for f in files:
                if os.path.splitext(f)[0] == base_name:
                    return os.path.join(w_root, f), None
            if base_name in dirs:
                return os.path.join(w_root, base_name), None

    return os.path.join(root_dir, raw_path), None


class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, bounding_box_face, bounding_box_body, crop_body=False, root_dir="", num_classes=8, dataset_name="", mask_context_face=False, daisee_merge_low=False):
        self.list_file = normalize_colab_path(list_file)
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = normalize_colab_path(bounding_box_face)
        self.bounding_box_body = normalize_colab_path(bounding_box_body)
        self.crop_body = crop_body
        self.root_dir = normalize_colab_path(root_dir)
        self.dataset_name = dataset_name
        self.mask_context_face = mask_context_face
        self.daisee_merge_low = daisee_merge_low
        
        # Debugging: Initialize for saving sample images
        self.debug_samples_path = 'debug_samples'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        if self.crop_body: # Only read body boxes if cropping is enabled
            self._read_body_boxes()

    def _read_boxs(self):
        if self.bounding_box_face and os.path.exists(self.bounding_box_face):
            with open(self.bounding_box_face, 'r') as f:
                self.boxs = json.load(f)
            # Match verification on sample_list
            match_cnt = 0
            check_subset = self.sample_list[:500] if len(self.sample_list) > 500 else self.sample_list
            for item in check_subset:
                rel = item[0].replace('./', '').replace('\\', '/')
                clean_rel = rel[len('CAER/'):] if rel.startswith('CAER/') else rel
                no_ext = os.path.splitext(clean_rel)[0]
                if no_ext in self.boxs or clean_rel in self.boxs or f"CAER/{no_ext}" in self.boxs or f"CAER/{clean_rel}" in self.boxs:
                    match_cnt += 1
            rate = (match_cnt / len(check_subset) * 100) if check_subset else 0
            print(f"[{self.dataset_name} {self.mode}] Loaded {len(self.boxs)} Face boxes from '{os.path.basename(self.bounding_box_face)}' | Match rate: {rate:.1f}%")
        else:
            self.boxs = {}
            print(f"[{self.dataset_name} {self.mode}] Warning: Face bounding box file not found or None ({self.bounding_box_face}).")

    def _read_body_boxes(self):
        if self.bounding_box_body and os.path.exists(self.bounding_box_body):
            with open(self.bounding_box_body, 'r') as f:
                self.body_boxes = json.load(f)
            # Match verification on sample_list
            match_cnt = 0
            check_subset = self.sample_list[:500] if len(self.sample_list) > 500 else self.sample_list
            for item in check_subset:
                rel = item[0].replace('./', '').replace('\\', '/')
                clean_rel = rel[len('CAER/'):] if rel.startswith('CAER/') else rel
                no_ext = os.path.splitext(clean_rel)[0]
                if no_ext in self.body_boxes or clean_rel in self.body_boxes or f"CAER/{no_ext}" in self.body_boxes or f"CAER/{clean_rel}" in self.body_boxes:
                    match_cnt += 1
            rate = (match_cnt / len(check_subset) * 100) if check_subset else 0
            print(f"[{self.dataset_name} {self.mode}] Loaded {len(self.body_boxes)} Body boxes from '{os.path.basename(self.bounding_box_body)}' | Match rate: {rate:.1f}%")
        else:
            self.body_boxes = {}
            print(f"[{self.dataset_name} {self.mode}] Warning: Body bounding box file not found or None ({self.bounding_box_body}).")


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self, img, box, margin=10, mode='face'):
        if box is None:
            if mode == 'face':
                if self.dataset_name == 'DAiSEE':
                    # DAiSEE Center Crop (50%): Student facing webcam, focus on eye gaze, eyebrows, and mouth
                    w, h = img.size
                    return img.crop((int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.75)))
                elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S']:
                    # CAER-S fallback when face box is missing: Center crop 50%
                    w, h = img.size
                    return img.crop((int(w * 0.25), int(h * 0.15), int(w * 0.75), int(h * 0.75)))
                elif self.dataset_name == 'EMOTIC':
                    # Fallback for EMOTIC: Center crop 60% to avoid copying full context
                    w, h = img.size
                    return img.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
                # Default FALLBACK: Return original image instead of black image if no face detected
                return img
            return img
        else:
            if isinstance(box, (list, tuple)) and len(box) > 0 and isinstance(box[0], (list, tuple)):
                box = box[0]
            left, upper, right, lower = box[:4]
            # Safety check: auto-scale normalized [0, 1] coordinates
            if 0.0 <= left <= 1.0 and 0.0 <= right <= 1.0 and right > left and img.width > 10:
                left = left * img.width
                upper = upper * img.height
                right = right * img.width
                lower = lower * img.height
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
            if self.dataset_name in ['CAER', 'CAER-S', 'CAER_S']:
                bw = right - left
                bh = lower - upper
                pad_w = int(bw * 0.15)
                pad_h = int(bh * 0.15)
                left = max(0, left - pad_w)
                upper = max(0, upper - pad_h)
                right = min(img.width, right + pad_w)
                lower = min(img.height, lower + pad_h)
            else:
                left = max(0, left - margin)
                upper = max(0, upper - margin)
                right = min(img.width, right + margin)
                lower = min(img.height, lower + margin)
            if mode == 'face':
                if right > left and lower > upper:
                    return img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image
    
    def _read_sample(self):
        self.sample_list = []
        if not os.path.exists(self.list_file):
            print(f"[{self.dataset_name} {self.mode}] Warning: list_file not found: {self.list_file}")
            return

        with open(self.list_file, 'r') as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        if not raw_lines:
            return

        # 1. Support DAiSEE direct CSV file (e.g., TrainLabels.csv, ValidationLabels.csv)
        if self.dataset_name == 'DAiSEE' and (self.list_file.lower().endswith('.csv') or (raw_lines and ',' in raw_lines[0])):
            for line in raw_lines:
                cols = [p.strip() for p in line.split(',') if p.strip()]
                if not cols or cols[0].lower().startswith(('clip', 'video', 'id')):
                    continue
                cid = cols[0]
                eng = int(cols[3]) if len(cols) >= 4 else int(cols[1])
                self.sample_list.append([cid, 300, eng])
            return

        # 2. Support DAiSEE raw text files with no inline label (e.g., Validation.txt, Train.txt)
        first_tokens = raw_lines[0].split()
        if self.dataset_name == 'DAiSEE' and len(first_tokens) == 1:
            label_dict = {}
            base_fname = os.path.basename(self.list_file).lower()
            split_prefix = "Validation" if "val" in base_fname else ("Test" if "test" in base_fname else "Train")
            
            cand_csv_paths = [
                os.path.join(self.root_dir, 'Labels', f'{split_prefix}Labels.csv'),
                os.path.join(os.path.dirname(self.list_file), 'Labels', f'{split_prefix}Labels.csv'),
                os.path.join(os.path.dirname(os.path.dirname(self.list_file)), 'Labels', f'{split_prefix}Labels.csv'),
                os.path.join(self.root_dir, f'{split_prefix}Labels.csv'),
                f'/kaggle/input/datasets/olgaparfenova/daisee/DAiSEE/Labels/{split_prefix}Labels.csv'
            ]
            
            found_csv = None
            for cp in cand_csv_paths:
                if os.path.exists(cp):
                    found_csv = cp
                    break
            
            if found_csv is None and os.path.exists('/kaggle/input'):
                for match in glob.glob(f'/kaggle/input/**/{split_prefix}Labels.csv', recursive=True):
                    found_csv = match
                    break

            if found_csv and os.path.exists(found_csv):
                print(f"[{self.dataset_name} {self.mode}] Auto-loaded labels from companion CSV: {found_csv}")
                with open(found_csv, 'r') as cf:
                    for cline in cf:
                        cols = [p.strip() for p in cline.split(',') if p.strip()]
                        if not cols or cols[0].lower().startswith(('clip', 'video', 'id')):
                            continue
                        c_id = os.path.splitext(cols[0])[0]
                        c_eng = int(cols[3]) if len(cols) >= 4 else int(cols[1])
                        label_dict[c_id] = c_eng
                        label_dict[cols[0]] = c_eng
            else:
                print(f"[{self.dataset_name} {self.mode}] Notice: Companion {split_prefix}Labels.csv not found. Defaulting labels to 0.")

            for line in raw_lines:
                p_path = line.strip()
                clip_base = os.path.splitext(os.path.basename(p_path))[0]
                lbl = label_dict.get(clip_base, label_dict.get(p_path, 0))
                self.sample_list.append([p_path, 300, lbl])
            return

        # 3. Standard multi-column space-separated parsing
        for line in raw_lines:
            parts = line.split(' ')
            if not parts or not parts[0]:
                continue
            if len(parts) == 2:
                # Case: [path, label]
                p_path = parts[0]
                if self.dataset_name == 'DAiSEE' and p_path.startswith('DAiSEE_data/'):
                    p_path = p_path[len('DAiSEE_data/'):]
                elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S'] and p_path.startswith('CAER/'):
                    p_path = p_path[len('CAER/'):]
                self.sample_list.append([p_path, 1, parts[1]])
            elif len(parts) == 6 and ',' in parts[1]:
                self.sample_list.append([parts[0], 1, parts[1], parts[2], parts[3], parts[4], parts[5]])
            elif len(parts) > 3:
                p_path = ' '.join(parts[:-2])
                num_frames = parts[-2]
                label = parts[-1]
                if self.dataset_name == 'DAiSEE' and p_path.startswith('DAiSEE_data/'):
                    p_path = p_path[len('DAiSEE_data/'):]
                elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S'] and p_path.startswith('CAER/'):
                    p_path = p_path[len('CAER/'):]
                self.sample_list.append([p_path, num_frames, label])
            else:
                p_path = parts[0]
                num_frames = parts[1]
                label = parts[2]
                if self.dataset_name == 'DAiSEE' and p_path.startswith('DAiSEE_data/'):
                    p_path = p_path[len('DAiSEE_data/'):]
                elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S'] and p_path.startswith('CAER/'):
                    p_path = p_path[len('CAER/'):]
                self.sample_list.append([p_path, num_frames, label])

    def _parse_list(self):
        # 
        # Data Form: [video_id, num_frames, class_idx]
        # 
        if self.dataset_name == 'DAiSEE':
            # Pre-resolve DAiSEE paths to actual existing videos/frames on disk
            total_items = len(self.sample_list)
            if total_items > 50:
                print(f"[{self.dataset_name} {self.mode}] 📂 Bắt đầu nạp & kiểm tra {total_items} video từ root: '{self.root_dir}'...", flush=True)
            cached_rule = None
            resolved_samples = []
            for idx, item in enumerate(self.sample_list):
                rel_p = item[0]
                full_cand = os.path.join(self.root_dir, rel_p)
                if os.path.exists(full_cand) or os.path.isdir(full_cand):
                    resolved_samples.append([full_cand] + item[1:])
                else:
                    real_p, new_rule = resolve_daisee_path(rel_p, self.root_dir, cached_rule)
                    if new_rule is not None:
                        cached_rule = new_rule
                    resolved_samples.append([real_p] + item[1:])
                
                # In % tiến độ mỗi 500 mẫu
                if (idx + 1) % 500 == 0 or (idx + 1) == total_items:
                    pct = (idx + 1) / total_items * 100.0
                    print(f"   ⏳ [{self.mode.upper()}] Tiến độ: {idx + 1:4d}/{total_items} video ({pct:5.1f}%)...", flush=True)

            self.video_list = [VideoRecord(item) for item in resolved_samples]
            if len(self.video_list) > 50:
                print(f"[{self.dataset_name} {self.mode}] ✓ Đã nạp thành công {len(self.video_list)} samples.", flush=True)
        elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S', 'CAER_Video', 'CAER-Video']:
            # Pre-resolve CAER paths to handle nested directories, casing, and extensions
            cached_rule = None
            resolved_samples = []
            for item in self.sample_list:
                rel_p = item[0]
                full_cand = os.path.join(self.root_dir, rel_p)
                if os.path.exists(full_cand) or os.path.isdir(full_cand):
                    resolved_samples.append([full_cand] + item[1:])
                else:
                    real_p, new_rule = resolve_caer_path(rel_p, self.root_dir, cached_rule)
                    if new_rule is not None:
                        cached_rule = new_rule
                    resolved_samples.append([real_p] + item[1:])
            self.video_list = [VideoRecord(item) for item in resolved_samples]
        elif self.dataset_name == 'RAER':
            resolved_samples = []
            for item in self.sample_list:
                rel_p = item[0]
                full_cand = os.path.join(self.root_dir, rel_p)
                if os.path.exists(full_cand) or os.path.isdir(full_cand):
                    resolved_samples.append([full_cand] + item[1:])
                    continue
                # Handle double RAER prefix (e.g. root_dir='.../RAER' and rel_p='RAER/images/...')
                norm_root = self.root_dir.rstrip('/\\')
                if (rel_p.startswith('RAER/') or rel_p.startswith('RAER\\')) and (norm_root.endswith('/RAER') or norm_root.endswith('\\RAER') or norm_root == 'RAER'):
                    sub_p = rel_p[5:]
                    cand2 = os.path.join(norm_root, sub_p)
                    if os.path.exists(cand2) or os.path.isdir(cand2):
                        resolved_samples.append([cand2] + item[1:])
                        continue
                # Try parent directory if root_dir ends with RAER
                parent_dir = os.path.dirname(norm_root) if os.path.dirname(norm_root) else '.'
                cand3 = os.path.join(parent_dir, rel_p)
                if os.path.exists(cand3) or os.path.isdir(cand3):
                    resolved_samples.append([cand3] + item[1:])
                    continue
                # Strip duplicate /RAER/RAER/ substring
                clean_full = full_cand.replace('/RAER/RAER/', '/RAER/').replace('\\RAER\\RAER\\', '\\RAER\\')
                if os.path.exists(clean_full) or os.path.isdir(clean_full):
                    resolved_samples.append([clean_full] + item[1:])
                    continue
                resolved_samples.append([full_cand] + item[1:])
            self.video_list = [VideoRecord(item) for item in resolved_samples]
        else:
            self.video_list = [VideoRecord([os.path.join(self.root_dir, item[0])] + item[1:]) for item in self.sample_list]

        print(('video number:%d' % (len(self.video_list))))
        if self.video_list:
            s0_path = self.video_list[0].path
            s0_exists = os.path.exists(s0_path) or os.path.isdir(s0_path)
            print(f"[{self.dataset_name} {self.mode}] Sample 0 path: '{s0_path}' | exists: {s0_exists}")

        # Automatically determine label offset:
        # If min integer label is >= 1 (e.g., RAER 1..5 or CAER video 1..7), label_offset = 1
        # If min integer label is 0 (e.g., CAER-S 0..6 or DAiSEE 0..3), label_offset = 0
        int_labels = []
        for item in self.sample_list:
            lbl_val = item[-1]
            if isinstance(lbl_val, str) and ',' not in lbl_val:
                try:
                    int_labels.append(int(lbl_val))
                except ValueError:
                    pass
            elif isinstance(lbl_val, int):
                int_labels.append(lbl_val)
        if int_labels:
            min_l = min(int_labels)
            self.label_offset = 1 if min_l >= 1 else 0
        else:
            self.label_offset = 0
        print(f"[{self.dataset_name}] Detected label_offset: {self.label_offset} (min integer label in split: {min(int_labels) if 'int_labels' in locals() and int_labels else 'N/A'})")

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        # 
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        # 
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        # Check if record.path is a directory (frames) or a file (video/image)
        IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        is_image_file = record.path.lower().endswith(IMAGE_EXTS)
        is_video_file = False
        is_valid = False
        num_real_frames = 0
        cap = None
        video_frames_path = []

        if is_image_file:
            # Static image (CAER-S / EMOTIC): read directly with PIL, treat as single frame
            is_video_file = False
            num_real_frames = 1
            is_valid = os.path.exists(record.path)
        elif os.path.isdir(record.path):
            video_frames_path = glob.glob(os.path.join(record.path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
            is_video_file = False
            is_valid = num_real_frames > 0
        elif self.dataset_name == 'DAiSEE':
            # Robust DAiSEE video/frame resolution
            if not os.path.exists(record.path) and not os.path.isdir(record.path):
                real_p, _ = resolve_daisee_path(record.path, self.root_dir)
                if os.path.exists(real_p) or os.path.isdir(real_p):
                    record._data[0] = real_p
        elif self.dataset_name == 'RAER':
            # Fix duplicate /RAER/RAER/ prefix if present
            if not os.path.exists(record.path) and not os.path.isdir(record.path):
                if '/RAER/RAER/' in record.path or '\\RAER\\RAER\\' in record.path:
                    fixed_p = record.path.replace('/RAER/RAER/', '/RAER/').replace('\\RAER\\RAER\\', '\\RAER\\')
                    if os.path.exists(fixed_p) or os.path.isdir(fixed_p):
                        record._data[0] = fixed_p

            # Now evaluate the resolved path
            if os.path.isdir(record.path):
                video_frames_path = glob.glob(os.path.join(record.path, '*'))
                video_frames_path.sort()
                num_real_frames = len(video_frames_path)
                is_video_file = False
                is_valid = num_real_frames > 0
            elif os.path.exists(record.path):
                is_video_file = True
                cap = cv2.VideoCapture(record.path)
                num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
                is_valid = cap.isOpened() and num_real_frames > 0
            else:
                is_video_file = True
                is_valid = False
        else:
            # Assume it's a video file
            is_video_file = True
            cap = cv2.VideoCapture(record.path)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            is_valid = cap.isOpened() and num_real_frames > 0

        if not is_valid:
            if not hasattr(self, '_missing_logged'):
                self._missing_logged = 0
            if self._missing_logged < 5:
                print(f"\n[CRITICAL ERROR] [{self.dataset_name} {self.mode}] Video NOT FOUND or FAILED to open: '{record.path}'! Returning dummy zero tensors! Please verify dataset path.", flush=True)
                self._missing_logged += 1
            dummy_shape = (self.num_segments, 3, self.image_size, self.image_size)
            if self.dataset_name == "EMOTIC":
                multi_hot = torch.zeros(26)
                if isinstance(record.label, list):
                    for l in record.label:
                        multi_hot[l] = 1.0
                else:
                    multi_hot[record.label] = 1.0
                return torch.zeros(dummy_shape), torch.zeros(dummy_shape), torch.zeros(dummy_shape), multi_hot
            else:
                raw_label = record.label - self.label_offset
                if self.daisee_merge_low and self.dataset_name == 'DAiSEE':
                    raw_label = 0 if raw_label <= 1 else (1 if raw_label == 2 else 2)
                return torch.zeros(dummy_shape), torch.zeros(dummy_shape), torch.zeros(dummy_shape), raw_label

        # Clamp indices to be valid
        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = list()
        images_face = list()
        images_context = list()
        
        current_frame_idx = -1
        last_valid_img = None
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_pil = None
                box = None
                
                # 1. Read Image
                if is_image_file:
                    try:
                        img_pil = Image.open(record.path).convert('RGB')
                        last_valid_img = img_pil
                    except Exception as e:
                        img_pil = last_valid_img if last_valid_img is not None else Image.new('RGB', (self.image_size, self.image_size))
                elif is_video_file:
                    gap = p - current_frame_idx
                    if gap == 1:
                        ret, frame = cap.read()
                    elif 1 < gap <= 5:
                        for _ in range(gap - 1):
                            cap.grab()
                        ret, frame = cap.read()
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                        ret, frame = cap.read()
                    current_frame_idx = p

                    if ret:
                        img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                        last_valid_img = img_pil
                    else:
                        img_pil = last_valid_img if last_valid_img is not None else Image.new('RGB', (self.image_size, self.image_size))
                else:
                    img_path = video_frames_path[p]
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                        last_valid_img = img_pil
                    except:
                        img_pil = last_valid_img if last_valid_img is not None else Image.new('RGB', (self.image_size, self.image_size))

                # 2. Key Lookup Strategy for Bounding Box
                rel_path = record.path.replace('./', '').replace('\\', '/')
                if self.root_dir and rel_path.startswith(self.root_dir.replace('\\', '/')):
                    rel_key = rel_path[len(self.root_dir.rstrip('/')) + 1:]
                else:
                    rel_key = rel_path
                    parts_full = rel_key.split('/')
                    for start in range(1, len(parts_full)):
                        candidate = '/'.join(parts_full[start:])
                        if candidate in self.boxs:
                            rel_key = candidate
                            break

                box = None
                if is_image_file:
                    # Flat JSON: {"train/Angry/1769.png": [x1,y1,x2,y2]} or {"mscoco/xxx.jpg": [x1,y1,x2,y2]}
                    if rel_key in self.boxs:
                        raw = self.boxs[rel_key]
                        box = raw if isinstance(raw[0], (int, float)) else raw[0]
                    else:
                        rel_key_parts = rel_key.split('/')
                        for start in range(1, len(rel_key_parts)):
                            candidate = '/'.join(rel_key_parts[start:])
                            if candidate in self.boxs:
                                raw = self.boxs[candidate]
                                box = raw if isinstance(raw[0], (int, float)) else raw[0]
                                break
                    if box is None:
                        base_fname = os.path.basename(rel_key)
                        if base_fname in self.boxs:
                            raw = self.boxs[base_fname]
                            box = raw if isinstance(raw[0], (int, float)) else raw[0]
                        else:
                            base_no_ext = os.path.splitext(base_fname)[0]
                            if base_no_ext in self.boxs:
                                raw = self.boxs[base_no_ext]
                                box = raw if isinstance(raw[0], (int, float)) else raw[0]
                else:
                    # Video / Frames nested or flat JSON
                    video_key_full = os.path.splitext(rel_key)[0]
                    clip_basename = os.path.basename(video_key_full)
                    if clip_basename == 'frames':
                        clip_basename = os.path.basename(os.path.dirname(video_key_full))

                    frame_key = f"{p}.jpg"
                    matched_video_key = None
                    candidates = [
                        video_key_full,
                        rel_key,
                        clip_basename,
                        f"CAER/{video_key_full}",
                        f"DAiSEE/{video_key_full}"
                    ]
                    if video_key_full.startswith('CAER/'):
                        candidates.append(video_key_full[5:])
                    if video_key_full.startswith('DAiSEE/'):
                        candidates.append(video_key_full[7:])
                    if video_key_full.startswith('DAiSEE_data/'):
                        candidates.append(video_key_full[12:])

                    for c in candidates:
                        if c in self.boxs:
                            matched_video_key = c
                            break
                    if matched_video_key is None:
                        parts = video_key_full.split('/')
                        for idx in range(1, len(parts)):
                            sub_key = '/'.join(parts[idx:])
                            if sub_key in self.boxs:
                                matched_video_key = sub_key
                                break
                            elif f"CAER/{sub_key}" in self.boxs:
                                matched_video_key = f"CAER/{sub_key}"
                                break
                            elif f"DAiSEE/{sub_key}" in self.boxs:
                                matched_video_key = f"DAiSEE/{sub_key}"
                                break

                    if matched_video_key:
                        v_box_entry = self.boxs[matched_video_key]
                        if isinstance(v_box_entry, (list, tuple)):
                            box = v_box_entry if isinstance(v_box_entry[0], (int, float)) else v_box_entry[0]
                        elif isinstance(v_box_entry, dict):
                            if frame_key in v_box_entry:
                                box = v_box_entry[frame_key]
                            elif str(p) in v_box_entry:
                                box = v_box_entry[str(p)]
                            elif 'default' in v_box_entry:
                                box = v_box_entry['default']
                            else:
                                try:
                                    avail_indices = sorted([int(k.split('.')[0]) for k in v_box_entry.keys() if k.replace('.jpg', '').isdigit()])
                                    if avail_indices:
                                        lowers = [k for k in avail_indices if k <= p]
                                        uppers = [k for k in avail_indices if k >= p]
                                        if lowers and uppers:
                                            idx_l, idx_u = lowers[-1], uppers[0]
                                            box_l = v_box_entry.get(f"{idx_l}.jpg", v_box_entry.get(str(idx_l)))
                                            box_u = v_box_entry.get(f"{idx_u}.jpg", v_box_entry.get(str(idx_u)))
                                            if box_l and box_u and len(box_l) == 4 and len(box_u) == 4:
                                                if idx_l == idx_u:
                                                    box = box_l
                                                else:
                                                    alpha = (p - idx_l) / float(idx_u - idx_l)
                                                    box = [box_l[i] * (1.0 - alpha) + box_u[i] * alpha for i in range(4)]
                                            else:
                                                closest_p = min(avail_indices, key=lambda x: abs(x - p))
                                                box = v_box_entry.get(f"{closest_p}.jpg", v_box_entry.get(str(closest_p)))
                                        else:
                                            closest_p = min(avail_indices, key=lambda x: abs(x - p))
                                            box = v_box_entry.get(f"{closest_p}.jpg", v_box_entry.get(str(closest_p)))
                                except Exception:
                                    pass
                                if box is None and len(v_box_entry) > 0:
                                    box = next(iter(v_box_entry.values()))

                # 3. Face Detection (Crop)
                img_pil_face = self._face_detect(img_pil, box, margin=10, mode='face')

                # 4. Body Stream Preprocessing
                img_pil_body = img_pil
                body_box = None
                if self.crop_body:
                    if is_image_file:
                        if rel_key in self.body_boxes:
                            raw = self.body_boxes[rel_key]
                            body_box = raw if isinstance(raw[0], (int, float)) else raw[0]
                        else:
                            rel_key_parts = rel_key.split('/')
                            for start in range(1, len(rel_key_parts)):
                                candidate = '/'.join(rel_key_parts[start:])
                                if candidate in self.body_boxes:
                                    raw = self.body_boxes[candidate]
                                    body_box = raw if isinstance(raw[0], (int, float)) else raw[0]
                                    break
                            if body_box is None:
                                base_fname = os.path.basename(rel_key)
                                if base_fname in self.body_boxes:
                                    raw = self.body_boxes[base_fname]
                                    body_box = raw if isinstance(raw[0], (int, float)) else raw[0]
                                else:
                                    base_no_ext = os.path.splitext(base_fname)[0]
                                    if base_no_ext in self.body_boxes:
                                        raw = self.body_boxes[base_no_ext]
                                        body_box = raw if isinstance(raw[0], (int, float)) else raw[0]
                    else:
                        matched_body_key = None
                        b_candidates = [
                            video_key_full,
                            rel_key,
                            clip_basename,
                            f"CAER/{video_key_full}",
                            f"DAiSEE/{video_key_full}"
                        ]
                        if video_key_full.startswith('CAER/'):
                            b_candidates.append(video_key_full[5:])
                        if video_key_full.startswith('DAiSEE/'):
                            b_candidates.append(video_key_full[7:])
                        if video_key_full.startswith('DAiSEE_data/'):
                            b_candidates.append(video_key_full[12:])
                        for bc in b_candidates:
                            if bc in self.body_boxes:
                                matched_body_key = bc
                                break
                        if matched_body_key is None:
                            parts = video_key_full.split('/')
                            for idx in range(1, len(parts)):
                                sub_key = '/'.join(parts[idx:])
                                if sub_key in self.body_boxes:
                                    matched_body_key = sub_key
                                    break
                                elif f"CAER/{sub_key}" in self.body_boxes:
                                    matched_body_key = f"CAER/{sub_key}"
                                    break
                                elif f"DAiSEE/{sub_key}" in self.body_boxes:
                                    matched_body_key = f"DAiSEE/{sub_key}"
                                    break
                        
                        if matched_body_key:
                            v_body_dict = self.body_boxes[matched_body_key]
                            if isinstance(v_body_dict, (list, tuple)):
                                body_box = v_body_dict if isinstance(v_body_dict[0], (int, float)) else v_body_dict[0]
                            elif isinstance(v_body_dict, dict):
                                if frame_key in v_body_dict:
                                    body_box = v_body_dict[frame_key]
                                elif str(p) in v_body_dict:
                                    body_box = v_body_dict[str(p)]
                                elif 'default' in v_body_dict:
                                    body_box = v_body_dict['default']
                                else:
                                    try:
                                        avail_indices = sorted([int(k.split('.')[0]) for k in v_body_dict.keys() if '.' in k or k.isdigit()])
                                        if avail_indices:
                                            lowers = [k for k in avail_indices if k <= p]
                                            uppers = [k for k in avail_indices if k >= p]
                                            if lowers and uppers:
                                                idx_l, idx_u = lowers[-1], uppers[0]
                                                b_l = v_body_dict.get(f"{idx_l}.jpg", v_body_dict.get(str(idx_l)))
                                                b_u = v_body_dict.get(f"{idx_u}.jpg", v_body_dict.get(str(idx_u)))
                                                if b_l and b_u and len(b_l) == 4 and len(b_u) == 4:
                                                    if idx_l == idx_u:
                                                        body_box = b_l
                                                    else:
                                                        alpha = (p - idx_l) / float(idx_u - idx_l)
                                                        body_box = [b_l[i] * (1.0 - alpha) + b_u[i] * alpha for i in range(4)]
                                                else:
                                                    closest_p = min(avail_indices, key=lambda x: abs(x - p))
                                                    body_box = v_body_dict.get(f"{closest_p}.jpg", v_body_dict.get(str(closest_p)))
                                            else:
                                                closest_p = min(avail_indices, key=lambda x: abs(x - p))
                                                body_box = v_body_dict.get(f"{closest_p}.jpg", v_body_dict.get(str(closest_p)))
                                    except:
                                        pass
                                    if body_box is None and len(v_body_dict) > 0:
                                        body_box = next(iter(v_body_dict.values()))
                    
                    if body_box is None and record.inline_bbox is not None:
                        body_box = record.inline_bbox

                # Specialized crops for body/posture when body bbox is missing
                if body_box is not None:
                    if isinstance(body_box, (list, tuple)) and len(body_box) > 0 and isinstance(body_box[0], (list, tuple)):
                        body_box = body_box[0]
                    left, upper, right, lower = body_box[:4]
                    if 0.0 <= left <= 1.0 and 0.0 <= right <= 1.0 and right > left and img_pil.width > 10:
                        left = left * img_pil.width
                        upper = upper * img_pil.height
                        right = right * img_pil.width
                        lower = lower * img_pil.height
                    left = max(0, int(left)); upper = max(0, int(upper))
                    right = min(img_pil.width, int(right)); lower = min(img_pil.height, int(lower))
                    if right > left and lower > upper:
                        img_pil_body = img_pil.crop((left, upper, right, lower))
                elif self.dataset_name == 'DAiSEE':
                    # DAiSEE Upper-Body Posture Crop: Captures shoulders, head tilt, posture, and leaning
                    if box is not None:
                        fx1, fy1, fx2, fy2 = box
                        fw = fx2 - fx1
                        fh = fy2 - fy1
                        # Expand horizontally to capture both shoulders, down to chest/torso
                        bx1 = max(0, int(fx1 - fw * 0.8))
                        by1 = max(0, int(fy1))
                        bx2 = min(img_pil.width, int(fx2 + fw * 0.8))
                        by2 = min(img_pil.height, int(fy2 + fh * 2.2))
                        if bx2 > bx1 and by2 > by1:
                            img_pil_body = img_pil.crop((bx1, by1, bx2, by2))
                        else:
                            w_b, h_b = img_pil.size
                            img_pil_body = img_pil.crop((int(w_b * 0.10), int(h_b * 0.05), int(w_b * 0.90), int(h_b * 0.85)))
                    else:
                        w_b, h_b = img_pil.size
                        img_pil_body = img_pil.crop((int(w_b * 0.10), int(h_b * 0.05), int(w_b * 0.90), int(h_b * 0.85)))
                elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S']:
                    # CAER-S: If face box is available, estimate upper torso region; otherwise upper 80%
                    if box is not None:
                        fx1, fy1, fx2, fy2 = box
                        face_w = fx2 - fx1
                        face_h = fy2 - fy1
                        bx1 = max(0, int(fx1 - face_w * 0.8))
                        by1 = max(0, int(fy1))
                        bx2 = min(img_pil.width, int(fx2 + face_w * 0.8))
                        by2 = min(img_pil.height, int(fy2 + face_h * 2.5))
                        if bx2 > bx1 and by2 > by1:
                            img_pil_body = img_pil.crop((bx1, by1, bx2, by2))
                        else:
                            w_b, h_b = img_pil.size
                            img_pil_body = img_pil.crop((int(w_b * 0.10), int(h_b * 0.05), int(w_b * 0.90), int(h_b * 0.85)))
                    else:
                        w_b, h_b = img_pil.size
                        img_pil_body = img_pil.crop((int(w_b * 0.10), int(h_b * 0.05), int(w_b * 0.90), int(h_b * 0.85)))

                # 5. Context Stream Preprocessing (with Face Masking option to prevent face leakage)
                img_pil_context = img_pil
                if self.mask_context_face and box is not None:
                    img_pil_context = img_pil.copy()
                    draw = ImageDraw.Draw(img_pil_context)
                    fx1, fy1, fx2, fy2 = box
                    draw.rectangle([max(0, int(fx1) - 5), max(0, int(fy1) - 5),
                                    min(img_pil.width, int(fx2) + 5), min(img_pil.height, int(fy2) + 5)],
                                   fill=(0, 0, 0))

                # Append streams (Face, Body, Context)
                images.append(img_pil_body)
                images_face.append(img_pil_face)
                images_context.append(img_pil_context)
                
                if p < num_real_frames - 1:
                    p += 1
        
        if is_video_file and 'cap' in dir() and cap is not None:
            cap.release()

        # Synchronized Spatial Augmentations across streams (Train mode only)
        # Guarantees identical horizontal flip and rotation across Face, Body, and Context streams
        # to preserve cross-modal spatial correspondence for CMAF.
        if self.mode == 'train':
            # 1. Synchronized Random Horizontal Flip (50% probability)
            if random.random() < 0.5:
                images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
                images_face = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images_face]
                images_context = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images_context]

            # 2. Synchronized Random Small Rotation
            max_rot = 8
            if self.dataset_name == 'EMOTIC':
                max_rot = 10
            elif self.dataset_name == 'DAiSEE':
                max_rot = 6
            angle = random.uniform(-max_rot, max_rot)
            if abs(angle) > 0.5:
                images = [img.rotate(angle, resample=Image.BILINEAR) for img in images]
                images_face = [img.rotate(angle, resample=Image.BILINEAR) for img in images_face]
                images_context = [img.rotate(angle, resample=Image.BILINEAR) for img in images_context]

        # Apply transforms
        process_data = self.transform(images) # (C*T, H, W)
        process_data_face = self.transform(images_face) # (C*T, H, W)
        process_data_context = self.transform(images_context) # (C*T, H, W)

        # Reshape to (T, C, H, W)
        process_data = process_data.view(-1, 3, self.image_size, self.image_size)
        process_data_face = process_data_face.view(-1, 3, self.image_size, self.image_size)
        process_data_context = process_data_context.view(-1, 3, self.image_size, self.image_size)
        
        # Process target to multi-hot if it's a list (EMOTIC)
        if isinstance(record.label, list):
            target_tensor = torch.tensor(record.label, dtype=torch.float32)
            return process_data_face, process_data, process_data_context, target_tensor
        else:
            final_label = record.label - self.label_offset
            if self.daisee_merge_low and self.dataset_name == 'DAiSEE':
                # 3-class merge: Very Low (0) & Low (1) -> 0 (Low); High (2) -> 1; Very High (3) -> 2
                final_label = 0 if final_label <= 1 else (1 if final_label == 2 else 2)
            return process_data_face, process_data, process_data_context, final_label

    def __len__(self):
        return len(self.video_list)


def train_data_loader(root_dir, list_file, num_segments, duration, image_size, dataset_name, bounding_box_face, bounding_box_body, crop_body=False, num_classes=8, mask_context_face=False, daisee_merge_low=False):
    # Note: Spatial augmentations (Horizontal Flip & Rotation) are synchronized across streams inside VideoDataset.get()
    # to maintain strict cross-modal spatial alignment for CMAF.
    if dataset_name == "RAER":
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            GroupRandomGrayscale(p=0.2),
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupRandomErasing(p=0.3, scale=(0.02, 0.15))  # occlusion simulation
        ])
    elif dataset_name in ["CAER", "CAER-S", "CAER_S"]:
        # CAER / CAER-S: Context-Aware Emotion dataset (Normalized for CLIP ViT)
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            GroupRandomGrayscale(p=0.15),
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            GroupRandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    elif dataset_name == "DAiSEE":
        # DAiSEE: E-learning engagement videos (Normalized for CLIP ViT)
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            GroupRandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    elif dataset_name == "EMOTIC":
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            GroupRandomGrayscale(p=0.1),
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            GroupRandomErasing(p=0.4, scale=(0.02, 0.2))
        ])
    else:
        # Default transforms for other datasets like CK+
        train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor()
        ])
            
    train_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body,
                              crop_body=crop_body,
                              num_classes=num_classes,
                              dataset_name=dataset_name,
                              mask_context_face=mask_context_face,
                              daisee_merge_low=daisee_merge_low
                              )
    return train_data


def test_data_loader(root_dir, list_file, num_segments, duration, image_size, bounding_box_face, bounding_box_body, crop_body=False, num_classes=8, dataset_name="", mask_context_face=False, daisee_merge_low=False):
    if dataset_name in ["EMOTIC", "CAER", "CAER-S", "CAER_S", "DAiSEE"]:
        test_transform = torchvision.transforms.Compose([
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        test_transform = torchvision.transforms.Compose([
            GroupResize(image_size),
            Stack(),
            ToTorchFormatTensor()
        ])

    test_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body,
                             crop_body=crop_body,
                             num_classes=num_classes,
                             dataset_name=dataset_name,
                             mask_context_face=mask_context_face,
                             daisee_merge_low=daisee_merge_low
                             )
    return test_data