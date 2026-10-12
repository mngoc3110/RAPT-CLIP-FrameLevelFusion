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

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size, bounding_box_face, bounding_box_body, crop_body=False, root_dir="", num_classes=8, dataset_name="", mask_context_face=False, daisee_merge_low=False):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self.crop_body = crop_body
        self.root_dir = root_dir
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
        else:
            # print(f"Warning: Face bounding box file not found or None ({self.bounding_box_face}). Proceeding without face boxes.")
            self.boxs = {}

    def _read_body_boxes(self):
        if self.bounding_box_body and os.path.exists(self.bounding_box_body):
            with open(self.bounding_box_body, 'r') as f:
                self.body_boxes = json.load(f)
        else:
            # print(f"Warning: Body bounding box file not found or None ({self.bounding_box_body}). Proceeding without body boxes.")
            self.body_boxes = {}


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
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
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
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if not parts or not parts[0]:
                    continue
                if len(parts) == 2:
                    # Case: [path, label] (e.g., EMOTIC / CAER-S images train.txt)
                    p_path = parts[0]
                    if self.dataset_name == 'DAiSEE' and p_path.startswith('DAiSEE_data/'):
                        if not os.path.exists(os.path.join(self.root_dir, p_path)):
                            p_path = p_path[len('DAiSEE_data/'):]
                    elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S'] and p_path.startswith('CAER/'):
                        if not os.path.exists(os.path.join(self.root_dir, p_path)):
                            p_path = p_path[len('CAER/'):]
                    self.sample_list.append([p_path, 1, parts[1]])
                elif len(parts) == 6 and ',' in parts[1]:
                    # Case: [path, label, x1, y1, x2, y2] (e.g., EMOTIC train_bbox.txt)
                    self.sample_list.append([parts[0], 1, parts[1], parts[2], parts[3], parts[4], parts[5]])
                elif len(parts) > 3:
                    # Path contains spaces, join all parts except the last two
                    p_path = ' '.join(parts[:-2])
                    num_frames = parts[-2]
                    label = parts[-1]
                    if self.dataset_name == 'DAiSEE' and p_path.startswith('DAiSEE_data/'):
                        if not os.path.exists(os.path.join(self.root_dir, p_path)):
                            p_path = p_path[len('DAiSEE_data/'):]
                    elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S'] and p_path.startswith('CAER/'):
                        if not os.path.exists(os.path.join(self.root_dir, p_path)):
                            p_path = p_path[len('CAER/'):]
                    self.sample_list.append([p_path, num_frames, label])
                else:
                    # Case: [path, num_frames, label] (e.g., standard RAER / DAiSEE / CAER video)
                    p_path = parts[0]
                    num_frames = parts[1]
                    label = parts[2]
                    if self.dataset_name == 'DAiSEE' and p_path.startswith('DAiSEE_data/'):
                        if not os.path.exists(os.path.join(self.root_dir, p_path)):
                            p_path = p_path[len('DAiSEE_data/'):]
                    elif self.dataset_name in ['CAER', 'CAER-S', 'CAER_S'] and p_path.startswith('CAER/'):
                        if not os.path.exists(os.path.join(self.root_dir, p_path)):
                            p_path = p_path[len('CAER/'):]
                    self.sample_list.append([p_path, num_frames, label])

    def _parse_list(self):
        # 
        # Data Form: [video_id, num_frames, class_idx]
        # 
        self.video_list = [VideoRecord([os.path.join(self.root_dir, item[0])] + item[1:]) for item in self.sample_list]
        print(('video number:%d' % (len(self.video_list))))

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
            # 0. Check if path needs DAiSEE_data prefix toggled
            if not os.path.exists(record.path):
                if 'DAiSEE_data' not in record.path:
                    cand = record.path.replace('DataSet', 'DAiSEE_data/DataSet')
                    if os.path.exists(cand):
                        record._data[0] = cand
                elif '/DAiSEE_data/DAiSEE_data/' in record.path:
                    cand = record.path.replace('/DAiSEE_data/DAiSEE_data/', '/DAiSEE_data/')
                    if os.path.exists(cand):
                        record._data[0] = cand

            # 1. If path ends with 'frames' but does not exist, look for .avi / .mp4 in clip folder
            if record.path.endswith('frames') and not os.path.exists(record.path):
                clip_dir = os.path.dirname(record.path)
                clip_id = os.path.basename(clip_dir)
                for ext in ['.avi', '.mp4', '.mkv', '.AVI', '.MP4']:
                    fallback_video = os.path.join(clip_dir, f"{clip_id}{ext}")
                    if os.path.exists(fallback_video):
                        record._data[0] = fallback_video
                        break
            
            # 2. If path is a video file that does not exist, look for 'frames' subdirectory
            if (record.path.endswith(('.avi', '.mp4', '.mkv', '.AVI', '.MP4'))) and not os.path.exists(record.path):
                frames_dir = os.path.join(os.path.dirname(record.path), 'frames')
                if os.path.isdir(frames_dir):
                    record._data[0] = frames_dir

            # 3. Now evaluate the resolved path
            if os.path.isdir(record.path):
                video_frames_path = glob.glob(os.path.join(record.path, '*'))
                video_frames_path.sort()
                num_real_frames = len(video_frames_path)
                is_video_file = False
                is_valid = num_real_frames > 0
            elif os.path.exists(record.path):
                is_video_file = True
                cap = cv2.VideoCapture(record.path)
                num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                is_valid = cap.isOpened() and num_real_frames > 0
        else:
            # Assume it's a video file
            is_video_file = True
            cap = cv2.VideoCapture(record.path)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            is_valid = cap.isOpened() and num_real_frames > 0

        if not is_valid:
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
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_pil = None
                box = None
                
                # 1. Read Image
                if is_image_file:
                    try:
                        img_pil = Image.open(record.path).convert('RGB')
                    except Exception as e:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
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
                    else:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
                else:
                    img_path = video_frames_path[p]
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                    except:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))

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
                                    avail_indices = [int(k.split('.')[0]) for k in v_box_entry.keys() if k.replace('.jpg', '').isdigit()]
                                    if avail_indices:
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
                                    nearest_5 = f"{round(p / 5) * 5}.jpg"
                                    if nearest_5 in v_body_dict:
                                        body_box = v_body_dict[nearest_5]
                                    else:
                                        try:
                                            avail_indices = [int(k.split('.')[0]) for k in v_body_dict.keys() if '.' in k or k.isdigit()]
                                            if avail_indices:
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
                    left, upper, right, lower = body_box
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
    if dataset_name == "RAER":
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            GroupRandomGrayscale(p=0.2),
            RandomRotation(8),               # 8° (more head tilt variation)
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupRandomErasing(p=0.3, scale=(0.02, 0.15))  # occlusion simulation
        ])
    elif dataset_name in ["CAER", "CAER-S", "CAER_S"]:
        # CAER-S: Static Context-Aware Emotion dataset
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            GroupRandomGrayscale(p=0.15),
            RandomRotation(8),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupRandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    elif dataset_name == "DAiSEE":
        # DAiSEE: E-learning engagement videos (subtle lighting shifts, head tilt)
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            RandomRotation(6),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupRandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    elif dataset_name == "EMOTIC":
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            GroupRandomGrayscale(p=0.1),
            RandomRotation(10),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            GroupRandomErasing(p=0.4, scale=(0.02, 0.2))
        ])
    else:
        # Default transforms for other datasets like CK+
        train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
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
    if dataset_name == "EMOTIC":
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